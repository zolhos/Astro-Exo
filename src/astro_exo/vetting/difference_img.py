"""
Difference imaging and in-transit centroid offset measurement on TESS Target Pixel Files (TPF).
"""

from typing import Dict, Any, Tuple
import numpy as np


def calculate_difference_image(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_days: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs out-of-transit, in-transit, and difference flux images.

    Returns
    -------
    i_out : np.ndarray
        Average out-of-transit flux image.
    i_in : np.ndarray
        Average in-transit flux image.
    i_diff : np.ndarray
        Difference image (missing flux deficit = i_out - i_in).
    sigma_diff : np.ndarray
        Propagated standard error per pixel.
    """
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period

    in_transit_mask = np.abs(phase) <= (0.5 * duration_days)
    out_transit_mask = (np.abs(phase) >= (0.75 * duration_days)) & (np.abs(phase) <= (1.75 * duration_days))

    n_in = int(np.sum(in_transit_mask))
    n_out = int(np.sum(out_transit_mask))

    if n_in < 2 or n_out < 4:
        raise ValueError(f"Insufficient cadences for difference image (n_in={n_in}, n_out={n_out})")

    i_in = np.nanmean(flux[in_transit_mask], axis=0)
    i_out = np.nanmean(flux[out_transit_mask], axis=0)
    i_diff = i_out - i_in

    var_in = np.nanmean(flux_err[in_transit_mask] ** 2, axis=0) / n_in
    var_out = np.nanmean(flux_err[out_transit_mask] ** 2, axis=0) / n_out
    sigma_diff = np.sqrt(var_in + var_out)

    return i_out, i_in, i_diff, sigma_diff


def measure_centroid_offset(
    i_diff: np.ndarray,
    sigma_diff: np.ndarray,
    target_pix_coord: Tuple[float, float],
    tess_pixel_scale_arcsec: float = 21.0,
    snr_threshold: float = 2.5,
    n_mc_perturbations: int = 500
) -> Dict[str, float]:
    """
    Calculates the first spatial moments of the difference image and
    computes offset from the target's catalog pixel coordinate with Monte Carlo error propagation.
    """
    ny, nx = i_diff.shape
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))

    snr_diff = i_diff / (sigma_diff + 1e-12)
    diff_mask = snr_diff >= snr_threshold

    if np.sum(diff_mask) < 1:
        # Fallback to entire image if no single pixel passes threshold
        diff_mask = np.ones_like(i_diff, dtype=bool)

    weights = np.clip(i_diff * diff_mask, 0.0, None)
    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        raise ValueError("No positive difference flux available to compute centroid.")

    x_diff_cen = float(np.sum(x_grid * weights) / weight_sum)
    y_diff_cen = float(np.sum(y_grid * weights) / weight_sum)

    pix_x_target, pix_y_target = target_pix_coord
    dx_pix = x_diff_cen - pix_x_target
    dy_pix = y_diff_cen - pix_y_target
    offset_pix = np.hypot(dx_pix, dy_pix)
    offset_arcsec = float(offset_pix * tess_pixel_scale_arcsec)
    dx_arcsec = float(dx_pix * tess_pixel_scale_arcsec)
    dy_arcsec = float(dy_pix * tess_pixel_scale_arcsec)

    # Monte Carlo perturbation for uncertainty
    mc_x = np.zeros(n_mc_perturbations)
    mc_y = np.zeros(n_mc_perturbations)
    rng = np.random.default_rng(42)

    for i in range(n_mc_perturbations):
        pert = i_diff + rng.normal(0.0, sigma_diff)
        p_weights = np.clip(pert * diff_mask, 0.0, None)
        p_sum = np.sum(p_weights)
        if p_sum > 0:
            mc_x[i] = np.sum(x_grid * p_weights) / p_sum
            mc_y[i] = np.sum(y_grid * p_weights) / p_sum
        else:
            mc_x[i], mc_y[i] = x_diff_cen, y_diff_cen

    sigma_x = float(np.std(mc_x))
    sigma_y = float(np.std(mc_y))
    sigma_offset_arcsec = float(np.hypot(sigma_x, sigma_y) * tess_pixel_scale_arcsec)
    significance = float(offset_arcsec / sigma_offset_arcsec) if sigma_offset_arcsec > 0 else 0.0

    return {
        "x_diff_cen": x_diff_cen,
        "y_diff_cen": y_diff_cen,
        "target_x": pix_x_target,
        "target_y": pix_y_target,
        "dx_arcsec": dx_arcsec,
        "dy_arcsec": dy_arcsec,
        "centroid_vec_arcsec": (dx_arcsec, dy_arcsec),
        "offset_pix": float(offset_pix),
        "offset_arcsec": offset_arcsec,
        "sigma_offset_arcsec": sigma_offset_arcsec,
        "offset_significance_sigma": significance
    }


def vet_target_pixel_file(
    tpf_data: Dict[str, Any],
    period: float,
    t0: float,
    duration_hours: float,
    max_allowed_offset_arcsec: float = 4.0,
    max_significance_sigma: float = 3.0
) -> Dict[str, Any]:
    """
    High-level automated vetting pipeline running difference imaging on a TPF dataset.

    Parameters
    ----------
    tpf_data : dict
        Output from read_tess_tpf(), containing 'time', 'flux', 'flux_err', 'target_pix'.
    period : float
        Orbital period in days.
    t0 : float
        Transit epoch in BTJD.
    duration_hours : float
        Transit duration in hours.
    max_allowed_offset_arcsec : float
        Maximum allowable centroid shift to consider transit on-target (default 4.0 arcsec, ~0.2 TESS pixel).
    max_significance_sigma : float
        Maximum statistical significance of the offset to rule out background eclipsing binaries (default 3.0-sigma).

    Returns
    -------
    result : dict
        Vetting metrics including difference images, centroids, offsets, and pass/fail classification.
    """
    time = tpf_data["time"]
    flux = tpf_data["flux"]
    flux_err = tpf_data["flux_err"]
    duration_days = duration_hours / 24.0

    # Determina a coordenada de referência do alvo
    target_pix = tpf_data.get("target_pix")
    if target_pix is None:
        # Fallback: centro geométrico da matriz de pixels
        ny, nx = flux.shape[1:]
        target_pix = (nx / 2.0 - 0.5, ny / 2.0 - 0.5)

    # Calcula imagens de diferença
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period,
        t0=t0,
        duration_days=duration_days
    )

    # Mede deslocamento do centróide
    offset_results = measure_centroid_offset(
        i_diff=i_diff,
        sigma_diff=sigma_diff,
        target_pix_coord=target_pix
    )

    offset_arcsec = offset_results["offset_arcsec"]
    sig_sigma = offset_results["offset_significance_sigma"]

    # Critério de aprovação: centróide centrado no alvo (< limiar angular ou insignificante estatisticamente)
    is_on_target = (offset_arcsec <= max_allowed_offset_arcsec) or (sig_sigma < max_significance_sigma)

    return {
        "passed": bool(is_on_target),
        "status": "PASS" if is_on_target else "FAIL_POSSIBLE_NEB",
        "offset_arcsec": offset_arcsec,
        "dx_arcsec": offset_results["dx_arcsec"],
        "dy_arcsec": offset_results["dy_arcsec"],
        "centroid_vec_arcsec": offset_results["centroid_vec_arcsec"],
        "offset_pix": offset_results["offset_pix"],
        "sigma_offset_arcsec": offset_results["sigma_offset_arcsec"],
        "offset_significance_sigma": sig_sigma,
        "target_pix": target_pix,
        "diff_centroid_pix": (offset_results["x_diff_cen"], offset_results["y_diff_cen"]),
        "images": {
            "i_out": i_out,
            "i_in": i_in,
            "i_diff": i_diff,
            "sigma_diff": sigma_diff
        }
    }
