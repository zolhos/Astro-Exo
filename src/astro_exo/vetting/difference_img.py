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
    offset_pix = np.hypot(x_diff_cen - pix_x_target, y_diff_cen - pix_y_target)
    offset_arcsec = float(offset_pix * tess_pixel_scale_arcsec)

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
        "offset_arcsec": offset_arcsec,
        "sigma_offset_arcsec": sigma_offset_arcsec,
        "offset_significance_sigma": significance
    }
