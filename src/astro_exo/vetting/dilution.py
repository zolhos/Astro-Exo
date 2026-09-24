"""
Dilution and maximum false-positive transit depth calculation for neighboring stars,
multispectral aperture contamination, and analytical de-dilution (transit restoration).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def calculate_critical_delta_mag(observed_transit_depth_ppm: float) -> float:
    """
    Computes the critical magnitude difference (Delta m_crit):
      Delta m_crit = -2.5 * log10(delta_obs)

    Any neighboring star fainter than (m_target + Delta m_crit) CANNOT produce
    the observed transit depth, even if it is a 100% eclipsing binary (total eclipse).
    """
    if observed_transit_depth_ppm <= 0:
        return 0.0
    delta_obs = observed_transit_depth_ppm * 1e-6
    return float(round(-2.5 * np.log10(delta_obs), 3))


def calculate_max_transit_depth(
    target_mag: float,
    neighbor_mag: float,
    neighbor_fraction_in_aperture: float = 1.0,
    max_eclipse_depth: float = 1.0
) -> float:
    """
    Computes the maximum possible transit depth (fractional flux)
    that a neighboring star could inject into the target aperture if it were a
    total eclipsing binary (delta_true = 100% or max_eclipse_depth).

    Flux ratio: F_neighbor / F_target = 10^(-0.4 * (m_neighbor - m_target))
    Observed dilution: delta_obs_max = (F_neighbor_in_aperture) / (F_target + F_all_neighbors) * delta_true
    """
    delta_mag = neighbor_mag - target_mag
    flux_ratio = 10.0 ** (-0.4 * delta_mag)

    # Maximum fractional depth injected into target flux
    max_depth_fraction = (flux_ratio * neighbor_fraction_in_aperture * max_eclipse_depth) / (1.0 + flux_ratio * neighbor_fraction_in_aperture)
    return float(max_depth_fraction)


def estimate_aperture_flux_fraction(
    dist_arcsec: float,
    tess_pixel_scale_arcsec: float = 21.0,
    prf_sigma_arcsec: float = 17.8,
    aperture_radius_arcsec: float = 42.0
) -> float:
    """
    Estimates the fraction of a neighboring star's flux falling into the target aperture,
    assuming a 2D Gaussian PRF profile integrated within the photometric aperture.
    """
    if dist_arcsec <= 5.0:
        # Star is well inside the core aperture (e.g. WASP-77B at 3.3")
        return 0.98

    if dist_arcsec <= aperture_radius_arcsec:
        # Inside aperture bounds, fraction decreases smoothly towards the edge
        return float(np.clip(1.0 - 0.5 * (dist_arcsec / aperture_radius_arcsec) ** 2, 0.20, 0.95))

    # Outside aperture boundary: PRF wing overlap
    separation_sigmas = (dist_arcsec - aperture_radius_arcsec) / prf_sigma_arcsec
    fraction = float(0.20 * np.exp(-0.5 * (separation_sigmas ** 2)))
    return float(np.clip(fraction, 0.0, 0.20))


def rule_out_neighbors_as_blends(
    observed_transit_depth_ppm: float,
    target_mag: float,
    neighbors: List[Dict[str, Any]],
    mag_key: str = "tess_mag"
) -> List[Dict[str, Any]]:
    """
    Screens all neighboring stars: if neighbor's max possible injected depth
    is LESS than the observed transit depth, it is physically impossible for that
    neighbor to be the source of the transit (even as a 100% eclipse binary).
    """
    observed_depth_frac = observed_transit_depth_ppm * 1e-6
    delta_m_crit = calculate_critical_delta_mag(observed_transit_depth_ppm)
    vetted_neighbors = []

    for star in neighbors:
        m_star = star.get(mag_key, star.get("phot_g_mean_mag", 99.0))
        dist_arcsec = star.get("dist_arcsec", 0.0)

        frac_in_ap = estimate_aperture_flux_fraction(dist_arcsec)
        max_depth = calculate_max_transit_depth(
            target_mag=target_mag,
            neighbor_mag=m_star,
            neighbor_fraction_in_aperture=frac_in_ap
        )
        can_produce_transit = max_depth >= observed_depth_frac

        star_result = dict(star)
        star_result["aperture_flux_fraction"] = frac_in_ap
        star_result["max_injected_depth_ppm"] = float(round(max_depth * 1e6, 1))
        star_result["delta_mag"] = float(round(m_star - target_mag, 3))
        star_result["delta_m_crit"] = delta_m_crit
        star_result["can_cause_transit"] = can_produce_transit
        star_result["ruled_out"] = not can_produce_transit
        vetted_neighbors.append(star_result)

    return vetted_neighbors


def calculate_dilution_factor(
    target_mag: float,
    neighbors: List[Dict[str, Any]],
    mag_key: str = "tess_mag"
) -> Dict[str, float]:
    """
    Computes the total flux dilution factor D within the photometric aperture:
      F_target = 10^(-0.4 * target_mag)
      F_blend = sum(10^(-0.4 * m_j) * f_j_in_ap)
      D = F_target / (F_target + F_blend)

    Returns
    -------
    Dict containing:
      - dilution_factor (D <= 1.0)
      - contamination_ratio (F_blend / F_target)
      - blend_flux_fraction (F_blend / F_total = 1 - D)
    """
    f_target = 10.0 ** (-0.4 * target_mag)
    f_blend = 0.0

    for star in neighbors:
        # Exclude the target star itself if it has dist_arcsec == 0
        if star.get("dist_arcsec", 99.0) < 0.1:
            continue

        m_star = star.get(mag_key, star.get("phot_g_mean_mag", 99.0))
        dist_arcsec = star.get("dist_arcsec", 0.0)
        frac_in_ap = star.get("aperture_flux_fraction", estimate_aperture_flux_fraction(dist_arcsec))

        flux_star = 10.0 ** (-0.4 * m_star)
        f_blend += flux_star * frac_in_ap

    f_total = f_target + f_blend
    dilution_factor = f_target / f_total if f_total > 0 else 1.0
    contamination_ratio = f_blend / f_target if f_target > 0 else 0.0

    return {
        "dilution_factor": float(round(dilution_factor, 5)),
        "contamination_ratio": float(round(contamination_ratio, 5)),
        "blend_flux_fraction": float(round(1.0 - dilution_factor, 5))
    }


def correct_diluted_transit_depth(
    observed_depth_ppm: float,
    dilution_factor: float
) -> float:
    """
    Restores the true physical transit depth (delta_true):
      delta_true = delta_obs / D
    """
    if dilution_factor <= 0:
        return observed_depth_ppm
    return float(round(observed_depth_ppm / dilution_factor, 2))


def restore_true_radius_ratio(
    rp_rs_obs: float,
    dilution_factor: float,
    rp_rs_err: Optional[float] = None
) -> Tuple[float, Optional[float]]:
    """
    Restores the true physical planet-to-star radius ratio:
      (Rp / Rs)_true = (Rp / Rs)_obs / sqrt(D)
    Propagates uncertainty assuming D is well-constrained.
    """
    if dilution_factor <= 0:
        return rp_rs_obs, rp_rs_err

    sqrt_d = np.sqrt(dilution_factor)
    rp_rs_true = float(round(rp_rs_obs / sqrt_d, 5))

    if rp_rs_err is not None:
        rp_rs_err_true = float(round(rp_rs_err / sqrt_d, 5))
        return rp_rs_true, rp_rs_err_true

    return rp_rs_true, None


def correct_binary_blend_wasp77(
    rp_rs_diluted: float = 0.1186,
    rp_rs_err: float = 0.0019,
    delta_tmag: float = 1.52
) -> Dict[str, Any]:
    """
    Analytic de-dilution for the WASP-77 (WASP-77A + WASP-77B) resolved binary system.
    WASP-77A (target host) and WASP-77B are separated by 3.3 arcsec (Delta Tmag ~ 1.52),
    falling entirely within the TESS single-pixel aperture core (21 arcsec/pixel).
    """
    flux_ratio = 10.0 ** (-0.4 * delta_tmag)  # F_B / F_A ~ 0.2466
    dilution_factor = 1.0 / (1.0 + flux_ratio)  # D ~ 0.8022

    rp_true, rp_err_true = restore_true_radius_ratio(
        rp_rs_obs=rp_rs_diluted,
        dilution_factor=dilution_factor,
        rp_rs_err=rp_rs_err
    )

    depth_diluted_ppm = (rp_rs_diluted ** 2) * 1e6
    depth_true_ppm = (rp_true ** 2) * 1e6

    return {
        "system": "WASP-77b / WASP-77A",
        "companion": "WASP-77B (sep = 3.3 arcsec)",
        "delta_tmag": delta_tmag,
        "flux_ratio_B_over_A": float(round(flux_ratio, 4)),
        "dilution_factor": float(round(dilution_factor, 4)),
        "observed_rp_rs": rp_rs_diluted,
        "observed_rp_rs_err": rp_rs_err,
        "observed_depth_ppm": float(round(depth_diluted_ppm, 1)),
        "true_rp_rs": rp_true,
        "true_rp_rs_err": rp_err_true,
        "true_depth_ppm": float(round(depth_true_ppm, 1)),
        "depth_correction_percent": float(round((depth_true_ppm / depth_diluted_ppm - 1.0) * 100.0, 2))
    }
