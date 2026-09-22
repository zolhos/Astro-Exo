"""
Dilution and maximum false-positive transit depth calculation for neighboring stars.
"""

from typing import List, Dict, Any
import numpy as np


def calculate_max_transit_depth(
    target_mag: float,
    neighbor_mag: float,
    neighbor_fraction_in_aperture: float = 1.0,
    max_eclipse_depth: float = 1.0
) -> float:
    """
    Computes the maximum possible transit depth (in ppm or fractional flux)
    that a neighboring star could inject into the target aperture if it were a
    total eclipsing binary (delta_true = 100% or max_eclipse_depth).

    Flux ratio: F_neighbor / F_target = 10^(-0.4 * (m_neighbor - m_target))
    Observed dilution: delta_obs_max = (F_neighbor_in_aperture) / (F_target + F_all_neighbors) * delta_true

    Parameters
    ----------
    target_mag : float
        Magnitude of the target star.
    neighbor_mag : float
        Magnitude of the neighboring star.
    neighbor_fraction_in_aperture : float
        Fraction of neighbor's PRF flux contained within the photometric aperture.
    max_eclipse_depth : float
        Maximum theoretical eclipse depth of the neighbor (default: 1.0 = 100%).

    Returns
    -------
    max_depth_fraction : float
        Maximum observable transit depth in fractional flux.
    """
    delta_mag = neighbor_mag - target_mag
    flux_ratio = 10.0 ** (-0.4 * delta_mag)

    # Maximum fractional depth injected into target flux
    max_depth_fraction = (flux_ratio * neighbor_fraction_in_aperture * max_eclipse_depth) / (1.0 + flux_ratio * neighbor_fraction_in_aperture)
    return float(max_depth_fraction)


def rule_out_neighbors_as_blends(
    observed_transit_depth_ppm: float,
    target_mag: float,
    neighbors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Screens all neighboring stars: if neighbor's max possible injected depth
    is LESS than the observed transit depth, it is physically impossible for that
    neighbor to be the source of the transit (even as a 100% eclipse binary).
    """
    observed_depth_frac = observed_transit_depth_ppm * 1e-6
    vetted_neighbors = []

    for star in neighbors:
        m_star = star.get("phot_g_mean_mag", 99.0)
        max_depth = calculate_max_transit_depth(target_mag, m_star)
        can_produce_transit = max_depth >= observed_depth_frac

        star_result = dict(star)
        star_result["max_injected_depth_ppm"] = float(max_depth * 1e6)
        star_result["can_cause_transit"] = can_produce_transit
        star_result["ruled_out"] = not can_produce_transit
        vetted_neighbors.append(star_result)

    return vetted_neighbors
