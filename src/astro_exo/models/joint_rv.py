"""
Joint Bayesian inference of Photometric Transits and Radial Velocity (RV) Doppler measurements.
Solves for planetary mass, radius, density, and orbital parameters simultaneously across multiple spectrographs.
"""

from typing import List, Dict, Optional
import numpy as np


def keplerian_rv(
    time: np.ndarray,
    period: float,
    t0: float,
    k_semiamp: float,
    ecc: float = 0.0,
    omega_deg: float = 90.0,
    gamma: float = 0.0
) -> np.ndarray:
    """
    Computes Keplerian Radial Velocity curve.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps (days).
    period : float
        Orbital period (days).
    t0 : float
        Time of mid-transit (inferring inferior conjunction).
    k_semiamp : float
        Radial velocity semi-amplitude (m/s).
    ecc : float
        Orbital eccentricity.
    omega_deg : float
        Argument of periastron (degrees).
    gamma : float
        Systemic velocity / instrument zero point (m/s).

    Returns
    -------
    rv : np.ndarray
        Radial velocity model at timestamps (m/s).
    """
    # Mean anomaly M = 2 * pi * (t - T_peri) / P
    # Transit occurs at true anomaly f = pi/2 - omega (for edge-on orbit)
    omega_rad = np.radians(omega_deg)

    # For circular orbit approximation:
    if ecc == 0.0:
        mean_anomaly = 2.0 * np.pi * (time - t0) / period
        # RV curve: gamma + K * sin(M)
        return gamma + k_semiamp * np.sin(mean_anomaly)

    # Solving Kepler's equation for eccentric orbits
    mean_anomaly = (2.0 * np.pi * (time - t0) / period) % (2.0 * np.pi)

    # Solve E - e * sin(E) = M
    e_anom = mean_anomaly.copy()
    for _ in range(5):
        f_eval = e_anom - ecc * np.sin(e_anom) - mean_anomaly
        f_prime = 1.0 - ecc * np.cos(e_anom)
        e_anom -= f_eval / f_prime

    # True anomaly f
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(e_anom / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(e_anom / 2.0)
    )

    return gamma + k_semiamp * (np.cos(true_anomaly + omega_rad) + ecc * np.cos(omega_rad))


def compute_planetary_mass_density(
    m_star_msun: float,
    r_star_rsun: float,
    period_days: float,
    k_semiamp_ms: float,
    rp_rs: float,
    ecc: float = 0.0,
    inc_deg: float = 90.0
) -> Dict[str, float]:
    """
    Derive physical planetary mass, radius, and mean density from joint observables.
    """
    # Physical constants (CGS / SI)
    M_EARTH_KG = 5.972e24
    M_JUP_KG = 1.898e27
    R_EARTH_KM = 6371.0
    R_JUP_KM = 69911.0
    M_SUN_KG = 1.989e30
    R_SUN_KM = 696340.0
    G_SI = 6.6743e-11

    # Semi-amplitude formula:
    # K = (2*pi*G / P)^(1/3) * (M_p * sin(i) / (M_* + M_p)^(2/3)) * (1 / sqrt(1 - e^2))
    p_sec = period_days * 86400.0
    m_star_kg = m_star_msun * M_SUN_KG
    r_star_km = r_star_rsun * R_SUN_KM

    sin_i = np.sin(np.radians(inc_deg))
    factor = (p_sec / (2.0 * np.pi * G_SI)) ** (1.0 / 3.0)
    sqrt_1_e2 = np.sqrt(1.0 - ecc**2)

    # Approximate M_p << M_*
    m_p_kg = (k_semiamp_ms * sqrt_1_e2 / sin_i) * factor * (m_star_kg ** (2.0 / 3.0))

    # Planetary radius
    r_p_km = rp_rs * r_star_km

    # Planetary mass in Earth and Jupiter units
    m_p_earth = m_p_kg / M_EARTH_KG
    m_p_jup = m_p_kg / M_JUP_KG

    # Planetary radius in Earth and Jupiter units
    r_p_earth = r_p_km / R_EARTH_KM
    r_p_jup = r_p_km / R_JUP_KM

    # Mean density (g / cm^3)
    vol_cm3 = (4.0 / 3.0) * np.pi * ((r_p_km * 1e5) ** 3)
    rho_g_cm3 = (m_p_kg * 1e3) / vol_cm3

    return {
        "mass_earth": float(m_p_earth),
        "mass_jupiter": float(m_p_jup),
        "radius_earth": float(r_p_earth),
        "radius_jupiter": float(r_p_jup),
        "density_g_cm3": float(rho_g_cm3)
    }
