"""
Parameter transformations and physical reparameterizations for exoplanet modeling.
Includes Kipping (2013) triangular limb darkening, eccentric orbit mappings,
and stellar density relations.
"""

import numpy as np


def kipping_to_quadratic(q1: float, q2: float) -> tuple[float, float]:
    """
    Transform Kipping (2013) unconstrained uniform parameters (q1, q2)
    to quadratic limb darkening coefficients (u1, u2).

    Parameters
    ----------
    q1, q2 : float or np.ndarray
        Uniform samples in [0, 1].

    Returns
    -------
    u1, u2 : float or np.ndarray
        Quadratic limb darkening coefficients satisfying physical stability.
    """
    sqrt_q1 = np.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def quadratic_to_kipping(u1: float, u2: float) -> tuple[float, float]:
    """
    Inverse transformation from quadratic limb darkening (u1, u2)
    to Kipping (2013) triangular parameters (q1, q2).

    Parameters
    ----------
    u1, u2 : float or np.ndarray
        Quadratic limb darkening coefficients.

    Returns
    -------
    q1, q2 : float or np.ndarray
        Kipping parameters in [0, 1].
    """
    q1 = (u1 + u2) ** 2
    q2 = 0.5 * u1 / (u1 + u2 + 1e-12)
    return q1, q2


def ecc_omega_to_xy(ecc: float, omega_rad: float) -> tuple[float, float]:
    """
    Transform eccentricity and argument of periastron (e, omega)
    to Cartesian sampling coordinates (sqrt(e)*cos(omega), sqrt(e)*sin(omega)).

    Parameters
    ----------
    ecc : float or np.ndarray
        Orbital eccentricity in [0, 1).
    omega_rad : float or np.ndarray
        Argument of periastron in radians.

    Returns
    -------
    h, k : float or np.ndarray
        Cartesian components: h = sqrt(e)*cos(omega), k = sqrt(e)*sin(omega).
    """
    sqrt_e = np.sqrt(ecc)
    h = sqrt_e * np.cos(omega_rad)
    k = sqrt_e * np.sin(omega_rad)
    return h, k


def xy_to_ecc_omega(h: float, k: float) -> tuple[float, float]:
    """
    Inverse transformation from Cartesian coordinates (h, k)
    to orbital eccentricity and argument of periastron (e, omega).

    Parameters
    ----------
    h, k : float or np.ndarray
        Cartesian components: h = sqrt(e)*cos(omega), k = sqrt(e)*sin(omega).

    Returns
    -------
    ecc, omega_rad : float or np.ndarray
        Orbital eccentricity in [0, 1) and argument of periastron in radians in [-pi, pi].
    """
    ecc = h**2 + k**2
    omega_rad = np.arctan2(k, h)
    return ecc, omega_rad


def impact_param_to_inclination(b: float, a_rs: float) -> float:
    """
    Convert impact parameter b and normalized semi-major axis a/R_*
    to orbital inclination in degrees for circular orbits.
    """
    cos_i = b / a_rs
    if np.any(cos_i > 1.0) or np.any(cos_i < 0.0):
        return np.nan
    return np.degrees(np.arccos(cos_i))


def compute_stellar_density(period_days: float, a_rs: float) -> float:
    """
    Compute mean stellar density in g/cm^3 from orbital period and scaled semi-major axis.

    rho_* / rho_sun = (4 * pi^2 / (G * P^2)) * (a/R_*)^3
    where rho_sun = 1.408 g / cm^3.
    """
    # G in (R_sun^3) / (M_sun * day^2) = 4 * pi^2 * (a_earth_rsun^3) / (365.256^2) = 2940.457
    G_ASTRO = 2940.457
    # Normalized density relative to Sun (rho_* / rho_sun)
    rho_ratio_to_sun = (4.0 * (np.pi ** 2) / (G_ASTRO * (period_days ** 2))) * (a_rs ** 3)
    # 1 Solar mean density = 1.408 g / cm^3
    return float(rho_ratio_to_sun * 1.408)
