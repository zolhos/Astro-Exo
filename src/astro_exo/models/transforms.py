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


def compute_transit_duration(period_days: float, rp_rs: float, a_rs: float, b: float) -> float:
    """
    Compute total transit duration T_14 (first to fourth contact) in hours for a circular orbit.

    T_14 = (P / pi) * arcsin( (1 / (a/R_*)) * sqrt((1 + Rp/R_*)^2 - b^2) / sin(i) )
    """
    if a_rs <= 0 or (1.0 + rp_rs)**2 < b**2:
        return 0.0

    cos_i = np.clip(b / a_rs, -1.0, 1.0)
    sin_i = np.sqrt(np.maximum(0.0, 1.0 - cos_i**2))
    if sin_i == 0.0:
        return 0.0

    arg = (1.0 / a_rs) * np.sqrt(np.maximum(0.0, (1.0 + rp_rs)**2 - b**2)) / sin_i
    if arg >= 1.0:
        return float(period_days * 24.0)

    t14_days = (period_days / np.pi) * np.arcsin(arg)
    return float(t14_days * 24.0)


def align_t0_to_dataset(t0: float, period: float, time_arr: np.ndarray) -> float:
    """
    Shifts reference transit epoch t0 by an integer number of orbital periods P
    so that it lies closest to the median time of the observation dataset.
    """
    t_mid = float(np.nanmedian(time_arr))
    n_epochs = round((t_mid - t0) / period)
    return float(t0 + n_epochs * period)
