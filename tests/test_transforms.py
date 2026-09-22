"""
Tests for parameter transformations and physical constraints.
"""

import numpy as np
from astro_exo.models.transforms import (
    kipping_to_quadratic,
    quadratic_to_kipping,
    ecc_omega_to_xy,
    xy_to_ecc_omega,
    impact_param_to_inclination,
    compute_stellar_density
)


def test_kipping_roundtrip():
    """Verify invertibility of Kipping limb darkening parameterization."""
    q1_vals = [0.1, 0.35, 0.7, 0.9]
    q2_vals = [0.2, 0.5, 0.8]

    for q1 in q1_vals:
        for q2 in q2_vals:
            u1, u2 = kipping_to_quadratic(q1, q2)
            # Physical stability criteria
            assert u1 + u2 < 1.0 + 1e-7
            assert u1 > 0.0 - 1e-7
            assert u1 + 2.0 * u2 > 0.0 - 1e-7

            # Roundtrip test
            q1_rec, q2_rec = quadratic_to_kipping(u1, u2)
            assert np.isclose(q1, q1_rec, atol=1e-5)
            assert np.isclose(q2, q2_rec, atol=1e-5)


def test_eccentricity_transforms():
    """Verify mapping between (e, omega) and (h, k)."""
    ecc = 0.25
    omega_rad = np.pi / 4.0

    h, k = ecc_omega_to_xy(ecc, omega_rad)
    ecc_rec, omega_rec = xy_to_ecc_omega(h, k)

    assert np.isclose(ecc, ecc_rec, atol=1e-6)
    assert np.isclose(omega_rad, omega_rec, atol=1e-6)


def test_impact_parameter_to_inclination():
    """Verify impact parameter conversion."""
    # Central transit (b=0) -> i = 90 degrees
    inc_central = impact_param_to_inclination(0.0, 10.0)
    assert np.isclose(inc_central, 90.0)

    # Grazing transit limit (b = a/Rs) -> i = 0 degrees (or cos_i = 1)
    inc_edge = impact_param_to_inclination(10.0, 10.0)
    assert np.isclose(inc_edge, 0.0)


def test_stellar_density_scaling():
    """Verify stellar density calculation matches solar density when appropriate."""
    # For a planet at P = 365.25 days and a/Rs = 215 (Earth-Sun system)
    rho_cgs = compute_stellar_density(period_days=365.25, a_rs=215.0)
    # Solar density is ~ 1.41 g/cm^3
    assert 1.2 < rho_cgs < 1.6
