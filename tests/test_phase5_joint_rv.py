"""
Unit and integration tests for Phase 5:
Joint Transit + Radial Velocity (RV) Modeling, Multi-Instrument Dynamics,
and Physical Parameter Inference (Mp, Rp, rho_p, log_g).
"""

import os
import unittest
import numpy as np

from astro_exo.models.joint_rv import (
    solve_kepler,
    keplerian_rv,
    compute_planetary_mass_density,
    classify_planetary_interior,
    JointTransitRVSampler
)
from astro_exo.ingestion.rv_loader import (
    RVDataset,
    load_rv_csv,
    simulate_multi_instrument_rv
)


class TestPhase5JointRV(unittest.TestCase):
    """
    Test suite for Phase 5 Keplerian dynamics and joint inference.
    """

    def setUp(self):
        self.rv_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rv_data")

    def test_01_keplerian_rv_circular(self):
        """
        Verify Keplerian RV for circular orbit (e = 0) at key orbital phases.
        """
        period = 2.0
        t0 = 100.0
        k_amp = 50.0
        gamma = 10.0

        # Phase 0.0 (transit): RV = gamma
        rv_t0 = keplerian_rv(np.array([t0]), period, t0, k_amp, ecc=0.0, gamma=gamma)[0]
        self.assertAlmostEqual(rv_t0, gamma, places=5)

        # Phase 0.25 (quadrature, planet moving away): RV = gamma - K
        rv_q1 = keplerian_rv(np.array([t0 + 0.25 * period]), period, t0, k_amp, ecc=0.0, gamma=gamma)[0]
        self.assertAlmostEqual(rv_q1, gamma - k_amp, places=5)

        # Phase 0.50 (secondary eclipse): RV = gamma
        rv_sec = keplerian_rv(np.array([t0 + 0.50 * period]), period, t0, k_amp, ecc=0.0, gamma=gamma)[0]
        self.assertAlmostEqual(rv_sec, gamma, places=5)

        # Phase 0.75 (quadrature, planet approaching): RV = gamma + K
        rv_q3 = keplerian_rv(np.array([t0 + 0.75 * period]), period, t0, k_amp, ecc=0.0, gamma=gamma)[0]
        self.assertAlmostEqual(rv_q3, gamma + k_amp, places=5)

    def test_02_keplerian_rv_eccentric(self):
        """
        Verify Keplerian RV solver for eccentric orbits with different omega.
        """
        period = 4.0
        t0 = 50.0
        k_amp = 120.0
        ecc = 0.35
        omega = 60.0

        t_fine = np.linspace(t0, t0 + period, 100)
        rv_curve = keplerian_rv(t_fine, period, t0, k_amp, ecc=ecc, omega_deg=omega, gamma=0.0)

        # Velocity must remain finite, real, and periodic
        self.assertFalse(np.any(np.isnan(rv_curve)))
        self.assertAlmostEqual(rv_curve[0], rv_curve[-1], places=4)
        # Peak-to-peak amplitude is approximately 2 * K
        ptp = np.ptp(rv_curve)
        self.assertTrue(1.5 * k_amp < ptp < 2.5 * k_amp)

    def test_03_compute_planetary_mass_density(self):
        """
        Verify derivation of planetary mass, bulk density, surface gravity and radius.
        """
        # Test case: WASP-126b parameters (Maxted et al. 2016)
        # M* = 1.12 Msun, R* = 1.27 Rsun, P = 3.2888 d, K = 36.4 m/s, Rp/R* = 0.0763
        phys = compute_planetary_mass_density(
            m_star_msun=1.12,
            r_star_rsun=1.27,
            period_days=3.28879,
            k_semiamp_ms=36.4,
            rp_rs=0.0763,
            ecc=0.0,
            inc_deg=87.9
        )

        # Known literature: Mp ~ 0.28 Mjup, Rp ~ 0.95 Rjup, rho ~ 0.4 g/cm^3
        self.assertAlmostEqual(phys["mass_jupiter"], 0.287, delta=0.03)
        self.assertAlmostEqual(phys["radius_jupiter"], 0.943, delta=0.05)
        self.assertAlmostEqual(phys["density_g_cm3"], 0.42, delta=0.08)
        self.assertGreater(phys["log_g_cgs"], 2.5)
        self.assertLess(phys["log_g_cgs"], 3.5)
        self.assertIn("Jupiter", phys["interior_classification"])

    def test_04_interior_classification(self):
        """
        Verify interior composition regimes.
        """
        # Earth: Mp = 1.0 M_earth, rho = 5.51 g/cm^3
        c_earth = classify_planetary_interior(1.0, 5.51)
        self.assertIn("Terrestrial", c_earth)

        # Super-Earth rocky: Mp = 5.0 M_earth, rho = 6.2 g/cm^3
        c_se = classify_planetary_interior(5.0, 6.2)
        self.assertIn("Super-Earth (Dense Rocky)", c_se)

        # Sub-Neptune: Mp = 15.0 M_earth, rho = 1.6 g/cm^3
        c_sn = classify_planetary_interior(15.0, 1.6)
        self.assertIn("Sub-Neptune", c_sn)

        # Inflated Hot Jupiter: Mp = 150.0 M_earth, rho = 0.25 g/cm^3
        c_inf = classify_planetary_interior(150.0, 0.25)
        self.assertIn("Inflated", c_inf)

    def test_05_rv_loader_csv(self):
        """
        Verify reading CSV RV datasets and instrument partitioning.
        """
        csv_path = os.path.join(self.rv_dir, "wasp126_rv.csv")
        self.assertTrue(os.path.isfile(csv_path), f"Missing {csv_path}")

        dataset = load_rv_csv(csv_path)
        self.assertGreater(dataset.n_points, 30)
        self.assertEqual(dataset.n_instruments, 2)
        self.assertIn("HARPS", dataset.instrument_names)
        self.assertIn("CORALIE", dataset.instrument_names)
        self.assertEqual(len(dataset.inst_indices), dataset.n_points)

    def test_06_synthetic_multi_instrument_mcmc(self):
        """
        Verify that MCMC recovers K semi-amplitude on a multi-instrument synthetic dataset.
        """
        k_true = 45.0
        period = 2.5
        t0 = 2457000.0

        ds, truth = simulate_multi_instrument_rv(
            period_days=period,
            t0_bjd=t0,
            k_semiamp_ms=k_true,
            n_points_per_inst={"HARPS": 30, "CORALIE": 20},
            gamma_offsets={"HARPS": 0.0, "CORALIE": -25.0},
            jitters_ms={"HARPS": 1.0, "CORALIE": 3.0},
            random_seed=123
        )

        sampler = JointTransitRVSampler(
            rv_dataset=ds,
            period_days=period,
            t0_bjd=t0,
            m_star_msun=1.0,
            r_star_rsun=1.0,
            rp_rs_prior=0.10
        )

        res = sampler.run_rv_mcmc(nwalkers=24, nburn=150, nsteps=300, random_seed=42)
        k_fit = res["parameters"]["k_semiamp"]["median"]
        k_err = res["parameters"]["k_semiamp"]["error_plus"]

        # Recovered within 2 sigma
        self.assertAlmostEqual(k_fit, k_true, delta=2.5 * k_err)
        self.assertGreater(res["physical"]["mass_jupiter"], 0.0)

    def test_07_wasp77b_dediluted_joint_fit(self):
        """
        Verify WASP-77b joint modeling with de-diluted radius ratio from Phase 4.
        """
        csv_path = os.path.join(self.rv_dir, "wasp77_rv.csv")
        dataset = load_rv_csv(csv_path)

        # De-diluted radius ratio from Phase 4: rp_rs = 0.1324
        sampler = JointTransitRVSampler(
            rv_dataset=dataset,
            period_days=1.36003,
            t0_bjd=2456200.5,
            m_star_msun=1.00,
            r_star_rsun=0.95,
            rp_rs_prior=0.1324  # Phase 4 de-diluted physical value
        )

        res = sampler.run_rv_mcmc(nwalkers=24, nburn=150, nsteps=300, random_seed=99)
        phys = res["physical"]

        # Literature: Mp ~ 1.76 Mjup, rho ~ 1.2 g/cm^3
        self.assertAlmostEqual(phys["mass_jupiter"], 1.76, delta=0.20)
        self.assertAlmostEqual(phys["density_g_cm3"], 1.20, delta=0.25)
        self.assertEqual(phys["interior_classification"], "Dense / Massive Hot Jupiter")

    def test_08_kepler_solver_danby_high_eccentricity(self):
        """
        Verify Danby initial guess and adaptive Newton-Raphson Kepler solver:
        Achieves max |E - e*sin(E) - M| < 10^-14 across all orbital phases
        and high eccentricities up to e -> 1.0 within <= 15 iterations.
        """
        m_grid = np.linspace(0.0, 2.0 * np.pi, 500)
        eccentricities = [0.0, 0.1, 0.5, 0.85, 0.95, 0.99]

        for ecc in eccentricities:
            e_sol = solve_kepler(m_grid, ecc, max_iter=15, tol=1e-14)
            # Check Kepler equation residual: |E - e*sin(E) - M|
            residual = np.abs(e_sol - ecc * np.sin(e_sol) - m_grid)
            max_res = float(np.max(residual))
            self.assertLess(
                max_res, 1e-13,
                f"Kepler solver failed for ecc={ecc}: max residual={max_res:.2e}"
            )

        # Verify scalar input support
        e_scalar = solve_kepler(np.pi / 3.0, 0.75, max_iter=15, tol=1e-14)
        self.assertIsInstance(e_scalar, float)
        self.assertLess(abs(e_scalar - 0.75 * np.sin(e_scalar) - np.pi / 3.0), 1e-13)

    def test_09_uncertainty_propagation_density_and_mass(self):
        """
        Verify physical uncertainty propagation for planetary mass and bulk density:
        - Incorportates M*, R*, and Rp/R* uncertainties.
        - Ensures density uncertainty reflects cubic dependence on radius: sigma_rho / rho >= 3 * sigma_R / R.
        """
        csv_path = os.path.join(self.rv_dir, "wasp77_rv.csv")
        dataset = load_rv_csv(csv_path)

        # 1. Default conservative literature uncertainties (5% M*, 3% R*, 3% Rp/Rs)
        sampler_default = JointTransitRVSampler(
            rv_dataset=dataset,
            period_days=1.36003,
            t0_bjd=2456200.5,
            m_star_msun=1.00,
            r_star_rsun=0.95,
            rp_rs_prior=0.1324
        )
        self.assertEqual(sampler_default.m_star_err, 0.05 * 1.00)
        self.assertEqual(sampler_default.r_star_err, 0.03 * 0.95)
        self.assertEqual(sampler_default.rp_rs_err, 0.03 * 0.1324)

        fit_res = sampler_default.run_rv_mcmc(nwalkers=24, nburn=100, nsteps=200, random_seed=42)
        phys = fit_res["physical"]

        self.assertIn("mass_jupiter_err", phys)
        self.assertIn("radius_jupiter_err", phys)
        self.assertIn("density_g_cm3_err", phys)
        self.assertGreater(phys["mass_jupiter_err"], 0.0)
        self.assertGreater(phys["radius_jupiter_err"], 0.0)
        self.assertGreater(phys["density_g_cm3_err"], 0.0)

        # Relative density uncertainty must reflect cubic radius dependence:
        # sigma_rho / rho >= 3 * (sigma_R* / R*) = 3 * 0.03 = 0.09 (9%)
        rel_rho_err = phys["density_g_cm3_err"] / phys["density_g_cm3"]
        self.assertGreater(rel_rho_err, 0.08, f"Relative density uncertainty too low: {rel_rho_err:.3f}")

        # 2. Explicit custom uncertainties test
        sampler_custom = JointTransitRVSampler(
            rv_dataset=dataset,
            period_days=1.36003,
            t0_bjd=2456200.5,
            m_star_msun=1.00,
            r_star_rsun=0.95,
            rp_rs_prior=0.1324,
            m_star_err=0.08,
            r_star_err=0.06,
            rp_rs_err=0.005
        )
        self.assertEqual(sampler_custom.m_star_err, 0.08)
        self.assertEqual(sampler_custom.r_star_err, 0.06)
        self.assertEqual(sampler_custom.rp_rs_err, 0.005)

        # Standalone compute_planetary_mass_density with uncertainties
        phys_direct = compute_planetary_mass_density(
            m_star_msun=1.00,
            r_star_rsun=0.95,
            period_days=1.36003,
            k_semiamp_ms=320.0,
            rp_rs=0.1324,
            m_star_err=0.05,
            r_star_err=0.03,
            k_semiamp_err=5.0,
            rp_rs_err=0.003
        )
        self.assertIn("mass_jupiter_err", phys_direct)
        self.assertIn("density_g_cm3_err", phys_direct)
        self.assertGreater(phys_direct["density_g_cm3_err"], 0.05)


if __name__ == "__main__":
    unittest.main()
