"""
Automated unit and integration tests for Phase 3: Bayesian Transit Inference.
Tests parameter transforms, Gelman-Rubin convergence diagnostics,
synthetic MCMC recovery, and real TESS photometry fitting on WASP-126b.
"""

import os
import unittest
import numpy as np

from astro_exo.models.transforms import (
    kipping_to_quadratic,
    quadratic_to_kipping,
    compute_transit_duration,
    compute_stellar_density,
    impact_param_to_inclination,
)
from astro_exo.models.diagnostics import compute_gelman_rubin, compute_effective_sample_size
from astro_exo.models.emcee_sampler import EmceeTransitFitter, evaluate_batman_model


class TestPhase3BayesianInference(unittest.TestCase):

    def test_01_transforms(self):
        """Validates Kipping, geometric, and stellar density transformations."""
        # Kipping invertible mapping
        q1_in, q2_in = 0.45, 0.35
        u1, u2 = kipping_to_quadratic(q1_in, q2_in)
        q1_out, q2_out = quadratic_to_kipping(u1, u2)
        self.assertAlmostEqual(q1_in, q1_out, places=5)
        self.assertAlmostEqual(q2_in, q2_out, places=5)

        # Transit duration
        dur_hours = compute_transit_duration(period_days=3.5, rp_rs=0.1, a_rs=12.0, b=0.2)
        self.assertGreater(dur_hours, 1.0)
        self.assertLess(dur_hours, 5.0)

        # Stellar density for Solar-like star (P=365.25 d, a/Rs=215 -> rho ~ 1.4 g/cm^3)
        rho_solar = compute_stellar_density(period_days=365.256, a_rs=215.0)
        self.assertAlmostEqual(rho_solar, 1.408, delta=0.2)

        # Inclination
        inc = impact_param_to_inclination(b=0.0, a_rs=15.0)
        self.assertEqual(inc, 90.0)

    def test_02_diagnostics(self):
        """Verifies Gelman-Rubin R-hat and ESS computations."""
        # Two identical chains should have R-hat = 1.0
        rng = np.random.default_rng(42)
        steps, chains, params = 200, 4, 3
        mock_converged = rng.standard_normal((steps, chains, params))
        r_hat = compute_gelman_rubin(mock_converged)
        self.assertEqual(len(r_hat), params)
        for val in r_hat:
            self.assertLess(val, 1.05)

        # Chains with different offsets should have high R-hat
        mock_unconverged = mock_converged.copy()
        mock_unconverged[:, 0, :] += 5.0
        r_hat_unconv = compute_gelman_rubin(mock_unconverged)
        for val in r_hat_unconv:
            self.assertGreater(val, 1.1)

    def test_03_synthetic_mcmc_recovery(self):
        """Generates synthetic transit data and verifies MAP & MCMC parameter recovery."""
        period = 3.0
        t0_true = 100.0
        rp_true = 0.09
        a_rs_true = 12.0
        b_true = 0.25
        q1_true = 0.35
        q2_true = 0.30
        f0_true = 1.0

        t = np.linspace(99.85, 100.15, 250)
        theta_true = np.array([t0_true, rp_true, a_rs_true, b_true, q1_true, q2_true, f0_true])
        pure_flux = evaluate_batman_model(theta_true, t, period)

        rng = np.random.default_rng(123)
        sigma = 0.0003  # 300 ppm
        noisy_flux = pure_flux + rng.normal(0, sigma, size=len(t))
        flux_err = np.full_like(t, sigma)

        fitter = EmceeTransitFitter(
            time=t,
            flux=noisy_flux,
            flux_err=flux_err,
            period=period,
            t0_expected=t0_true
        )

        # Test MAP
        map_theta = fitter.fit_map()
        self.assertAlmostEqual(map_theta[1], rp_true, delta=0.03)

        # Test short MCMC run
        fitter.run_mcmc(nwalkers=16, nburn=100, nprod=200, seed=42)
        summary = fitter.get_summary()

        # Check recovery of Rp/Rs within 3-sigma
        rp_rec = summary["rp"]["median"]
        self.assertAlmostEqual(rp_rec, rp_true, delta=0.025)

        # Check derived parameters existence
        self.assertIn("duration_hours", summary)
        self.assertIn("depth_ppm", summary)
        self.assertIn("inc_deg", summary)
        self.assertIn("rho_star_g_cm3", summary)

        # Check diagnostics
        diag = fitter.get_diagnostics()
        self.assertIn("r_hat", diag)
        self.assertIn("acceptance_fraction_mean", diag)
        self.assertGreater(diag["acceptance_fraction_mean"], 0.1)

    def test_04_real_tess_wasp126b_mcmc(self):
        """Runs MCMC on real detrended TESS photometry of WASP-126b."""
        lc_path = "data/photometry/TIC_25155310/tess_tic25155310_s0027_lc.fits"
        if not os.path.exists(lc_path):
            self.skipTest("Arquivo FITS de WASP-126b não encontrado no disco local.")

        from astro_exo.ingestion.fits_reader import read_tess_lightcurve
        import wotan

        lc = read_tess_lightcurve(lc_path)
        time = lc["time"]
        flux = lc["flux"]
        err = lc["flux_err"]

        period = 3.28879
        t0 = 2037.8986
        dur_hours = 2.68
        dur_days = dur_hours / 24.0

        # Detrend with wotan
        flat_flux = wotan.flatten(
            time, flux, method="biweight", window_length=0.75,
            edge_cutoff=0.5, break_tolerance=0.5
        )

        # Slice 2 transits
        phase = (time - t0 + 0.5 * period) % period - 0.5 * period
        mask = np.abs(phase) < (1.5 * dur_days)

        t_sub = time[mask]
        f_sub = flat_flux[mask]
        e_sub = err[mask]

        fitter = EmceeTransitFitter(
            time=t_sub,
            flux=f_sub,
            flux_err=e_sub,
            period=period,
            t0_expected=t0
        )

        fitter.fit_map()
        fitter.run_mcmc(nwalkers=16, nburn=100, nprod=250, seed=42)
        summary = fitter.get_summary()

        # WASP-126b has published Rp/Rs ~ 0.078 (depth ~ 6100 ppm)
        rp_med = summary["rp"]["median"]
        self.assertGreater(rp_med, 0.060)
        self.assertLess(rp_med, 0.095)

        depth_ppm = summary["depth_ppm"]["median"]
        self.assertGreater(depth_ppm, 4000.0)
        self.assertLess(depth_ppm, 9000.0)


if __name__ == "__main__":
    unittest.main()
