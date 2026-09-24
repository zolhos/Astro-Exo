"""
Comprehensive all-module verification test suite on novel small samples.
Tests all Astro-Exo modules:
  1. Ingestion & Stellar Detrending (NASA Archive, Detrending, Search/TLS, Multi-instrument RV)
  2. Sub-pixel Spatial Vetting (Difference Imaging, Centroid Shift, PRF 2D, Pixel LCs, Gaia Blends, TRICERATOPS)
  3. Bayesian Modeling & Inference (Kipping transforms, GP SHO noise, emcee, JAX/NumPyro NUTS, R-hat)
  4. Keplerian Dynamics & Joint Characterization (Doppler solver, Mp, Rp, rho_p, EOS classification)
  5. Pipeline & Batch Orchestration (Runner, BatchProcessor, consolidated products)
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure src and repo root are in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Module 1 imports: Ingestion
from astro_exo.ingestion.nasa_archive import fetch_nasa_tois
from astro_exo.ingestion.detrending import flatten_lightcurve, iterative_flatten
from astro_exo.ingestion.search import verify_against_exoplanet_archive
from astro_exo.ingestion.rv_loader import RVDataset, simulate_multi_instrument_rv

# Module 2 imports: Spatial Vetting
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.prf_fit import fit_tess_prf_subpixel, gaussian_2d_prf
from astro_exo.vetting.pixel_lc import extract_pixel_lightcurves, locate_transit_pixel
from astro_exo.vetting.gaia import estimate_tess_mag_from_gaia, query_gaia_neighbors
from astro_exo.vetting.dilution import (
    calculate_critical_delta_mag,
    calculate_max_transit_depth,
    rule_out_neighbors_as_blends,
    calculate_dilution_factor,
    restore_true_radius_ratio
)
from astro_exo.vetting.triceratops_vet import run_triceratops_validation, BayesianFalsePositiveEngine

# Module 3 imports: Bayesian Models
from astro_exo.models.transforms import (
    kipping_to_quadratic,
    quadratic_to_kipping,
    ecc_omega_to_xy,
    xy_to_ecc_omega,
    compute_stellar_density,
    impact_param_to_inclination,
    compute_transit_duration
)
from astro_exo.models.gp_noise import CeleriteGPNoiseModel
from astro_exo.models.diagnostics import compute_gelman_rubin, compute_effective_sample_size
from astro_exo.models.emcee_sampler import EmceeTransitFitter, evaluate_batman_model
from astro_exo.models.jax_nuts import JaxNutsTransitFitter

# Module 4 imports: Keplerian RV & Interiors
from astro_exo.models.joint_rv import (
    keplerian_rv,
    compute_planetary_mass_density,
    classify_planetary_interior,
    JointTransitRVSampler
)

# Module 5 imports: Pipeline & Batch
from astro_exo.pipeline.config import TargetConfig, PipelineConfig
from astro_exo.pipeline.schemas import VettingReport, TransitInferenceResult, FullCandidateProduct
from astro_exo.pipeline.batch import BatchProcessor, BatchTargetResult


class TestAllModulesWithNovelSamples(unittest.TestCase):
    """
    Test suite executing all pipeline modules using small, novel candidate samples.
    """

    @classmethod
    def setUpClass(cls):
        """Prepares novel small samples: confirmed, shallow terrestrial, FP blend, and synthetic eccentric."""
        cls.rng = np.random.default_rng(2026)

        # Novel Sample A: TOI-1011.01 (TIC 114018671) - Shallow Terrestrial / Sub-Neptune
        # P = 2.4705 d, depth ~ 273 ppm (rp/rs ~ 0.0165), duration ~ 1.25 hours
        cls.sample_a = {
            "tic_id": 114018671,
            "name": "TOI-1011.01",
            "period_days": 2.470504,
            "t0_bjd": 2459228.6,
            "duration_hours": 1.25,
            "depth_ppm": 273.0,
            "r_star_rsun": 0.88,
            "m_star_msun": 0.90,
            "expected_disp": "PC"
        }

        # Novel Sample B: TOI-104.01 (TIC 231670397) - Hot Gas Giant for Joint RV
        # P = 4.0873 d, depth ~ 3572 ppm (rp/rs ~ 0.0598), duration ~ 3.2 hours
        cls.sample_b = {
            "tic_id": 231670397,
            "name": "TOI-104.01",
            "period_days": 4.087296,
            "t0_bjd": 2458327.5,
            "duration_hours": 3.2,
            "depth_ppm": 3572.0,
            "r_star_rsun": 1.05,
            "m_star_msun": 1.10,
            "k_rv_ms": 78.5, # Semiamplitude Doppler ~78.5 m/s
            "expected_disp": "CP"
        }

        # Novel Sample C: TOI-1000.01 (TIC 50365310) - Simulated Off-Target BEB False Positive
        cls.sample_c = {
            "tic_id": 50365310,
            "name": "TOI-1000.01",
            "period_days": 2.171348,
            "t0_bjd": 2459229.63,
            "duration_hours": 2.02,
            "depth_ppm": 656.9,
            "r_star_rsun": 2.17,
            "m_star_msun": 1.0,
            "expected_disp": "FP"
        }

        # Novel Sample D: Compact Synthetic Transit with Eccentricity and GP Noise
        cls.sample_d_p = 1.782
        cls.sample_d_e = 0.15
        cls.sample_d_w_deg = 48.0
        cls.sample_d_dur = 2.1 / 24.0

    # -------------------------------------------------------------------------
    # 1. INGESTION & DETRENDING MODULES
    # -------------------------------------------------------------------------
    def test_01_ingestion_nasa_archive_novel_sample(self):
        """Verifies ingestion of novel TOIs from NASA Archive cache with expected columns."""
        tois = fetch_nasa_tois(limit=25, use_cache=True)
        self.assertGreater(len(tois), 0)
        # Verify schema integrity for ingested rows
        row = tois[0]
        for field in ["tic_id", "name", "period_days", "t0_bjd", "duration_hours", "depth_ppm"]:
            self.assertIn(field, row)
            self.assertIsNotNone(row[field])

    def test_02_detrending_wotan_and_archive_verification(self):
        """Verifies light curve detrending and cross-matching for novel sample A."""
        p = self.sample_a["period_days"]
        t0 = self.sample_a["t0_bjd"]
        dur_d = self.sample_a["duration_hours"] / 24.0

        # Create small synthetic time series: 120 points around transit
        t = np.linspace(t0 - 0.25, t0 + 0.25, 120)
        # Add smooth stellar variability: sinusoid with amplitude 1000 ppm
        stellar_trend = 1.0 + 0.001 * np.sin(2.0 * np.pi * (t - t0) / 1.5)
        # Transit signal
        in_transit = np.abs((t - t0 + 0.5 * p) % p - 0.5 * p) < (0.5 * dur_d)
        f_raw = stellar_trend.copy()
        f_raw[in_transit] -= (self.sample_a["depth_ppm"] / 1e6)
        f_raw += self.rng.normal(0, 3e-5, len(t))

        # Iterative detrending
        f_flat, trend, mask = iterative_flatten(t, f_raw, p, t0, dur_d, window_length=0.3)
        self.assertEqual(len(f_flat), len(t))
        self.assertAlmostEqual(float(np.nanmedian(f_flat)), 1.0, places=3)

        # Cross-match against archive (gracefully handles online/offline)
        is_novel, match_str = verify_against_exoplanet_archive(self.sample_a["tic_id"], p, t0)
        self.assertIsInstance(is_novel, bool)
        self.assertTrue("TIC" in match_str or "archive" in match_str.lower() or "vetting" in match_str.lower())

    def test_03_multi_instrument_rv_ingestion(self):
        """Verifies multi-spectrograph Doppler data structuring and simulation."""
        rv_data, ground_truth = simulate_multi_instrument_rv(
            period_days=self.sample_b["period_days"],
            t0_bjd=self.sample_b["t0_bjd"],
            k_semiamp_ms=self.sample_b["k_rv_ms"],
            ecc=0.04,
            omega_deg=30.0,
            n_points_per_inst={"HARPS": 15, "ESPRESSO": 12},
            gamma_offsets={"HARPS": 12.5, "ESPRESSO": -4.2},
            jitters_ms={"HARPS": 1.5, "ESPRESSO": 0.8},
            nominal_errors_ms={"HARPS": 2.0, "ESPRESSO": 1.0},
            baseline_days=30.0,
            random_seed=42
        )
        self.assertIsInstance(rv_data, RVDataset)
        self.assertEqual(len(rv_data), 27)
        self.assertIn("HARPS", rv_data.instruments)
        self.assertIn("ESPRESSO", rv_data.instruments)
        self.assertEqual(ground_truth["k_semiamp_ms"], self.sample_b["k_rv_ms"])

    # -------------------------------------------------------------------------
    # 2. SPATIAL VETTING & SUB-PIXEL LOCALIZATION MODULES
    # -------------------------------------------------------------------------
    def test_04_spatial_vetting_and_prf_fitting(self):
        """Verifies 2D difference imaging, centroid shift, and sub-pixel PRF fitting on novel sample."""
        n_cadences = 80
        ny, nx = 7, 7
        t0 = self.sample_b["t0_bjd"]
        period = self.sample_b["period_days"]
        dur_d = self.sample_b["duration_hours"] / 24.0

        times = np.linspace(t0 - 0.2, t0 + 0.2, n_cadences)
        in_transit = np.abs((times - t0 + 0.5 * period) % period - 0.5 * period) < (0.5 * dur_d)

        # Target located precisely at pixel (3.25, 3.10)
        target_true_pos = (3.25, 3.10)
        flux_tpf = np.full((n_cadences, ny, nx), 1500.0)
        err_tpf = np.full_like(flux_tpf, 5.0)

        y_grid, x_grid = np.mgrid[0:ny, 0:nx]
        base_star = gaussian_2d_prf((x_grid, y_grid), target_true_pos[0], target_true_pos[1], amplitude=8000.0, sigma_x=1.1, sigma_y=1.1)
        flux_tpf += base_star[None, :, :]

        # Drop flux in transit according to PRF deficit
        deficit = base_star * (self.sample_b["depth_ppm"] / 1e6)
        flux_tpf[in_transit] -= deficit[None, :, :]
        flux_tpf += self.rng.normal(0, 4.0, flux_tpf.shape)

        # 1. Difference imaging
        i_out, i_in, i_diff, sigma_diff = calculate_difference_image(times, flux_tpf, err_tpf, period, t0, dur_d)
        self.assertEqual(i_diff.shape, (7, 7))

        # 2. Centroid shift measurement
        cen_res = measure_centroid_offset(i_diff, sigma_diff, target_pix_coord=(3.0, 3.0), tess_pixel_scale_arcsec=21.0, n_mc_perturbations=100)
        self.assertLess(cen_res["offset_arcsec"], 15.0) # Well within sub-pixel radius

        # 3. Sub-pixel PRF fit
        prf_fit_res = fit_tess_prf_subpixel(i_diff, sigma_diff, target_catalog_xy=(3.0, 3.0))
        self.assertAlmostEqual(prf_fit_res["prf_x"], target_true_pos[0], delta=0.5)
        self.assertAlmostEqual(prf_fit_res["prf_y"], target_true_pos[1], delta=0.5)
        self.assertTrue(prf_fit_res["fit_success"])

        # 4. Pixel-by-pixel LC extraction and localization
        pix_lcs, pix_errs = extract_pixel_lightcurves(flux_tpf, err_tpf)
        self.assertEqual(pix_lcs.shape, (7, 7, n_cadences))
        best_y, best_x, max_dip = locate_transit_pixel(times, flux_tpf, period, t0, dur_d)
        self.assertEqual((best_y, best_x), (3, 3))

    def test_05_gaia_dilution_and_triceratops_validation(self):
        """Verifies Gaia magnitude conversion, blend dilution bounds, and TRICERATOPS scoring."""
        # Gaia DR3 synthetic magnitude TIC v8
        tmag = estimate_tess_mag_from_gaia(phot_g=11.2, bp_rp=0.95)
        self.assertGreater(tmag, 9.0)
        self.assertLess(tmag, 13.0)

        # Critical delta magnitude threshold
        depth_ppm = self.sample_b["depth_ppm"]
        delta_m_crit = calculate_critical_delta_mag(depth_ppm)
        self.assertGreater(delta_m_crit, 5.0)

        # Neighbors screening: 1 close contaminating faint star, 1 distant ruled-out star
        neighbors = [
            {"source_id": 1001, "distance_arcsec": 4.5, "phot_g_mean_mag": 15.2, "tess_mag": 14.8},
            {"source_id": 1002, "distance_arcsec": 45.0, "phot_g_mean_mag": 19.5, "tess_mag": 19.0}
        ]
        vetted_nb = rule_out_neighbors_as_blends(depth_ppm, target_mag=tmag, neighbors=neighbors)
        self.assertTrue(vetted_nb[1]["ruled_out"])

        # Dilution factor
        dil_res = calculate_dilution_factor(target_mag=tmag, neighbors=vetted_nb)
        self.assertGreater(dil_res["dilution_factor"], 0.90)

        # True radius ratio restoration
        obs_rp_rs = np.sqrt(depth_ppm / 1e6)
        true_rp_rs, _ = restore_true_radius_ratio(obs_rp_rs, dil_res["dilution_factor"])
        self.assertGreaterEqual(true_rp_rs, obs_rp_rs)

        # TRICERATOPS statistical validation
        val_res = run_triceratops_validation(
            tic_id=self.sample_b["tic_id"],
            sectors=[1],
            period=self.sample_b["period_days"],
            depth=depth_ppm,
            duration_days=self.sample_b["duration_hours"] / 24.0,
            rp_rs=obs_rp_rs,
            target_tmag=tmag,
            centroid_offset_arcsec=0.25,
            centroid_sigma_arcsec=0.10,
            neighbors=vetted_nb
        )
        self.assertIn("fpp", val_res)
        self.assertIn("nfpp", val_res)
        self.assertLess(val_res["fpp"], 0.05)
        self.assertTrue(val_res["validated"])

    # -------------------------------------------------------------------------
    # 3. BAYESIAN INFERENCE & PARAMETER MODELING MODULES
    # -------------------------------------------------------------------------
    def test_06_transforms_and_gp_noise(self):
        """Verifies geometric transforms, density, and GP SHO noise model."""
        # Kipping transforms
        q1, q2 = 0.35, 0.45
        u1, u2 = kipping_to_quadratic(q1, q2)
        q1_rec, q2_rec = quadratic_to_kipping(u1, u2)
        self.assertAlmostEqual(q1, q1_rec, places=5)
        self.assertAlmostEqual(q2, q2_rec, places=5)

        # Eccentricity reparameterization (e, w) <-> (h, k) with omega in radians
        e_in = self.sample_d_e
        w_in_rad = float(np.radians(self.sample_d_w_deg))
        h, k = ecc_omega_to_xy(e_in, w_in_rad)
        e_out, w_out_rad = xy_to_ecc_omega(h, k)
        self.assertAlmostEqual(e_in, e_out, places=5)
        self.assertAlmostEqual(w_in_rad, w_out_rad, places=5)

        # Transit duration & stellar density
        dur = compute_transit_duration(period_days=3.0, rp_rs=0.08, a_rs=14.0, b=0.25)
        self.assertGreater(dur, 1.5)
        self.assertLess(dur, 4.0)

        rho = compute_stellar_density(period_days=365.25, a_rs=215.0)
        self.assertAlmostEqual(rho, 1.408, delta=0.2)

        # GP Noise likelihood
        t_pts = np.linspace(0, 1.0, 50)
        yerr = np.full(50, 0.001)
        residuals = self.rng.normal(0, 0.001, 50)
        gp_model = CeleriteGPNoiseModel(t_pts, yerr)
        log_l = gp_model.compute_gp_log_likelihood(residuals, sigma_gp=0.0005, rho_gp=0.1)
        self.assertTrue(np.isfinite(log_l))

    def test_07_bayesian_mcmc_emcee_and_diagnostics(self):
        """Verifies MCMC sampling via emcee, batman model, and Gelman-Rubin convergence."""
        p = self.sample_a["period_days"]
        t0 = 0.0

        # Small 60-point phase-folded sample
        t_pts = np.linspace(-0.06, 0.06, 60)
        rp_expected = np.sqrt(self.sample_a["depth_ppm"] / 1e6)
        theta_true = [t0, rp_expected, 12.5, 0.2, 0.3, 0.2, 1.0]
        flux_true = evaluate_batman_model(theta_true, t_pts, p)
        flux_obs = flux_true + self.rng.normal(0, 5e-5, len(t_pts))
        flux_err = np.full_like(flux_obs, 5e-5)

        fitter = EmceeTransitFitter(t_pts, flux_obs, flux_err, period=p, t0_expected=t0)
        fitter.run_mcmc(nwalkers=16, nburn=50, nprod=100)
        summary = fitter.get_summary()

        self.assertIn("rp", summary)
        self.assertAlmostEqual(summary["rp"]["median"], rp_expected, delta=0.015)

        diag = fitter.get_diagnostics()
        self.assertIn("r_hat", diag)
        for val in diag["r_hat"].values():
            if np.isfinite(val):
                self.assertLess(val, 2.0)

    def test_08_bayesian_jax_nuts_sampler(self):
        """Verifies differentiable NUTS sampling with JAX/NumPyro on novel sample."""
        p = self.sample_b["period_days"]
        t0 = 0.0

        t_pts = np.linspace(-0.08, 0.08, 50)
        rp_expected = np.sqrt(self.sample_b["depth_ppm"] / 1e6)
        theta_true = [t0, rp_expected, 11.0, 0.15, 0.35, 0.25, 1.0]
        flux_true = evaluate_batman_model(theta_true, t_pts, p)
        flux_obs = flux_true + self.rng.normal(0, 8e-5, len(t_pts))
        flux_err = np.full_like(flux_obs, 8e-5)

        fitter = JaxNutsTransitFitter(t_pts, flux_obs, flux_err, period=p, t0_prior_mean=t0)
        samples = fitter.run_nuts(num_warmup=40, num_samples=60, num_chains=1)
        self.assertIsInstance(samples, dict)
        self.assertIn("rp", samples)
        self.assertIn("a_rs", samples)
        summary = fitter.get_summary()
        self.assertAlmostEqual(summary["rp"]["median"], rp_expected, delta=0.02)

    # -------------------------------------------------------------------------
    # 4. KEPLERIAN DYNAMICS, JOINT RV & INTERIOR CLASSIFICATION
    # -------------------------------------------------------------------------
    def test_09_kepler_solver_and_joint_rv_mcmc(self):
        """Verifies Keplerian RV solver across phases and Joint Transit+RV characterization."""
        p = self.sample_b["period_days"]
        t0 = self.sample_b["t0_bjd"]
        k_amp = self.sample_b["k_rv_ms"]
        gamma = 15.0

        # RV solver verification
        t_eval = np.array([t0, t0 + 0.25 * p, t0 + 0.50 * p, t0 + 0.75 * p])
        rvs = keplerian_rv(t_eval, p, t0, k_amp, ecc=0.0, omega_deg=0.0, gamma=gamma)
        self.assertAlmostEqual(rvs[0], gamma, places=3)
        self.assertAlmostEqual(rvs[1], gamma - k_amp, places=3)
        self.assertAlmostEqual(rvs[2], gamma, places=3)
        self.assertAlmostEqual(rvs[3], gamma + k_amp, places=3)

        # Joint Physical parameter derivation (WASP/Hot Jupiter scale)
        phys = compute_planetary_mass_density(
            m_star_msun=self.sample_b["m_star_msun"],
            r_star_rsun=self.sample_b["r_star_rsun"],
            period_days=p,
            k_semiamp_ms=k_amp,
            rp_rs=np.sqrt(self.sample_b["depth_ppm"] / 1e6),
            ecc=0.0,
            inc_deg=87.5
        )
        self.assertIn("mass_jupiter", phys)
        self.assertIn("radius_jupiter", phys)
        self.assertIn("density_g_cm3", phys)
        self.assertIn("log_g_cgs", phys)
        self.assertGreater(phys["mass_jupiter"], 0.2)
        self.assertGreater(phys["density_g_cm3"], 0.1)

        # Interior classification
        classification = classify_planetary_interior(
            mass_earth=float(phys["mass_earth"]),
            density_g_cm3=float(phys["density_g_cm3"])
        )
        self.assertIn("Jupiter", classification)

        # Joint Transit + RV Sampler execution
        times_rv = np.linspace(t0, t0 + 2.0 * p, 20)
        rv_sim = keplerian_rv(times_rv, p, t0, k_amp, ecc=0.0, omega_deg=0.0, gamma=gamma)
        rv_sim += self.rng.normal(0, 1.5, len(times_rv))
        rv_err = np.full_like(rv_sim, 1.5)

        t_tr = np.linspace(-0.05, 0.05, 40)
        theta_tr = [0.0, 0.06, 11.0, 0.2, 0.3, 0.2, 1.0]
        f_tr = evaluate_batman_model(theta_tr, t_tr, p)
        f_tr += self.rng.normal(0, 5e-5, len(t_tr))
        e_tr = np.full_like(f_tr, 5e-5)

        rv_dataset_test = RVDataset(
            time_bjd=times_rv,
            rv_ms=rv_sim,
            rv_err_ms=rv_err,
            instruments=np.array(["HARPS"] * len(times_rv))
        )
        joint_sampler = JointTransitRVSampler(
            rv_dataset=rv_dataset_test,
            phot_time=t_tr,
            phot_flux=f_tr,
            phot_err=e_tr,
            period_days=p,
            t0_bjd=t0,
            m_star_msun=self.sample_b["m_star_msun"],
            r_star_rsun=self.sample_b["r_star_rsun"],
            rp_rs_prior=0.06
        )
        joint_fit = joint_sampler.run_mcmc(nwalkers=16, nburn=20, nsteps=40)
        self.assertIn("parameters", joint_fit)
        self.assertIn("physical", joint_fit)
        self.assertAlmostEqual(joint_fit["parameters"]["k_semiamp"]["median"], k_amp, delta=15.0)

    # -------------------------------------------------------------------------
    # 5. PIPELINE & BATCH PROCESSING ORCHESTRATION MODULES
    # -------------------------------------------------------------------------
    def test_10_batch_processor_with_novel_catalog(self):
        """Verifies end-to-end batch processing with mock vetting and consolidation on novel catalog."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "novel_candidates.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("tic_id,name,toi,sector,period_days,t0_bjd,duration_hours,depth_ppm,r_star_rsun,m_star_msun,expected_disp\n")
                f.write(f"{self.sample_a['tic_id']},{self.sample_a['name']},1011.01,1,{self.sample_a['period_days']},{self.sample_a['t0_bjd']},{self.sample_a['duration_hours']},{self.sample_a['depth_ppm']},0.88,0.90,PC\n")
                f.write(f"{self.sample_b['tic_id']},{self.sample_b['name']},104.01,1,{self.sample_b['period_days']},{self.sample_b['t0_bjd']},{self.sample_b['duration_hours']},{self.sample_b['depth_ppm']},1.05,1.10,CP\n")
                f.write(f"{self.sample_c['tic_id']},{self.sample_c['name']},1000.01,1,{self.sample_c['period_days']},{self.sample_c['t0_bjd']},{self.sample_c['duration_hours']},{self.sample_c['depth_ppm']},2.17,1.00,FP\n")

            out_dir = os.path.join(tmp_dir, "batch_out")
            batch = BatchProcessor(mode="mock", output_dir=out_dir)
            results = batch.process_catalog(csv_path)

            self.assertEqual(len(results), 3)

            # Check novel sample A (PC) and B (CP) passed
            res_a = next(r for r in results if r.tic_id == self.sample_a["tic_id"])
            self.assertEqual(res_a.status, "PASSED")
            self.assertTrue(res_a.spatial_vetting_passed)

            res_b = next(r for r in results if r.tic_id == self.sample_b["tic_id"])
            self.assertEqual(res_b.status, "PASSED")
            self.assertTrue(res_b.spatial_vetting_passed)

            # Check novel sample C (FP BEB) was properly rejected by spatial vetting
            res_c = next(r for r in results if r.tic_id == self.sample_c["tic_id"])
            self.assertEqual(res_c.status, "REJECTED_FP")
            self.assertFalse(res_c.spatial_vetting_passed)
            self.assertGreater(res_c.centroid_offset_arcsec, 3.0)

            # Check output files were created
            self.assertTrue(os.path.exists(os.path.join(out_dir, "batch_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "batch_summary.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, f"TIC_{self.sample_a['tic_id']}.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, f"TIC_{self.sample_b['tic_id']}.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, f"TIC_{self.sample_c['tic_id']}.json")))


if __name__ == "__main__":
    unittest.main()
