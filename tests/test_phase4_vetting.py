"""
Unit tests for Phase 4: Gaia DR3 multispectral vetting, dilution calculations,
de-dilution transit restoration (WASP-77b), and TRICERATOPS Bayesian false positive validation.
"""

import unittest
import numpy as np

from astro_exo.vetting.gaia import (
    estimate_tess_mag_from_gaia,
    query_gaia_neighbors
)
from astro_exo.vetting.dilution import (
    calculate_critical_delta_mag,
    calculate_max_transit_depth,
    estimate_aperture_flux_fraction,
    rule_out_neighbors_as_blends,
    calculate_dilution_factor,
    restore_true_radius_ratio,
    correct_binary_blend_wasp77
)
from astro_exo.vetting.triceratops_vet import (
    BayesianFalsePositiveEngine,
    run_triceratops_validation
)


class TestPhase4Vetting(unittest.TestCase):

    def test_estimate_tess_mag_from_gaia(self):
        """Test empirical Gaia DR3 to TESS magnitude transformation (TIC v8)."""
        # Solar-type star: G = 10.0, BP - RP = 0.82
        tmag_solar = estimate_tess_mag_from_gaia(phot_g=10.0, bp_rp=0.82)
        self.assertAlmostEqual(tmag_solar, 9.53, delta=0.10)

        # Red M dwarf: G = 14.0, BP - RP = 2.50
        tmag_red = estimate_tess_mag_from_gaia(phot_g=14.0, bp_rp=2.50)
        self.assertLess(tmag_red, 13.5)  # TESS is red-sensitive, so Tmag < G

        # Missing color fallback
        tmag_fallback = estimate_tess_mag_from_gaia(phot_g=10.0, bp_rp=None)
        self.assertAlmostEqual(tmag_fallback, 9.57, delta=0.01)

    def test_calculate_critical_delta_mag(self):
        """Test Delta m_crit threshold calculation for 100% eclipse bounding."""
        # 10,000 ppm (1% transit) -> Delta m_crit = -2.5 * log10(0.01) = 5.0
        delta_m_1pct = calculate_critical_delta_mag(10000.0)
        self.assertAlmostEqual(delta_m_1pct, 5.0, places=2)

        # 1,000 ppm (0.1% transit) -> Delta m_crit = -2.5 * log10(0.001) = 7.5
        delta_m_01pct = calculate_critical_delta_mag(1000.0)
        self.assertAlmostEqual(delta_m_01pct, 7.5, places=2)

        # 20,000 ppm (2% transit) -> Delta m_crit = -2.5 * log10(0.02) = 4.247
        delta_m_2pct = calculate_critical_delta_mag(20000.0)
        self.assertAlmostEqual(delta_m_2pct, 4.247, places=2)

    def test_rule_out_neighbors_as_blends(self):
        """Test screening of faint neighbors based on Delta m_crit."""
        target_mag = 10.0
        observed_depth_ppm = 10000.0  # 1% transit, Delta m_crit = 5.0

        neighbors = [
            # Star A: Delta mag = 3.0 (Tmag = 13.0) at 10 arcsec -> CAN cause transit
            {"source_id": 1, "tess_mag": 13.0, "dist_arcsec": 10.0},
            # Star B: Delta mag = 6.0 (Tmag = 16.0) at 10 arcsec -> CANNOT cause transit (ruled out)
            {"source_id": 2, "tess_mag": 16.0, "dist_arcsec": 10.0},
            # Star C: Delta mag = 2.0 (Tmag = 12.0) at 100 arcsec (outside aperture) -> ruled out
            {"source_id": 3, "tess_mag": 12.0, "dist_arcsec": 100.0}
        ]

        vetted = rule_out_neighbors_as_blends(
            observed_transit_depth_ppm=observed_depth_ppm,
            target_mag=target_mag,
            neighbors=neighbors
        )

        self.assertTrue(vetted[0]["can_cause_transit"])
        self.assertFalse(vetted[0]["ruled_out"])

        self.assertFalse(vetted[1]["can_cause_transit"])
        self.assertTrue(vetted[1]["ruled_out"])

    def test_wasp77_dedilution(self):
        """Test analytical transit de-dilution for WASP-77b (WASP-77A + WASP-77B at 3.3 arcsec)."""
        res = correct_binary_blend_wasp77(
            rp_rs_diluted=0.1186,
            rp_rs_err=0.0019,
            delta_tmag=1.52
        )

        self.assertAlmostEqual(res["dilution_factor"], 0.8022, places=3)
        self.assertAlmostEqual(res["flux_ratio_B_over_A"], 0.2466, places=3)
        # Diluted Rp/Rs = 0.1186 -> True Rp/Rs = 0.1186 / sqrt(0.8022) ~= 0.1324
        self.assertAlmostEqual(res["true_rp_rs"], 0.1324, delta=0.001)
        self.assertAlmostEqual(res["true_rp_rs_err"], 0.0021, delta=0.0003)
        # Depth restored from ~14,066 ppm to ~17,535 ppm
        self.assertGreater(res["true_depth_ppm"], 17000.0)
        self.assertAlmostEqual(res["depth_correction_percent"], 24.66, delta=0.5)

    def test_triceratops_bayesian_clean_planet(self):
        """Test that a clean, on-target Hot Jupiter passes statistical validation (FPP < 1%, NFPP < 1e-3)."""
        engine = BayesianFalsePositiveEngine(
            target_tmag=9.71,
            period_days=4.4119,
            depth_ppm=12458.0,
            duration_hours=3.78,
            rp_rs=0.1116,
            centroid_offset_arcsec=0.25,
            centroid_sigma_arcsec=0.60,
            neighbors=[
                {"source_id": 1, "tess_mag": 16.5, "dist_arcsec": 52.3, "can_cause_transit": False}
            ]
        )
        res = engine.calculate_scenario_probabilities()

        self.assertLess(res["fpp"], 0.010, "Clean planet must have FPP < 1%")
        self.assertLess(res["nfpp"], 1e-3, "Clean planet must have NFPP < 0.1%")
        self.assertTrue(res["validated"], "Clean planet must pass validation")
        self.assertGreater(res["probabilities"]["TP"], 0.90)

    def test_triceratops_bayesian_false_positive_toi1019(self):
        """Test that TOI-1019.01 (offset 16.36 arcsec towards neighbor) is classified as a False Positive."""
        neighbors = [
            # Neighbor at 16.36 arcsec matching the difference deficit
            {"source_id": 341420329002, "tess_mag": 12.20, "dist_arcsec": 16.36, "can_cause_transit": True}
        ]
        engine = BayesianFalsePositiveEngine(
            target_tmag=10.63,
            period_days=5.2341,
            depth_ppm=19830.0,
            duration_hours=3.71,
            rp_rs=0.1408,
            centroid_offset_arcsec=16.36,
            centroid_sigma_arcsec=1.20,
            neighbors=neighbors
        )
        res = engine.calculate_scenario_probabilities()

        self.assertFalse(res["validated"], "TOI-1019.01 must NOT pass validation")
        self.assertGreater(res["nfpp"], 0.50, "NFPP must be high due to nearby BEB")
        self.assertGreater(res["probabilities"]["BEB"], 0.50, "BEB must be dominant scenario")
        self.assertLess(res["probabilities"]["TP"], 0.05, "TP must be ruled out by offset")

    def test_gaia_cache_retrieval(self):
        """Test loading cached Gaia DR3 stars for benchmark targets."""
        stars = query_gaia_neighbors(
            ra_deg=63.37394,
            dec_deg=-69.22682,
            target_tic=25155310
        )
        self.assertGreaterEqual(len(stars), 4)
        self.assertEqual(stars[0]["dist_arcsec"], 0.0)
        self.assertIn("tess_mag", stars[0])

    def test_triceratops_2d_vector_breaks_ring_degeneracy(self):
        """Test that 2D Euclidean vector distance distinguishes between opposite neighbors at equal scalar distance."""
        # Deficit is at (+10.0", 0.0")
        centroid_vec = (10.0, 0.0)

        # Star A is at (+10.0", 0.0") -> exact match
        neighbor_a = [{"source_id": 101, "tess_mag": 12.0, "dist_arcsec": 10.0, "d_ra_arcsec": 10.0, "d_dec_arcsec": 0.0, "can_cause_transit": True}]
        engine_a = BayesianFalsePositiveEngine(
            target_tmag=10.0,
            period_days=3.0,
            depth_ppm=15000.0,
            duration_hours=2.5,
            rp_rs=0.12,
            centroid_offset_arcsec=10.0,
            centroid_sigma_arcsec=1.0,
            centroid_vec_arcsec=centroid_vec,
            neighbors=neighbor_a
        )
        res_a = engine_a.calculate_scenario_probabilities()

        # Star B is at opposite position (-10.0", 0.0") -> 20 arcsec separation from deficit
        neighbor_b = [{"source_id": 102, "tess_mag": 12.0, "dist_arcsec": 10.0, "d_ra_arcsec": -10.0, "d_dec_arcsec": 0.0, "can_cause_transit": True}]
        engine_b = BayesianFalsePositiveEngine(
            target_tmag=10.0,
            period_days=3.0,
            depth_ppm=15000.0,
            duration_hours=2.5,
            rp_rs=0.12,
            centroid_offset_arcsec=10.0,
            centroid_sigma_arcsec=1.0,
            centroid_vec_arcsec=centroid_vec,
            neighbors=neighbor_b
        )
        res_b = engine_b.calculate_scenario_probabilities()

        # Star A must have significant BEB probability, while Star B must have negligible BEB probability
        self.assertGreater(res_a["probabilities"]["BEB"], 0.40, "Star A matches 2D deficit vector and must have high BEB")
        self.assertLess(res_b["probabilities"]["BEB"], 0.01, "Star B is on opposite side (20 arcsec from deficit) and must have low BEB")


if __name__ == "__main__":
    unittest.main()

