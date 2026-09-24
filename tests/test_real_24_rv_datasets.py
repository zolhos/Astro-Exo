"""
Tests for the 24 Real Doppler Radial Velocity Datasets and Metadata.
Validates VizieR J/A+A/677/A33 HARPS-N observations and NASA Exoplanet Archive parameters.
"""

import unittest
import os
import json
import numpy as np
from astro_exo.ingestion.rv_loader import load_rv_csv, RVDataset


class TestReal24RVDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.metadata_file = os.path.join(cls.repo_root, "data", "real_24_rv_targets_metadata.json")
        cls.rv_dir = os.path.join(cls.repo_root, "data", "rv_data")

    def test_metadata_file_exists_and_valid(self):
        self.assertTrue(os.path.isfile(self.metadata_file), "Metadata file missing")
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["n_systems"], 24)
        self.assertEqual(len(data["targets"]), 24)
        self.assertGreater(data["total_rv_observations"], 2000)

    def test_all_24_rv_csv_files_loadable_and_non_empty(self):
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for target in data["targets"]:
            rel_path = target["rv_file"]
            abs_path = os.path.join(self.repo_root, rel_path)
            self.assertTrue(os.path.isfile(abs_path), f"RV CSV file not found: {abs_path}")

            ds = load_rv_csv(abs_path)
            self.assertIsInstance(ds, RVDataset)
            self.assertEqual(ds.n_points, target["n_rv_observations"])
            self.assertGreaterEqual(ds.n_points, 15)
            self.assertIn("HARPS-N", ds.instrument_names)

            # Check no NaN values
            self.assertFalse(np.isnan(ds.time_bjd).any(), f"NaN in time_bjd for {target['system_name']}")
            self.assertFalse(np.isnan(ds.rv_ms).any(), f"NaN in rv_ms for {target['system_name']}")
            self.assertFalse(np.isnan(ds.rv_err_ms).any(), f"NaN in rv_err_ms for {target['system_name']}")

            # Check chronological ordering
            self.assertTrue(np.all(np.diff(ds.time_bjd) >= 0), f"Times not sorted for {target['system_name']}")

    def test_all_targets_have_valid_astrophysical_parameters(self):
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for target in data["targets"]:
            sname = target["system_name"]
            self.assertIsNotNone(target["period_days"], f"Missing period for {sname}")
            self.assertGreater(target["period_days"], 0.0)

            self.assertIsNotNone(target["t0_bjd"], f"Missing t0 for {sname}")
            self.assertGreater(target["t0_bjd"], 2450000.0)

            self.assertIsNotNone(target["rp_rearth"], f"Missing rp for {sname}")
            self.assertGreater(target["rp_rearth"], 0.0)

            self.assertIsNotNone(target["mp_mearth"], f"Missing mp for {sname}")
            self.assertGreater(target["mp_mearth"], 0.0)

            self.assertIsNotNone(target["transit_duration_hours"], f"Missing dur for {sname}")
            self.assertGreater(target["transit_duration_hours"], 0.0)

            self.assertTrue(target["tic_id"].startswith("TIC"), f"Invalid TIC ID for {sname}")
            self.assertIsNotNone(target["tic_number"], f"Invalid TIC number for {sname}")

    def test_consolidated_28_real_systems_catalog(self):
        cat_path = os.path.join(self.repo_root, "docs", "assets", "consolidated_catalog.json")
        csv_path = os.path.join(self.repo_root, "docs", "assets", "consolidated_catalog.csv")
        self.assertTrue(os.path.isfile(cat_path), "consolidated_catalog.json missing")
        self.assertTrue(os.path.isfile(csv_path), "consolidated_catalog.csv missing")

        with open(cat_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        self.assertEqual(len(catalog), 28, "Catalog must contain exactly 28 real exoplanet systems")

        # 4 Hot Jupiters + 24 Kepler/K2 systems
        wasp_names = ["WASP-77b", "WASP-126b", "WASP-62b", "WASP-46b"]
        kepler_k2_names = [
            "Kepler-10b", "Kepler-19b", "Kepler-20b", "Kepler-21b", "Kepler-22b",
            "Kepler-78b", "Kepler-93b", "Kepler-102b", "Kepler-103b", "Kepler-107b",
            "Kepler-109b", "Kepler-409b", "Kepler-454b", "Kepler-538b", "Kepler-1655b",
            "K2-2b", "K2-3b", "K2-36b", "K2-38b", "K2-110b", "K2-111b", "K2-131b",
            "K2-141b", "K2-222b"
        ]
        all_expected = wasp_names + kepler_k2_names
        actual_names = [t["name"] for t in catalog]

        for exp in all_expected:
            self.assertIn(exp, actual_names, f"Target {exp} missing from consolidated catalog")

        # Verify physical parameters and RV files
        for tgt in catalog:
            name = tgt["name"]
            self.assertEqual(tgt["status"], "PASSED")
            self.assertGreater(tgt["period_days"], 0.0)
            self.assertGreater(tgt["radius_earth"], 0.0)
            self.assertGreater(tgt["radius_jupiter"], 0.0)
            self.assertGreater(tgt["mass_earth"], 0.0)
            self.assertGreater(tgt["mass_jupiter"], 0.0)
            self.assertGreater(tgt["density_g_cm3"], 0.0)
            self.assertGreater(tgt["k_semiamp_ms"], 0.0)
            self.assertGreater(tgt["transit_duration_hours"], 0.0)
            self.assertGreater(tgt["depth_ppm"], 0.0)
            self.assertGreater(tgt["tic_id"], 0)
            self.assertGreater(tgt["r_star_rsun"], 0.0)
            self.assertGreater(tgt["m_star_msun"], 0.0)

            # RV file exists
            rv_f = os.path.join(self.repo_root, tgt["rv_file"])
            self.assertTrue(os.path.isfile(rv_f), f"RV file for {name} missing: {rv_f}")

    def test_all_168_scientific_figures_exist(self):
        cat_path = os.path.join(self.repo_root, "docs", "assets", "consolidated_catalog.json")
        with open(cat_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        req_tabs = ["transit", "corner", "diff_img", "gaia", "triceratops", "rv"]
        total_figs = 0

        for tgt in catalog:
            name = tgt["name"]
            figs = tgt.get("figures", {})
            for tab in req_tabs:
                self.assertIn(tab, figs, f"Tab {tab} missing in figures for {name}")
                fig_path = os.path.join(self.repo_root, "docs", figs[tab])
                self.assertTrue(os.path.isfile(fig_path), f"Figure for {name} ({tab}) missing: {fig_path}")
                self.assertGreater(os.path.getsize(fig_path), 5000, f"Figure too small / empty: {fig_path}")
                total_figs += 1

        self.assertEqual(total_figs, 28 * 6)  # 168 scientific figures


if __name__ == "__main__":
    unittest.main()

