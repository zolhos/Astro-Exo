"""
Unit tests for the 22 New Real Observational RV Datasets and NASA Archive Metadata.
Validates:
  - 14 Bonomo et al. 2023 (HARPS-N) Doppler datasets
  - 8 WASP (CORALIE/HARPS) Doppler datasets
  - NASA Exoplanet Archive parameters (pscomppars)
  - Data ingestion via rv_loader.load_rv_csv
"""

import unittest
import os
import json
import numpy as np
from astro_exo.ingestion.rv_loader import load_rv_csv, RVDataset


class TestReal22NewRVDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.metadata_file = os.path.join(cls.repo_root, "data", "real_22_new_targets_metadata.json")
        cls.rv_dir = os.path.join(cls.repo_root, "data", "rv_data")

    def test_metadata_file_exists_and_complete(self):
        self.assertTrue(os.path.isfile(self.metadata_file), "Metadata file missing")
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["n_systems"], 22)
        self.assertEqual(len(data["targets"]), 22)
        self.assertEqual(data["total_rv_observations"], 1746)

    def test_all_22_rv_csv_files_loadable_and_valid(self):
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

            # Check no NaN values
            self.assertFalse(np.isnan(ds.time_bjd).any(), f"NaN in time_bjd for {target['system_name']}")
            self.assertFalse(np.isnan(ds.rv_ms).any(), f"NaN in rv_ms for {target['system_name']}")
            self.assertFalse(np.isnan(ds.rv_err_ms).any(), f"NaN in rv_err_ms for {target['system_name']}")

            # Check all error bars strictly positive
            self.assertTrue(np.all(ds.rv_err_ms > 0), f"Non-positive errors for {target['system_name']}")

            # Check chronological ordering
            self.assertTrue(np.all(np.diff(ds.time_bjd) >= 0), f"Times not sorted for {target['system_name']}")

            # Check baseline
            self.assertAlmostEqual(ds.baseline_days, target["baseline_days"], places=4)

    def test_all_22_targets_have_valid_astrophysical_parameters(self):
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
            self.assertGreater(target["tic_number"], 0)

            # Stellar parameters
            self.assertIsNotNone(target["r_star_rsun"], f"Missing r_star for {sname}")
            self.assertGreater(target["r_star_rsun"], 0.0)
            self.assertIsNotNone(target["m_star_msun"], f"Missing m_star for {sname}")
            self.assertGreater(target["m_star_msun"], 0.0)


if __name__ == "__main__":
    unittest.main()
