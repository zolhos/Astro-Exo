"""
Test suite validating real TPF sub-pixel difference image centroid vetting
on actual TESS FITS data without any synthetic or mock inputs.
"""

import os
import unittest
import numpy as np

from astro_exo.ingestion.fits_reader import read_tess_tpf
from astro_exo.vetting.difference_img import vet_target_pixel_file


class TestRealCentroidBenchmark(unittest.TestCase):

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.tpf_pass = os.path.join(self.project_root, "data", "photometry", "TIC_25155310", "tess_tic25155310_s0027_tp.fits")
        self.tpf_fail = os.path.join(self.project_root, "data", "photometry", "TIC_107782586", "tess_tic107782586_s0034_tp.fits")

    def test_positive_control_wasp126b_passes(self):
        """WASP-126b must be validated as an on-target transit (PASS) on real TESS FITS."""
        if not os.path.exists(self.tpf_pass):
            self.skipTest("Real TPF for WASP-126b not available.")

        tpf = read_tess_tpf(self.tpf_pass)
        res = vet_target_pixel_file(
            tpf_data=tpf,
            period=3.28879,
            t0=2037.8986,
            duration_hours=2.68,
            max_allowed_offset_arcsec=4.0
        )

        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "PASS")
        self.assertLess(res["offset_pix"], 0.20)
        self.assertLess(res["offset_arcsec"], 4.0)

    def test_negative_control_toi1009_fails(self):
        """TOI-1009.01 in Sector 34 must detect significant centroid shift (> 6 arcsec, 11 sigma) on real FITS."""
        if not os.path.exists(self.tpf_fail):
            self.skipTest("Real TPF for TOI-1009.01 not available.")

        tpf = read_tess_tpf(self.tpf_fail)
        p = 1.960028
        t0_bjd = 2459229.2309
        t_min = np.nanmin(tpf["time"])
        n_ep = round((t_min - (t0_bjd - 2457000.0)) / p)
        t0_s34 = (t0_bjd - 2457000.0) + n_ep * p

        res = vet_target_pixel_file(
            tpf_data=tpf,
            period=p,
            t0=t0_s34,
            duration_hours=2.01,
            max_allowed_offset_arcsec=4.0
        )

        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "FAIL_POSSIBLE_NEB")
        self.assertGreater(res["offset_arcsec"], 4.0)
        self.assertGreater(res["offset_significance_sigma"], 5.0)


if __name__ == "__main__":
    unittest.main()
