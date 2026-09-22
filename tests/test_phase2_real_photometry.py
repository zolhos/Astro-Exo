import os
import unittest
import numpy as np
from astro_exo.ingestion.fits_reader import read_tess_lightcurve, read_tess_tpf
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset, vet_target_pixel_file


class TestPhase2RealPhotometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data/photometry/TIC_25155310"
        cls.lc_path = os.path.join(cls.data_dir, "tess_tic25155310_s0027_lc.fits")
        cls.tpf_path = os.path.join(cls.data_dir, "tess_tic25155310_s0027_tp.fits")
        cls.period = 3.28879
        cls.t0 = 2037.8986
        cls.duration_hours = 2.68

    def test_01_read_lightcurve_fits(self):
        if not os.path.exists(self.lc_path):
            self.skipTest("Arquivo FITS de curva de luz não encontrado no disco local.")

        lc = read_tess_lightcurve(self.lc_path)
        self.assertEqual(lc["tic_id"], 25155310)
        self.assertEqual(lc["sector"], 27)
        self.assertGreater(len(lc["time"]), 10000)
        self.assertAlmostEqual(np.nanmedian(lc["flux"]), 1.0, places=3)
        self.assertFalse(np.any(np.isnan(lc["flux"])))
        self.assertFalse(np.any(np.isnan(lc["time"])))

    def test_02_wotan_detrending(self):
        if not os.path.exists(self.lc_path):
            self.skipTest("Arquivo FITS de curva de luz não encontrado.")

        import wotan
        lc = read_tess_lightcurve(self.lc_path)
        time = lc["time"]
        flux = lc["flux"]

        flatten_lc, trend_lc = wotan.flatten(
            time,
            flux,
            method="biweight",
            window_length=0.75,
            edge_cutoff=0.5,
            break_tolerance=0.5,
            return_trend=True
        )

        self.assertEqual(len(flatten_lc), len(flux))
        self.assertAlmostEqual(np.nanmedian(flatten_lc), 1.0, places=3)

        phase = (time - self.t0 + 0.5 * self.period) % self.period - 0.5 * self.period
        in_transit = np.abs(phase) < (0.5 * self.duration_hours / 24.0)
        out_transit = (np.abs(phase) > (0.75 * self.duration_hours / 24.0)) & (np.abs(phase) < (1.75 * self.duration_hours / 24.0))

        depth = np.nanmedian(flatten_lc[out_transit]) - np.nanmedian(flatten_lc[in_transit])
        depth_ppm = depth * 1e6
        self.assertGreater(depth_ppm, 4000.0)
        self.assertLess(depth_ppm, 10000.0)

    def test_03_read_tpf_and_wcs(self):
        if not os.path.exists(self.tpf_path):
            self.skipTest("Arquivo TPF não encontrado no disco local.")

        tpf = read_tess_tpf(self.tpf_path)
        self.assertEqual(tpf["tic_id"], 25155310)
        self.assertEqual(tpf["sector"], 27)
        self.assertEqual(tpf["spatial_shape"], (11, 11))
        self.assertIsNotNone(tpf["target_pix"])

        target_x, target_y = tpf["target_pix"]
        self.assertAlmostEqual(target_x, 4.70, delta=0.1)
        self.assertAlmostEqual(target_y, 4.19, delta=0.1)

    def test_04_spatial_difference_vetting(self):
        if not os.path.exists(self.tpf_path):
            self.skipTest("Arquivo TPF não encontrado.")

        tpf = read_tess_tpf(self.tpf_path)
        result = vet_target_pixel_file(
            tpf_data=tpf,
            period=self.period,
            t0=self.t0,
            duration_hours=self.duration_hours,
            max_allowed_offset_arcsec=10.0
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["offset_pix"], 0.3)
        self.assertLess(result["offset_arcsec"], 6.0)


if __name__ == "__main__":
    unittest.main()
