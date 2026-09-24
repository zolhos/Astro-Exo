import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astro_exo.ingestion.fits_reader import read_tess_lightcurve, read_tess_tpf
from astro_exo.ingestion.detrending import flatten_lightcurve
from astro_exo.ingestion.mast_client import fetch_tess_photometry
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

    def test_05_err_raw_validation(self):
        """Valida que flux_err inválido (<= 0 ou NaN) é filtrado por good_mask."""
        with tempfile.NamedTemporaryFile(suffix="_lc.fits") as f:
            tbl = Table()
            tbl["TIME"] = np.arange(12, dtype=np.float64)
            tbl["PDCSAP_FLUX"] = np.array([100.0] * 12)
            tbl["PDCSAP_FLUX_ERR"] = np.array([1.0, np.nan, -0.5, 0.0, 1.2, 1.1, 1.0, 0.9, 1.3, 1.0, 1.1, 1.0])
            tbl["QUALITY"] = np.zeros(12, dtype=int)
            tbl["CADENCENO"] = np.arange(12, dtype=int)

            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header["TICID"] = 12345
            primary_hdu.header["SECTOR"] = 1
            bin_table = fits.BinTableHDU(tbl)
            hdul = fits.HDUList([primary_hdu, bin_table])
            hdul.writeto(f.name, overwrite=True)

            res = read_tess_lightcurve(f.name)
            self.assertEqual(len(res["time"]), 9)
            self.assertTrue(np.all(res["flux_err"] > 0))
            self.assertFalse(np.any(np.isnan(res["flux_err"])))

    def test_06_tpf_nan_error_inf(self):
        """Valida que NaNs em FLUX_ERR do TPF são substituídos por np.inf."""
        with tempfile.NamedTemporaryFile(suffix="_tp.fits") as f:
            tbl = Table()
            tbl["TIME"] = np.array([1.0, 2.0])
            flux = np.ones((2, 3, 3), dtype=np.float32)
            err = np.ones((2, 3, 3), dtype=np.float32)
            err[0, 1, 1] = np.nan
            tbl["FLUX"] = flux
            tbl["FLUX_ERR"] = err
            tbl["QUALITY"] = np.zeros(2, dtype=int)

            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header["TICID"] = 12345
            primary_hdu.header["SECTOR"] = 1
            bin_table = fits.BinTableHDU(tbl)
            hdul = fits.HDUList([primary_hdu, bin_table])
            hdul.writeto(f.name, overwrite=True)

            res_tpf = read_tess_tpf(f.name)
            self.assertTrue(np.isinf(res_tpf["flux_err"][0, 1, 1]))
            self.assertFalse(np.any(np.isnan(res_tpf["flux_err"])))

    def test_07_detrending_wotan_break_tolerance(self):
        """Valida que flatten_lightcurve funciona com wotan e break_tolerance=0.5."""
        if not os.path.exists(self.lc_path):
            self.skipTest("Arquivo FITS de curva de luz não encontrado.")

        lc = read_tess_lightcurve(self.lc_path)
        f_flat, trend = flatten_lightcurve(
            time=lc["time"],
            flux=lc["flux"],
            method="biweight",
            window_length=0.75
        )
        self.assertEqual(len(f_flat), len(lc["flux"]))
        self.assertAlmostEqual(np.nanmedian(f_flat), 1.0, places=3)

    def test_08_detrending_savgol_fallback_and_transit_mask(self):
        """Valida o fallback Savitzky-Golay com cadência real e interpolação de máscara de trânsito."""
        time = np.linspace(0, 10, 500)
        flux = np.ones(500)
        transit_mask = (time > 4.8) & (time < 5.2)
        flux[transit_mask] -= 0.02

        with patch.dict("sys.modules", {"wotan": None}):
            f_flat_nomask, _ = flatten_lightcurve(time, flux, transit_mask=None)
            f_flat_mask, _ = flatten_lightcurve(time, flux, transit_mask=transit_mask)

        depth_mask = 1.0 - np.min(f_flat_mask)
        depth_nomask = 1.0 - np.min(f_flat_nomask)

        self.assertAlmostEqual(depth_mask, 0.02, delta=0.001)
        self.assertGreater(depth_mask, depth_nomask * 1.5)

    def test_09_mast_cache_priority(self):
        """Valida que fetch_tess_photometry prioriza cache local antes de requisições de rede."""
        with patch("astro_exo.ingestion.mast_client.resolve_tess_download_urls", side_effect=RuntimeError("Rede chamada indevidamente")):
            lc_path = fetch_tess_photometry(25155310, sector=27, product="lc")
            self.assertTrue(os.path.exists(lc_path))
            self.assertGreater(os.path.getsize(lc_path), 10000)

            lc_path_nosec = fetch_tess_photometry(25155310, product="lc")
            self.assertTrue(os.path.exists(lc_path_nosec))

            tp_path = fetch_tess_photometry(25155310, sector=27, product="tp")
            self.assertTrue(os.path.exists(tp_path))


if __name__ == "__main__":
    unittest.main()
