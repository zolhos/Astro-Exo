"""
Tests for difference imaging, centroid offset, and neighbor dilution calculations.
"""

import numpy as np
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.dilution import calculate_max_transit_depth, rule_out_neighbors_as_blends


def test_calculate_difference_image_synthetic():
    """Verify difference imaging on synthetic pixel time-series."""
    n_cadences = 100
    ny, nx = 5, 5
    time = np.linspace(0, 10, n_cadences)
    period = 5.0
    t0 = 2.5
    dur_days = 0.5

    # Out of transit flux is 1000 across all pixels
    flux = np.full((n_cadences, ny, nx), 1000.0)
    flux_err = np.full((n_cadences, ny, nx), 1.0)

    # In transit, inject a 10% dip at center pixel (2, 2)
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) <= (0.5 * dur_days)
    flux[in_transit, 2, 2] -= 100.0

    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        time, flux, flux_err, period, t0, dur_days
    )

    assert i_diff[2, 2] > 50.0
    assert np.all(i_diff[0, 0] < 10.0)


def test_centroid_offset_synthetic():
    """Verify center-of-light aligns with injected pixel deficit."""
    ny, nx = 7, 7
    i_diff = np.zeros((ny, nx))
    sigma_diff = np.full((ny, nx), 0.1)

    # Place deficit at (x=4, y=3)
    i_diff[3, 4] = 100.0
    i_diff[3, 3] = 20.0
    i_diff[3, 5] = 20.0

    target_pix = (4.0, 3.0)
    res = measure_centroid_offset(i_diff, sigma_diff, target_pix, tess_pixel_scale_arcsec=21.0)

    assert np.isclose(res["x_diff_cen"], 4.0, atol=0.1)
    assert np.isclose(res["y_diff_cen"], 3.0, atol=0.1)
    assert res["offset_arcsec"] < 3.0  # Within 3 arcsec
    assert res["offset_significance_sigma"] < 2.0


def test_dilution_and_ruling_out_neighbors():
    """Verify screening of faint neighbors that cannot produce the transit depth."""
    target_mag = 10.0
    # Neighbor is 5 magnitudes fainter (delta_mag = 5 -> flux factor 100x fainter)
    neighbor_mag = 15.0
    max_depth = calculate_max_transit_depth(target_mag, neighbor_mag)
    # Expected max depth: ~ 1 / 101 ~= 0.0099 (9900 ppm)
    assert 0.009 < max_depth < 0.011

    # If observed transit is 20,000 ppm (2%), a 15th magnitude star CANNOT produce it
    neighbors = [{"source_id": 12345, "phot_g_mean_mag": 15.0}]
    vetted = rule_out_neighbors_as_blends(
        observed_transit_depth_ppm=20000.0,
        target_mag=target_mag,
        neighbors=neighbors
    )
    assert vetted[0]["ruled_out"] is True
    assert vetted[0]["can_cause_transit"] is False


def test_centroid_conjunction_acceptance():
    """Verify vet_target_pixel_file requires simultaneous angular and statistical compliance."""
    from astro_exo.vetting.difference_img import vet_target_pixel_file

    n_cadences = 100
    ny, nx = 7, 7
    time = np.linspace(0, 10, n_cadences)
    period = 5.0
    t0 = 2.5
    dur_days = 0.5

    # Case A: On-target deficit at target pixel (3.0, 3.0)
    flux_on = np.full((n_cadences, ny, nx), 1000.0)
    flux_err = np.full((n_cadences, ny, nx), 1.0)
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) <= (0.5 * dur_days)
    flux_on[in_transit, 3, 3] -= 100.0

    tpf_on = {
        "time": time,
        "flux": flux_on,
        "flux_err": flux_err,
        "target_pix": (3.0, 3.0)
    }
    res_on = vet_target_pixel_file(tpf_on, period, t0, dur_days * 24.0, max_allowed_offset_arcsec=4.0)
    assert res_on["passed"] is True
    assert res_on["status"] == "PASS"
    assert res_on["offset_arcsec"] < 1.0

    # Case B: Significant off-target contaminant at pixel (5.0, 3.0), ~42 arcsec offset
    flux_off = np.full((n_cadences, ny, nx), 1000.0)
    flux_off[in_transit, 3, 5] -= 100.0
    tpf_off = {
        "time": time,
        "flux": flux_off,
        "flux_err": flux_err,
        "target_pix": (3.0, 3.0)
    }
    res_off = vet_target_pixel_file(tpf_off, period, t0, dur_days * 24.0, max_allowed_offset_arcsec=4.0)
    assert res_off["passed"] is False
    assert res_off["status"] == "FAIL_POSSIBLE_NEB"
    assert res_off["offset_arcsec"] > 35.0


def test_prf_fit_2d_hessian_uncertainty():
    """Verify PRF subpixel fitting with 2D Hessian computes stable, positive uncertainties."""
    from astro_exo.vetting.prf_fit import fit_tess_prf_subpixel, gaussian_2d_prf

    ny, nx = 7, 7
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    xy_grid = (x_coords, y_coords)
    i_diff = gaussian_2d_prf(xy_grid, 3.2, 3.1, 400.0, 1.1, 1.1, 0.0, 0.0)
    sigma_diff = np.full((ny, nx), 1.0)

    res = fit_tess_prf_subpixel(i_diff, sigma_diff, (3.0, 3.0))

    assert res["fit_success"] is True
    assert res["sigma_prf_x_arcsec"] > 0.0
    assert res["sigma_prf_y_arcsec"] > 0.0
    assert res["sigma_offset_arcsec"] > 0.0
    assert np.isfinite(res["sigma_offset_arcsec"])
    assert res["offset_significance_sigma"] > 0.0

