"""
Pixel Response Function (PRF) sub-pixel fitting on TESS difference images.
"""

from typing import Tuple, Dict, Any
import numpy as np


def gaussian_2d_prf(
    xy: Tuple[np.ndarray, np.ndarray],
    x0: float,
    y0: float,
    amplitude: float,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    theta_rad: float = 0.0,
    background: float = 0.0
) -> np.ndarray:
    """
    2D elliptical Gaussian model serving as an analytical approximation to the TESS PRF.
    """
    x, y = xy
    a = (np.cos(theta_rad)**2) / (2 * sigma_x**2) + (np.sin(theta_rad)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta_rad)) / (4 * sigma_x**2) + (np.sin(2 * theta_rad)) / (4 * sigma_y**2)
    c = (np.sin(theta_rad)**2) / (2 * sigma_x**2) + (np.cos(theta_rad)**2) / (2 * sigma_y**2)

    g = amplitude * np.exp(-(a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2))) + background
    return g


def fit_tess_prf_subpixel(
    i_diff: np.ndarray,
    sigma_diff: np.ndarray,
    target_catalog_xy: Tuple[float, float],
    tess_pixel_scale_arcsec: float = 21.0
) -> Dict[str, float]:
    """
    Fits the PRF to the difference image using non-linear least squares to obtain
    sub-pixel (x0, y0) position of the transit deficit source.
    """
    ny, nx = i_diff.shape
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    xy_grid = (x_coords, y_coords)

    # Initial estimates
    x_init, y_init = target_catalog_xy
    amp_init = np.nanmax(i_diff)
    p0 = [x_init, y_init, amp_init, 1.2, 1.2, 0.0]

    def loss(p):
        x0, y0, amp, sx, sy, bg = p
        if amp < 0 or sx < 0.2 or sy < 0.2:
            return 1e10
        model = gaussian_2d_prf(xy_grid, x0, y0, amp, sx, sy, 0.0, bg)
        chi2 = np.nansum(((i_diff - model) / (sigma_diff + 1e-12)) ** 2)
        return chi2

    res = minimize(loss, p0, method="Nelder-Mead")
    x_fit, y_fit, amp_fit, sx_fit, sy_fit, bg_fit = res.x

    # Compute numerical curvature (second derivatives) along x and y to get formal uncertainties
    eps = 1e-4
    f0 = loss(res.x)
    p_x_plus = res.x.copy(); p_x_plus[0] += eps
    p_x_minus = res.x.copy(); p_x_minus[0] -= eps
    d2_x = max((loss(p_x_plus) - 2.0 * f0 + loss(p_x_minus)) / (eps ** 2), 1e-6)
    sigma_x_pix = 1.0 / np.sqrt(0.5 * d2_x)

    p_y_plus = res.x.copy(); p_y_plus[1] += eps
    p_y_minus = res.x.copy(); p_y_minus[1] -= eps
    d2_y = max((loss(p_y_plus) - 2.0 * f0 + loss(p_y_minus)) / (eps ** 2), 1e-6)
    sigma_y_pix = 1.0 / np.sqrt(0.5 * d2_y)

    offset_pix = np.hypot(x_fit - x_init, y_fit - y_init)
    offset_arcsec = float(offset_pix * tess_pixel_scale_arcsec)
    sigma_offset_arcsec = float(np.hypot(sigma_x_pix, sigma_y_pix) * tess_pixel_scale_arcsec)
    significance = float(offset_arcsec / (sigma_offset_arcsec + 1e-9))

    return {
        "prf_x": float(x_fit),
        "prf_y": float(y_fit),
        "sigma_prf_x_arcsec": float(sigma_x_pix * tess_pixel_scale_arcsec),
        "sigma_prf_y_arcsec": float(sigma_y_pix * tess_pixel_scale_arcsec),
        "prf_amplitude": float(amp_fit),
        "offset_arcsec": offset_arcsec,
        "sigma_offset_arcsec": sigma_offset_arcsec,
        "offset_significance_sigma": significance,
        "fit_success": bool(res.success)
    }

