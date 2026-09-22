"""
Pixel-by-pixel light curve extraction across the TESS target pixel matrix.
"""

from typing import Tuple, Dict, Any
import numpy as np


def extract_pixel_lightcurves(
    tpf_flux: np.ndarray,
    tpf_flux_err: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts normalized light curves for each individual pixel (y, x).
    Shape: (ny, nx, n_cadences)
    """
    # tpf_flux has shape (n_cadences, ny, nx)
    n_cadences, ny, nx = tpf_flux.shape
    pixel_lcs = np.zeros((ny, nx, n_cadences))
    pixel_errs = np.zeros((ny, nx, n_cadences))

    for y in range(ny):
        for x in range(nx):
            pix_series = tpf_flux[:, y, x]
            pix_err = tpf_flux_err[:, y, x]
            med = np.nanmedian(pix_series)
            if med > 0:
                pixel_lcs[y, x, :] = pix_series / med
                pixel_errs[y, x, :] = pix_err / med
            else:
                pixel_lcs[y, x, :] = np.nan
                pixel_errs[y, x, :] = np.nan

    return pixel_lcs, pixel_errs


def locate_transit_pixel(
    time: np.ndarray,
    tpf_flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float
) -> Tuple[int, int, float]:
    """
    Identifies which individual pixel contains the maximum fractional transit dip.

    Returns
    -------
    best_y, best_x : int
        Pixel indices of the primary dip source.
    max_dip_depth : float
        Fractional depth in the most transit-active pixel.
    """
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    in_mask = np.abs(phase) <= (0.5 * duration_days)
    out_mask = (np.abs(phase) >= (0.75 * duration_days)) & (np.abs(phase) <= (1.75 * duration_days))

    n_cadences, ny, nx = tpf_flux.shape
    max_depth = -1.0
    best_y, best_x = 0, 0

    for y in range(ny):
        for x in range(nx):
            in_val = np.nanmedian(tpf_flux[in_mask, y, x])
            out_val = np.nanmedian(tpf_flux[out_mask, y, x])
            if out_val > 0 and in_val > 0:
                dip = (out_val - in_val) / out_val
                if dip > max_depth:
                    max_depth = dip
                    best_y, best_x = y, x

    return best_y, best_x, float(max_depth)
