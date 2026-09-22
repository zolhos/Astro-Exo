"""
Stellar variability filtering and iterative transit-masked detrending.
"""

from typing import Tuple, Optional
import numpy as np


def flatten_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = "biweight",
    window_length: float = 0.75,
    transit_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust time-window filtering using wotan to flatten stellar rotation
    and instrumental drifts without attenuating transit depth.

    Parameters
    ----------
    time : np.ndarray
        Timestamps in days.
    flux : np.ndarray
        Raw or normalized flux measurements.
    method : str
        Detrending algorithm ('biweight', 'hspline', 'cosine').
    window_length : float
        Filter window length in days (default: 0.75 days ~ 18 hours).
    transit_mask : np.ndarray, optional
        Boolean mask where True indicates in-transit cadences to exclude
        during trend calculation.

    Returns
    -------
    flatten_flux : np.ndarray
        Detrended normalized flux.
    trend : np.ndarray
        Inferred continuum trend.
    """
    try:
        from wotan import flatten

        flatten_flux, trend = flatten(
            time,
            flux,
            method=method,
            window_length=window_length,
            edge_cutoff=0.5,
            mask=transit_mask,
            return_trend=True
        )
        return flatten_flux, trend
    except ImportError:
        # Fallback to Savitzky-Golay / rolling median if wotan is absent
        from scipy.signal import savgol_filter

        win = int(window_length * 48)  # Assuming 30-min cadence approx
        if win % 2 == 0:
            win += 1
        win = max(win, 11)

        trend = savgol_filter(flux, window_length=win, polyorder=2)
        flatten_flux = flux / trend
        return flatten_flux, trend
