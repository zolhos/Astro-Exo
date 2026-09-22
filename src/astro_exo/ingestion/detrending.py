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
        win = max(int(window_length * 48), 11)
        if win % 2 == 0:
            win += 1
        try:
            from scipy.signal import savgol_filter
            trend = savgol_filter(flux, window_length=win, polyorder=2)
        except ImportError:
            # Pure numpy moving average fallback
            k = max(int(window_length * 48), 5)
            if k % 2 == 0:
                k += 1
            kernel = np.ones(k) / k
            trend = np.convolve(flux, kernel, mode="same")
            trend[:k//2] = trend[k//2]
            trend[-k//2:] = trend[-k//2 - 1]
        
        flatten_flux = flux / (trend + 1e-12)
        return flatten_flux, trend


def iterative_flatten(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    buffer_factor: float = 1.5,
    method: str = "biweight",
    window_length: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage iterative detrending:
    Stage 1: Generates transit mask based on detected candidate ephemerides.
    Stage 2: Re-computes smooth continuum while excluding in-transit cadences,
             preventing the classic transit-shallowing bias.

    Returns
    -------
    flatten_flux : np.ndarray
        Clean detrended flux.
    trend : np.ndarray
        Continuum baseline.
    transit_mask : np.ndarray
        Boolean array of masked cadences.
    """
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    transit_mask = np.abs(phase) <= (0.5 * duration_days * buffer_factor)

    flatten_flux, trend = flatten_lightcurve(
        time=time,
        flux=flux,
        method=method,
        window_length=window_length,
        transit_mask=transit_mask
    )
    return flatten_flux, trend, transit_mask

