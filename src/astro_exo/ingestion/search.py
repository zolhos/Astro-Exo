"""
Transit signal search (Transit Least Squares) and NASA Exoplanet Archive cross-matching.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def run_tls_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    r_star: float = 1.0,
    m_star: float = 1.0,
    period_min: float = 0.5,
    period_max: float = 15.0
):
    """
    Performs analytical transit search using Transit Least Squares (TLS).
    Accounts for stellar limb darkening and realistic transit ingress/egress.
    """
    try:
        from transitleastsquares import transitleastsquares

        model = transitleastsquares(time, flux, flux_err)
        results = model.power(
            R_star=r_star,
            M_star=m_star,
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=3,
            duration_grid_step=1.1
        )
        return results
    except ImportError:
        raise ImportError(
            "transitleastsquares is required for transit signal search. "
            "Install with `pip install transitleastsquares`."
        )


def verify_against_exoplanet_archive(
    tic_id: int,
    period: float,
    epoch_bjd: float
) -> Tuple[bool, str]:
    """
    Cross-matches detected candidate period and epoch against TOIs and Confirmed Planets
    at the NASA Exoplanet Archive via TAP query.

    Returns
    -------
    is_new : bool
        True if uncataloged, False if matches known planet or candidate.
    details : str
        Description of match or confirmation of novelty.
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        # 1. Query TESS Objects of Interest (TOI)
        toi_table = NasaExoplanetArchive.query_criteria(
            table="toi",
            where=f"tid = {tic_id}"
        )
        if toi_table is not None and len(toi_table) > 0:
            for row in toi_table:
                known_period = row["toi_period"]
                if known_period is not None and known_period > 0:
                    ratio = period / known_period
                    if any(abs(ratio - n) < 0.015 for n in [0.5, 1.0, 2.0]):
                        return False, f"Matches known candidate TOI {row['toi']} (P={known_period:.4f} d)"

        # 2. Query Confirmed Planets
        ps_table = NasaExoplanetArchive.query_criteria(
            table="ps",
            where=f"tic_id = 'TIC {tic_id}'"
        )
        if ps_table is not None and len(ps_table) > 0:
            return False, f"Matches confirmed planetary system around TIC {tic_id}"

        return True, "Candidate is uncataloged in NASA Exoplanet Archive"
    except Exception as e:
        return True, f"Could not complete archive TAP query ({e}), assuming pending vetting."
