"""
Data retrieval for TESS and Kepler time-series and Target Pixel Files (TPFs).
"""

from typing import Optional
import numpy as np


def fetch_tess_lightcurve(
    tic_id: int,
    sector: Optional[int] = None,
    author: str = "SPOC"
):
    """
    Fetch calibrated TESS light curve from MAST via lightkurve.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog identifier.
    sector : int, optional
        TESS observing sector.
    author : str
        Data provenance: 'SPOC', 'QLP', 'TGLC'.

    Returns
    -------
    lc : lightkurve.LightCurve
        Cleaned, NaN-removed light curve.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        sector=sector,
        author=author
    )
    if len(search) == 0:
        raise ValueError(f"No light curve found for TIC {tic_id} (sector={sector}, author={author})")

    lc = search.download().remove_nans()
    # Mask low quality cadences (momentum dumps, Earth/Moon stray light)
    if hasattr(lc, "quality"):
        lc = lc[lc.quality == 0]

    return lc


def fetch_tess_tpf(
    tic_id: int,
    sector: Optional[int] = None,
    author: str = "SPOC"
):
    """
    Fetch calibrated TESS Target Pixel File (TPF) from MAST.
    """
    import lightkurve as lk

    search = lk.search_targetpixelfile(
        f"TIC {tic_id}",
        mission="TESS",
        sector=sector,
        author=author
    )
    if len(search) == 0:
        raise ValueError(f"No TPF found for TIC {tic_id} (sector={sector}, author={author})")

    tpf = search.download(quality_bitmask="default")
    tpf = tpf[~np.isnan(tpf.time.value)]
    return tpf
