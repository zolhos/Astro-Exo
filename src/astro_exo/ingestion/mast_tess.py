"""
Data retrieval for TESS and Kepler time-series and Target Pixel Files (TPFs).
Leverages direct cached MAST downloads and seamless fits_reader integration.
"""

from typing import Optional, Dict, Any
import numpy as np

from astro_exo.ingestion.mast_client import fetch_tess_photometry
from astro_exo.ingestion.fits_reader import read_tess_lightcurve, read_tess_tpf


def fetch_tess_lightcurve(
    tic_id: int,
    sector: Optional[int] = None,
    author: str = "SPOC",
    as_lightkurve: bool = False
):
    """
    Fetch calibrated TESS light curve from MAST with automated disk caching.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog identifier.
    sector : int, optional
        TESS observing sector.
    author : str
        Data provenance: 'SPOC', 'QLP', 'TGLC'.
    as_lightkurve : bool
        If True, returns a lightkurve.LightCurve object (requires lightkurve).
        If False, returns a clean dict with 'time', 'flux', 'flux_err', 'quality'.

    Returns
    -------
    data : dict or lightkurve.LightCurve
        Calibrated, cleaned, NaN-removed photometric time series.
    """
    try:
        fits_file = fetch_tess_photometry(tic_id, sector=sector, product="lc")
        if as_lightkurve:
            import lightkurve as lk
            return lk.read(fits_file)
        return read_tess_lightcurve(fits_file)
    except Exception as e:
        # Fallback para busca direta via lightkurve se a busca CAOM falhar
        try:
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
            if hasattr(lc, "quality"):
                lc = lc[lc.quality == 0]
            if as_lightkurve:
                return lc
            return {
                "time": lc.time.value,
                "flux": lc.flux.value / np.nanmedian(lc.flux.value),
                "flux_err": lc.flux_err.value / np.nanmedian(lc.flux.value),
                "quality": lc.quality.value if hasattr(lc, "quality") else np.zeros(len(lc)),
                "tic_id": tic_id,
                "sector": sector or 0
            }
        except ImportError:
            raise RuntimeError(f"Falha ao obter curva de luz para TIC {tic_id}: {e}")


def fetch_tess_tpf(
    tic_id: int,
    sector: Optional[int] = None,
    author: str = "SPOC",
    as_lightkurve: bool = False
):
    """
    Fetch calibrated TESS Target Pixel File (TPF) from MAST with disk caching.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog identifier.
    sector : int, optional
        TESS observing sector.
    author : str
        Data provenance: 'SPOC', 'QLP'.
    as_lightkurve : bool
        If True, returns lightkurve.TargetPixelFile.
        If False, returns dict with 'time', 'flux' (3D), 'flux_err' (3D).

    Returns
    -------
    tpf : dict or lightkurve.TargetPixelFile
    """
    try:
        fits_file = fetch_tess_photometry(tic_id, sector=sector, product="tp")
        if as_lightkurve:
            import lightkurve as lk
            return lk.read(fits_file)
        return read_tess_tpf(fits_file)
    except Exception as e:
        try:
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
            if as_lightkurve:
                return tpf
            return {
                "time": tpf.time.value,
                "flux": tpf.flux.value,
                "flux_err": tpf.flux_err.value,
                "tic_id": tic_id,
                "sector": sector or 0,
                "spatial_shape": tpf.flux.shape[1:],
                "ra": getattr(tpf, "ra", 0.0),
                "dec": getattr(tpf, "dec", 0.0),
                "wcs": getattr(tpf, "wcs", None)
            }
        except ImportError:
            raise RuntimeError(f"Falha ao obter TPF para TIC {tic_id}: {e}")
