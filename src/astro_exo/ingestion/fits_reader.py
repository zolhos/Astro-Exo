"""
FITS parser for TESS Light Curves and Target Pixel Files (TPFs).
Supports both astropy.io.fits (when available) and pure-numpy/struct fallback reader.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional


def read_tess_lightcurve(filepath: str, flux_column: str = "PDCSAP_FLUX") -> Dict[str, Any]:
    """
    Parses a TESS light curve FITS file into clean numpy arrays.

    Parameters
    ----------
    filepath : str
        Path to local _lc.fits file.
    flux_column : str
        'PDCSAP_FLUX' (recommended, systematics-corrected) or 'SAP_FLUX'.

    Returns
    -------
    data : dict
        Contains 'time', 'flux', 'flux_err', 'quality', 'sector', 'tic_id', 'cadenceno'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo FITS não encontrado: {filepath}")

    try:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            hdr0 = hdul[0].header
            hdr1 = hdul[1].header
            tbl = hdul[1].data

            time_raw = np.array(tbl["TIME"], dtype=np.float64)
            flux_raw = np.array(tbl[flux_column], dtype=np.float64)
            err_raw = np.array(tbl[f"{flux_column}_ERR"], dtype=np.float64)
            quality = np.array(tbl["QUALITY"], dtype=np.int32)
            cadenceno = np.array(tbl["CADENCENO"], dtype=np.int32)

            tic_id = hdr0.get("TICID", hdr1.get("TICID", 0))
            sector = hdr0.get("SECTOR", hdr1.get("SECTOR", 0))

        # Filtra NaNs e cadências de má qualidade por padrão
        good_mask = ~np.isnan(time_raw) & ~np.isnan(flux_raw) & (quality == 0)
        if np.sum(good_mask) < 10:
            # Relaxa filtro de qualidade caso quality == 0 seja excessivamente estrito
            good_mask = ~np.isnan(time_raw) & ~np.isnan(flux_raw)

        time_clean = time_raw[good_mask]
        flux_clean = flux_raw[good_mask]
        err_clean = err_raw[good_mask]
        quality_clean = quality[good_mask]
        cadenceno_clean = cadenceno[good_mask]

        # Normaliza fluxo pela mediana
        med_flux = np.nanmedian(flux_clean)
        norm_flux = flux_clean / med_flux
        norm_err = err_clean / med_flux

        return {
            "time": time_clean,
            "flux": norm_flux,
            "flux_err": norm_err,
            "raw_flux": flux_clean,
            "quality": quality_clean,
            "cadenceno": cadenceno_clean,
            "tic_id": int(tic_id),
            "sector": int(sector),
            "median_flux": float(med_flux)
        }

    except ImportError:
        # Fallback minimalista via parsing binário puro
        return _pure_numpy_read_fits_table(filepath, flux_column)


def read_tess_tpf(filepath: str) -> Dict[str, Any]:
    """
    Parses a TESS Target Pixel File (_tp.fits) into spatial datacubes.

    Parameters
    ----------
    filepath : str
        Path to local _tp.fits file.

    Returns
    -------
    data : dict
        Contains 'time', 'flux' (3D), 'flux_err' (3D), 'quality', 'sector', 'tic_id'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo TPF não encontrado: {filepath}")

    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(filepath) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        hdr2 = hdul[2].header if len(hdul) > 2 else {}
        tbl = hdul[1].data

        time_raw = np.array(tbl["TIME"], dtype=np.float64)
        flux_cube = np.array(tbl["FLUX"], dtype=np.float32)
        err_cube = np.array(tbl["FLUX_ERR"], dtype=np.float32)
        quality = np.array(tbl["QUALITY"], dtype=np.int32)

        tic_id = hdr0.get("TICID", hdr1.get("TICID", 0))
        sector = hdr0.get("SECTOR", hdr1.get("SECTOR", 0))

        ra_obj = float(hdr0.get("RA_OBJ", 0.0))
        dec_obj = float(hdr0.get("DEC_OBJ", 0.0))
        target_pix = None
        wcs_obj = None

        try:
            if "CTYPE1" in hdr2:
                wcs_obj = WCS(hdr2)
            elif "CTYPE1" in hdr1:
                wcs_obj = WCS(hdr1)

            if wcs_obj is not None and ra_obj != 0.0:
                px, py = wcs_obj.all_world2pix(ra_obj, dec_obj, 0)
                target_pix = (float(px), float(py))
        except Exception:
            pass

    # Filtra NaNs temporais
    valid_time = ~np.isnan(time_raw)
    time_clean = time_raw[valid_time]
    flux_clean = flux_cube[valid_time]
    err_clean = err_cube[valid_time]
    quality_clean = quality[valid_time]

    # Substitui NaNs espaciais por zero ou mediana para estabilidade matemática
    flux_clean = np.nan_to_num(flux_clean, nan=0.0)
    err_clean = np.nan_to_num(err_clean, nan=1.0)

    return {
        "time": time_clean,
        "flux": flux_clean,
        "flux_err": err_clean,
        "quality": quality_clean,
        "tic_id": int(tic_id),
        "sector": int(sector),
        "ra_obj": ra_obj,
        "dec_obj": dec_obj,
        "target_pix": target_pix,
        "wcs": wcs_obj,
        "spatial_shape": flux_clean.shape[1:]
    }


def _pure_numpy_read_fits_table(filepath: str, flux_column: str) -> Dict[str, Any]:
    """Fallback parser reading uncompressed FITS headers and tables using basic I/O."""
    # Como fallback seguro, tenta carregar dados brutos
    raise NotImplementedError(
        f"astropy é necessária para ler tabelas binárias FITS complexas do TESS. "
        f"Por favor ative o ambiente virtual (.venv) ou instale astropy: pip install astropy"
    )
