"""
Direct MAST REST / CAOM API client for querying and downloading TESS calibrated photometric products.
Implements local disk caching and streaming download to avoid re-fetching large FITS files.
"""

import os
import sys
import json
import ssl
import time
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, List

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
DEFAULT_PHOTOMETRY_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "photometry")
)


def query_tess_observations(
    tic_id: int,
    sector: Optional[int] = None,
    dataproduct_type: Optional[str] = "timeseries"
) -> List[Dict[str, Any]]:
    """
    Queries MAST CAOM for TESS observations of a target star.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog identifier.
    sector : int, optional
        Specific TESS sector. If None, returns all observed sectors.
    dataproduct_type : str, optional
        'timeseries' for light curves or None for all products.

    Returns
    -------
    observations : list of dict
        Observation metadata records containing obs_id, dataURL, etc.
    """
    filters = [
        {"paramName": "target_name", "values": [str(tic_id)]},
        {"paramName": "obs_collection", "values": ["TESS"]}
    ]
    if dataproduct_type:
        filters.append({"paramName": "dataproduct_type", "values": [dataproduct_type]})
    if sector is not None:
        filters.append({"paramName": "sequence_number", "values": [int(sector)]})

    query = {
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "params": {
            "columns": "*",
            "filters": filters
        }
    }

    ctx = ssl._create_unverified_context()
    data = urllib.parse.urlencode({"request": json.dumps(query)}).encode("utf-8")
    req = urllib.request.Request(
        MAST_INVOKE_URL,
        data=data,
        headers={"User-Agent": "AstroExo/0.1"}
    )

    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    if res.get("status") != "COMPLETE":
        print(f"[MAST AVISO] Status da consulta: {res.get('status')}")
        return []

    return res.get("data", [])


def resolve_tess_download_urls(
    tic_id: int,
    sector: Optional[int] = None,
    author_preference: str = "SPOC"
) -> Dict[str, Any]:
    """
    Finds direct download URLs for both Light Curve (_lc.fits) and Target Pixel File (_tp.fits).
    """
    obs_list = query_tess_observations(tic_id, sector=sector)
    if not obs_list:
        raise FileNotFoundError(f"Nenhuma observação TESS encontrada no MAST para TIC {tic_id} (setor={sector})")

    # Filtra por autor de preferência (ex: SPOC)
    spoc_obs = [o for o in obs_list if o.get("provenance_name") == author_preference]
    # Prioriza cadência padrão de 2 minutos (-s_lc.fits) em relação a fast-cadence
    std_spoc_obs = [o for o in spoc_obs if "-s_lc.fits" in o.get("dataURL", "")]
    
    if std_spoc_obs:
        target_obs = std_spoc_obs[0]
    elif spoc_obs:
        target_obs = spoc_obs[0]
    else:
        target_obs = obs_list[0]

    lc_data_url = target_obs.get("dataURL", "")
    sec_num = target_obs.get("sequence_number")

    # A URL direta para o TPF correspondente segue a nomenclatura SPOC
    # Ex: ..._lc.fits -> ..._tp.fits
    tpf_data_url = lc_data_url.replace("_lc.fits", "_tp.fits")

    lc_http_url = f"{MAST_DOWNLOAD_URL}?uri={lc_data_url}"
    tpf_http_url = f"{MAST_DOWNLOAD_URL}?uri={tpf_data_url}"

    return {
        "tic_id": tic_id,
        "sector": sec_num,
        "author": target_obs.get("provenance_name", "SPOC"),
        "lc_uri": lc_data_url,
        "lc_url": lc_http_url,
        "tpf_uri": tpf_data_url,
        "tpf_url": tpf_http_url,
        "ra": target_obs.get("s_ra"),
        "dec": target_obs.get("s_dec")
    }


def download_fits_file(
    url: str,
    output_path: str,
    use_cache: bool = True
) -> str:
    """
    Downloads a FITS file with local caching and progress feedback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if use_cache and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        print(f"[CACHE FOTOMÉTRICO] Reutilizando arquivo FITS local: {os.path.basename(output_path)} ({os.path.getsize(output_path) / 1024:.1f} KB)")
        return output_path

    print(f"[MAST DOWNLOAD] Baixando FITS de: {url[:70]}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "AstroExo/0.1"})

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        content = resp.read()

    with open(output_path, "wb") as f_out:
        f_out.write(content)

    dt = time.perf_counter() - t0
    size_mb = len(content) / (1024 * 1024)
    print(f"[MAST SUCESSO] {os.path.basename(output_path)} salvo ({size_mb:.2f} MB em {dt:.2f}s - {size_mb/dt:.2f} MB/s)")
    return output_path


def fetch_tess_photometry(
    tic_id: int,
    sector: Optional[int] = None,
    product: str = "lc",
    cache_dir: str = DEFAULT_PHOTOMETRY_CACHE
) -> str:
    """
    Convenience function: Resolves and downloads Light Curve or TPF for a TIC ID.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog ID.
    sector : int, optional
        TESS observing sector.
    product : str
        'lc' for light curve, 'tp' for target pixel file.
    cache_dir : str
        Base cache folder.

    Returns
    -------
    filepath : str
        Path to local calibrated FITS file.
    """
    prod_type = "lc" if product.lower() in ["lc", "lightcurve"] else "tp"
    tic_folder = os.path.join(cache_dir, f"TIC_{tic_id}")

    # 1. Verifica cache local ANTES de qualquer requisição de rede ao MAST
    if os.path.isdir(tic_folder):
        if sector is not None:
            candidate_names = [
                f"tess_tic{tic_id}_s{int(sector):04d}_{prod_type}.fits",
                f"tess_tic{tic_id}_s{int(sector)}_{prod_type}.fits",
                f"tess_tic{tic_id}_{prod_type}.fits"
            ]
            for c_name in candidate_names:
                c_path = os.path.join(tic_folder, c_name)
                if os.path.isfile(c_path) and os.path.getsize(c_path) > 10000:
                    print(f"[CACHE FOTOMÉTRICO] Reutilizando arquivo FITS local: {os.path.basename(c_path)} ({os.path.getsize(c_path) / 1024:.1f} KB)")
                    return c_path
        else:
            matches = [
                f for f in sorted(os.listdir(tic_folder))
                if f.startswith(f"tess_tic{tic_id}") and f.endswith(f"_{prod_type}.fits")
            ]
            for m_name in matches:
                m_path = os.path.join(tic_folder, m_name)
                if os.path.isfile(m_path) and os.path.getsize(m_path) > 10000:
                    print(f"[CACHE FOTOMÉTRICO] Reutilizando arquivo FITS local: {os.path.basename(m_path)} ({os.path.getsize(m_path) / 1024:.1f} KB)")
                    return m_path

    # 2. Se não encontrado em cache local, consulta MAST via API REST e baixa
    resolved = resolve_tess_download_urls(tic_id, sector=sector)
    sec = resolved["sector"]

    os.makedirs(tic_folder, exist_ok=True)

    if prod_type == "lc":
        filename = f"tess_tic{tic_id}_s{sec:04d}_lc.fits" if sec else f"tess_tic{tic_id}_lc.fits"
        dest_path = os.path.join(tic_folder, filename)
        return download_fits_file(resolved["lc_url"], dest_path)
    else:
        filename = f"tess_tic{tic_id}_s{sec:04d}_tp.fits" if sec else f"tess_tic{tic_id}_tp.fits"
        dest_path = os.path.join(tic_folder, filename)
        return download_fits_file(resolved["tpf_url"], dest_path)
