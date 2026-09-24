"""
Light curve ingestion and target pixel file retrieval.
"""

from astro_exo.ingestion.mast_tess import fetch_tess_lightcurve, fetch_tess_tpf
from astro_exo.ingestion.detrending import flatten_lightcurve
from astro_exo.ingestion.search import run_tls_search, verify_against_exoplanet_archive
from astro_exo.ingestion.rv_loader import RVDataset, load_rv_csv, simulate_multi_instrument_rv

__all__ = [
    "fetch_tess_lightcurve",
    "fetch_tess_tpf",
    "flatten_lightcurve",
    "run_tls_search",
    "verify_against_exoplanet_archive",
    "RVDataset",
    "load_rv_csv",
    "simulate_multi_instrument_rv",
]
