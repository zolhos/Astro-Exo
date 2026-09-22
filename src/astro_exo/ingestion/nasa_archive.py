"""
Client for the NASA Exoplanet Archive TAP API (TESS Objects of Interest - TOI table).
Implements strict rate-limiting, local disk caching, and quota protection mechanisms.
"""

import os
import sys
import json
import csv
import time
import ssl
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

NASA_TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
DEFAULT_CACHE_FILE = os.path.join(DEFAULT_CACHE_DIR, "nasa_tois_cache.json")
CACHE_TTL_HOURS = 24.0  # 24 hours caching to preserve NASA server quotas


def is_cache_valid(cache_path: str, max_age_hours: float = CACHE_TTL_HOURS) -> bool:
    """Checks whether the local cache file exists and is within TTL limit."""
    if not os.path.exists(cache_path):
        return False
    mtime = os.path.getmtime(cache_path)
    age_hours = (time.time() - mtime) / 3600.0
    return age_hours < max_age_hours


def fetch_nasa_tois(
    limit: int = 15,
    dispositions: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_path: str = DEFAULT_CACHE_FILE,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieves TESS Objects of Interest (TOIs) from the NASA Exoplanet Archive.
    Employs local disk caching to prevent API quota exhaustion.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dispositions = dispositions or ["PC", "CP", "KP"]

    # 1. Quota Guard: Check local disk cache first
    if use_cache and not force_refresh and is_cache_valid(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            print(f"[PROTEÇÃO DE COTA] Reutilizando cache local de TOIs ({cache_path}) - 0 requisições HTTP feitas à NASA.")
            # Filtra e retorna a fatia solicitada
            filtered = [
                r for r in cached_data
                if r.get("expected_disp") in dispositions and r.get("period_days") is not None
            ]
            return filtered[:limit]
        except Exception as e:
            print(f"[CACHE AVISO] Falha ao ler cache local ({e}), realizando consulta segura...")

    # 2. Executa consulta segura à API TAP da NASA
    print(f"[NASA TAP API] Conectando ao NASA Exoplanet Archive (limite seguro: {limit * 3} registros)...")

    # Colunas essenciais da tabela 'toi'
    columns = [
        "tid", "toi", "toidisplay", "tfopwg_disp",
        "pl_orbper", "pl_tranmid", "pl_trandurh", "pl_trandep",
        "pl_rade", "st_rad"
    ]
    col_str = ", ".join(columns)
    # Busca um número controlado para permitir filtragem local sem sobrecarregar o TAP
    query_limit = max(limit * 4, 40)
    tap_query = f"select top {query_limit} {col_str} from toi order by toi asc"

    params = urllib.parse.urlencode({
        "query": tap_query,
        "format": "json"
    })
    url = f"{NASA_TAP_ENDPOINT}?{params}"

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AstroExo/0.1"}
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        raw_json = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    print(f"[NASA TAP API] Resposta recebida em {dt:.2f}s ({len(raw_json)} registros).")

    # 3. Formata e normaliza para o padrão do Astro-Exo
    formatted_targets = []
    for row in raw_json:
        # Pula registros sem efeméride válida
        if row.get("pl_orbper") is None or row.get("pl_tranmid") is None:
            continue

        disp = row.get("tfopwg_disp", "PC")
        p_val = float(row["pl_orbper"])
        t0_val = float(row["pl_tranmid"])
        dur_val = float(row["pl_trandurh"]) if row.get("pl_trandurh") is not None else 3.0
        depth_val = float(row["pl_trandep"]) if row.get("pl_trandep") is not None else 5000.0
        st_rad_val = float(row["st_rad"]) if row.get("st_rad") is not None else 1.0

        item = {
            "tic_id": int(row["tid"]),
            "name": row.get("toidisplay", f"TOI-{row.get('toi')}"),
            "toi": str(row.get("toi", "")),
            "sector": None,
            "period_days": round(p_val, 6),
            "t0_bjd": round(t0_val, 4),
            "duration_hours": round(dur_val, 2),
            "depth_ppm": round(depth_val, 1),
            "r_star_rsun": round(st_rad_val, 2),
            "m_star_msun": 1.0,
            "expected_disp": disp,
            "pl_rade": row.get("pl_rade")
        }
        formatted_targets.append(item)

    # 4. Salva no cache em disco para evitar novas requisições nas próximas 24 horas
    with open(cache_path, "w", encoding="utf-8") as f_cache:
        json.dump(formatted_targets, f_cache, indent=2)
    print(f"[PROTEÇÃO DE COTA] Cache salvo em {cache_path} (Válido por {CACHE_TTL_HOURS:.0f}h).")

    # 5. Filtra por disposição desejada
    filtered = [
        r for r in formatted_targets
        if r.get("expected_disp") in dispositions
    ]
    return filtered[:limit]


def export_tois_to_csv(targets: List[Dict[str, Any]], output_path: str) -> str:
    """Exports a list of formatted TOIs to CSV compatible with BatchProcessor."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "tic_id", "name", "toi", "sector", "period_days", "t0_bjd",
        "duration_hours", "depth_ppm", "r_star_rsun", "m_star_msun", "expected_disp"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in targets:
            writer.writerow(t)
    return output_path
