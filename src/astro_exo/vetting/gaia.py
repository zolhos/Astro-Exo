"""
Gaia DR3 cross-matching within TESS apertures, multispectral conversions, and WCS pixel overlay.
"""

import os
import json
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def estimate_tess_mag_from_gaia(
    phot_g: float,
    bp_rp: Optional[float] = None,
    phot_bp: Optional[float] = None,
    phot_rp: Optional[float] = None
) -> float:
    """
    Estimates the TESS passband magnitude (T_mag) from Gaia DR3 G, BP, and RP magnitudes
    using the empirical polynomial transformation from the TESS Input Catalog (TIC v8; Stassun et al. 2019).

    For -0.5 <= (G_BP - G_RP) <= 4.5:
      T = G - 0.00522555*(BP-RP)^3 + 0.0891337*(BP-RP)^2 - 0.633923*(BP-RP) + 0.0324473

    If color is missing or out-of-range, a robust fallback based on typical dwarf/giant offsets is applied.
    """
    if bp_rp is None and phot_bp is not None and phot_rp is not None:
        bp_rp = float(phot_bp) - float(phot_rp)

    if bp_rp is not None and np.isfinite(bp_rp):
        color = float(np.clip(bp_rp, -0.5, 4.5))
        delta = (
            -0.00522555 * (color ** 3)
            + 0.0891337 * (color ** 2)
            - 0.633923 * color
            + 0.0324473
        )
        tmag = phot_g + delta
    else:
        # Mean offset for solar-type FGK stars in Gaia DR3 / TESS
        tmag = phot_g - 0.430

    return float(round(tmag, 4))


def _get_cache_filename(
    cache_dir: str,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    target_tic: Optional[int] = None
) -> str:
    """Generates standardized cache path for Gaia queries."""
    if target_tic is not None:
        return os.path.join(cache_dir, f"gaia_tic_{target_tic}.json")
    return os.path.join(cache_dir, f"gaia_ra{ra_deg:.4f}_dec{dec_deg:.4f}_r{radius_arcmin:.1f}.json")


def query_gaia_neighbors(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 2.5,
    mag_limit: float = 20.0,
    cache_dir: Optional[str] = "data/gaia_cache",
    target_tic: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Query Gaia DR3 stars within a radius (default 2.5 arcmin = 150 arcsec) around target coordinates.
    Extracts multispectral photometry (G, BP, RP), colors, astrometry (parallax, proper motion),
    computes synthetic TESS magnitudes (T_mag), and caches results locally for offline reproducibility.
    """
    # 1. Check local disk cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _get_cache_filename(cache_dir, ra_deg, dec_deg, radius_arcmin, target_tic)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                return cached_data
            except Exception:
                pass

    # 2. Try online TAP / astroquery query
    neighbors = []
    try:
        from astroquery.gaia import Gaia
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        j = Gaia.cone_search_async(coord, radius=radius_arcmin * u.arcmin)
        r = j.get_results()

        for row in r:
            phot_g = float(row["phot_g_mean_mag"]) if "phot_g_mean_mag" in row and row["phot_g_mean_mag"] is not None else 99.0
            if phot_g <= mag_limit:
                phot_bp = float(row["phot_bp_mean_mag"]) if "phot_bp_mean_mag" in row and row["phot_bp_mean_mag"] is not None else None
                phot_rp = float(row["phot_rp_mean_mag"]) if "phot_rp_mean_mag" in row and row["phot_rp_mean_mag"] is not None else None
                bp_rp = float(row["bp_rp"]) if "bp_rp" in row and row["bp_rp"] is not None else (phot_bp - phot_rp if phot_bp and phot_rp else None)

                tmag = estimate_tess_mag_from_gaia(phot_g, bp_rp, phot_bp, phot_rp)

                star_ra = float(row["ra"])
                star_dec = float(row["dec"])

                # Calculate angular separation if not present
                if "dist" in row and row["dist"] is not None:
                    dist_arcsec = float(row["dist"]) * 3600.0
                else:
                    dist_rad = 2.0 * np.arcsin(np.clip(np.sqrt(
                        np.sin(np.radians(star_dec - dec_deg) / 2.0) ** 2 +
                        np.cos(np.radians(dec_deg)) * np.cos(np.radians(star_dec)) *
                        np.sin(np.radians(star_ra - ra_deg) / 2.0) ** 2
                    ), 0.0, 1.0))
                    dist_arcsec = float(np.degrees(dist_rad) * 3600.0)

                neighbors.append({
                    "source_id": int(row["source_id"]),
                    "ra": star_ra,
                    "dec": star_dec,
                    "dist_arcsec": float(round(dist_arcsec, 3)),
                    "phot_g_mean_mag": phot_g,
                    "phot_bp_mean_mag": phot_bp,
                    "phot_rp_mean_mag": phot_rp,
                    "bp_rp": bp_rp,
                    "tess_mag": tmag,
                    "parallax": float(row["parallax"]) if "parallax" in row and row["parallax"] is not None else 0.0,
                    "parallax_error": float(row["parallax_error"]) if "parallax_error" in row and row["parallax_error"] is not None else 0.0,
                    "pmra": float(row["pmra"]) if "pmra" in row and row["pmra"] is not None else 0.0,
                    "pmdec": float(row["pmdec"]) if "pmdec" in row and row["pmdec"] is not None else 0.0,
                    "ruwe": float(row["ruwe"]) if "ruwe" in row and row["ruwe"] is not None else 1.0,
                    "teff_gspphot": float(row["teff_gspphot"]) if "teff_gspphot" in row and row["teff_gspphot"] is not None else None
                })
        neighbors = sorted(neighbors, key=lambda x: x["dist_arcsec"])
    except Exception as e:
        # Fallback empty list; cache generation scripts or mock catalogs can populate it
        neighbors = []

    # 3. Save to disk cache if populated
    if cache_dir is not None and len(neighbors) > 0:
        cache_file = _get_cache_filename(cache_dir, ra_deg, dec_deg, radius_arcmin, target_tic)
        try:
            with open(cache_file, "w") as f:
                json.dump(neighbors, f, indent=2)
        except Exception:
            pass

    return neighbors


def overlay_gaia_on_wcs(
    neighbors: List[Dict[str, Any]],
    wcs
) -> List[Dict[str, Any]]:
    """
    Projects celestial (ra, dec) coordinates of Gaia neighbors onto pixel (x, y).
    """
    projected = []
    for star in neighbors:
        pix_x, pix_y = wcs.all_world2pix(star["ra"], star["dec"], 0)
        star_copy = dict(star)
        star_copy["pix_x"] = float(pix_x)
        star_copy["pix_y"] = float(pix_y)
        projected.append(star_copy)
    return projected
