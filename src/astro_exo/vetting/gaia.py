"""
Gaia DR3 cross-matching within TESS apertures and WCS pixel overlay.
"""

from typing import Tuple, List, Dict, Any
import numpy as np


def query_gaia_neighbors(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 1.0,
    mag_limit: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Query Gaia DR3 stars within a radius around target coordinates.
    """
    try:
        from astroquery.gaia import Gaia

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        j = Gaia.cone_search_async(coord, radius=radius_arcmin * u.arcmin)
        r = j.get_results()

        neighbors = []
        for row in r:
            phot_g = float(row["phot_g_mean_mag"]) if "phot_g_mean_mag" in row and row["phot_g_mean_mag"] is not None else 99.0
            if phot_g <= mag_limit:
                neighbors.append({
                    "source_id": int(row["source_id"]),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "phot_g_mean_mag": phot_g,
                    "parallax": float(row["parallax"]) if "parallax" in row and row["parallax"] is not None else 0.0,
                    "dist_arcsec": float(row["dist"]) * 3600.0 if "dist" in row else 0.0
                })
        return sorted(neighbors, key=lambda x: x["phot_g_mean_mag"])
    except Exception as e:
        # Fallback or offline placeholder
        return []


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
