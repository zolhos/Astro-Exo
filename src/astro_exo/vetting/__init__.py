"""
High-resolution spatial vetting suite: Difference Imaging, Gaia DR3 overlay,
contamination calculation, pixel-level extraction, PRF fitting, and TRICERATOPS validation.
"""

from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.gaia import query_gaia_neighbors, overlay_gaia_on_wcs, estimate_tess_mag_from_gaia
from astro_exo.vetting.dilution import (
    calculate_max_transit_depth,
    calculate_critical_delta_mag,
    calculate_dilution_factor,
    correct_diluted_transit_depth,
    restore_true_radius_ratio,
    correct_binary_blend_wasp77,
    rule_out_neighbors_as_blends
)
from astro_exo.vetting.pixel_lc import extract_pixel_lightcurves, locate_transit_pixel
from astro_exo.vetting.prf_fit import fit_tess_prf_subpixel
from astro_exo.vetting.triceratops_vet import run_triceratops_validation, BayesianFalsePositiveEngine

__all__ = [
    "calculate_difference_image",
    "measure_centroid_offset",
    "query_gaia_neighbors",
    "overlay_gaia_on_wcs",
    "estimate_tess_mag_from_gaia",
    "calculate_max_transit_depth",
    "calculate_critical_delta_mag",
    "calculate_dilution_factor",
    "correct_diluted_transit_depth",
    "restore_true_radius_ratio",
    "correct_binary_blend_wasp77",
    "rule_out_neighbors_as_blends",
    "extract_pixel_lightcurves",
    "locate_transit_pixel",
    "fit_tess_prf_subpixel",
    "run_triceratops_validation",
    "BayesianFalsePositiveEngine"
]
