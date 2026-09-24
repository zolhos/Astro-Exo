"""
Astro-Exo: High-Precision Bayesian Inference, Spatial Vetting, and GPU-Accelerated Pipeline
for Exoplanet Discovery and Transit Characterization.
"""

__version__ = "1.0.0"
__author__ = "Diego Zolhos"

from astro_exo.models.transforms import kipping_to_quadratic, quadratic_to_kipping, ecc_omega_to_xy, xy_to_ecc_omega

__all__ = [
    "__version__",
    "kipping_to_quadratic",
    "quadratic_to_kipping",
    "ecc_omega_to_xy",
    "xy_to_ecc_omega"
]
