"""
Output data structures, classes, and verification schemas for exoplanet validation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json


@dataclass
class VettingReport:
    """Detailed spatial and photometric false-positive vetting report."""
    centroid_offset_arcsec: float
    centroid_significance_sigma: float
    target_pixel_x: float
    target_pixel_y: float
    diff_centroid_x: float
    diff_centroid_y: float
    gaia_neighbors_count: int
    neighbors_ruling_out_count: int
    dilution_factor: float = 1.0
    critical_delta_mag: Optional[float] = None
    corrected_rp_rs: Optional[float] = None
    corrected_rp_rs_err: Optional[float] = None
    triceratops_fpp: Optional[float] = None
    triceratops_nfpp: Optional[float] = None
    statistical_validation_passed: bool = False
    scenario_probabilities: Optional[Dict[str, float]] = None
    passed_spatial_vetting: bool = False


@dataclass
class TransitInferenceResult:
    """Posterior distributions and derived physical properties."""
    t0_bjd: float
    t0_err: float
    rp_rs: float
    rp_rs_err: float
    a_rs: float
    a_rs_err: float
    impact_parameter_b: float
    impact_parameter_err: float
    inclination_deg: float
    inclination_err: float
    limb_dark_q1: float
    limb_dark_q2: float
    stellar_density_g_cm3: float
    reduced_chi2: Optional[float] = None
    gelman_rubin_rhat_max: Optional[float] = None


@dataclass
class FullCandidateProduct:
    """Comprehensive candidate evaluation product."""
    tic_id: int
    sector: Optional[int]
    period_days: float
    is_uncataloged: bool
    vetting: VettingReport
    inference: TransitInferenceResult
    disposition: str  # 'VALIDATED_PLANET', 'CANDIDATE', 'FALSE_POSITIVE'

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)
