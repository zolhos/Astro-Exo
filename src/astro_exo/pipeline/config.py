"""
Configuration dataclasses for pipeline runs and target specifications.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class TargetConfig:
    """Configuration for a specific astrophysical target."""
    tic_id: int
    sector: Optional[int] = None
    period_days: Optional[float] = None
    t0_bjd: Optional[float] = None
    duration_hours: Optional[float] = None
    r_star_rsun: float = 1.0
    m_star_msun: float = 1.0
    author: str = "SPOC"


@dataclass
class PipelineConfig:
    """Global configuration settings for pipeline execution."""
    run_vetting: bool = True
    run_difference_imaging: bool = True
    run_gaia_overlay: bool = True
    run_triceratops: bool = False
    sampler_backend: str = "emcee"  # 'emcee' or 'jax_nuts'
    mcmc_walkers: int = 32
    mcmc_burnin: int = 400
    mcmc_production: int = 1000
    nuts_warmup: int = 500
    nuts_samples: int = 1000
    enable_gp: bool = False
    output_dir: str = "results"
