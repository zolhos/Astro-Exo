# Astro-Exo 🪐✨

**High-Precision Bayesian Inference, Spatial Vetting, and GPU-Accelerated Pipeline for Exoplanet Discovery and Transit Characterization.**

[![CI](https://github.com/zolhos/Astro-Exo/actions/workflows/ci.yml/badge.svg)](https://github.com/zolhos/Astro-Exo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/astro--ph.EP-arXiv-B31B1B.svg)](https://arxiv.org)

---

## Overview

**Astro-Exo** is an end-to-end astrophysical framework designed to detect, spatially vet, and statistically validate transiting exoplanets observed by NASA space missions (**TESS**, **Kepler**, **K2**). 

Unlike classical 1D transit pipelines that treat photometric points in isolation and underestimate parameter errors, **Astro-Exo** couples **sub-pixel spatial vetting** (Pixel Response Function fitting, difference imaging, and Gaia DR3 blend screening) with **GPU-accelerated Bayesian parameter estimation** (Hamiltonian Monte Carlo / No-U-Turn Sampler via JAX/NumPyro and `celerite2` Gaussian Processes).

```
   Raw TESS Cadences / TPF
             │
             ▼
┌─────────────────────────┐
│ 1. Ingestion & Search   │ ── TLS (Transit Least Squares) + NASA Exoplanet Archive
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. Spatial Vetting      │ ── In/Out Difference Image, Centroid Shift, Gaia DR3 Overlay,
│    (Sub-pixel PRF)      │    Aperture Contamination Bound, TRICERATOPS (FPP & NFPP)
└─────────────────────────┘
             │
             ▼ (Passes Spatial Vetting: Offset < 3σ, FPP < 1%)
┌─────────────────────────┐
│ 3. Bayesian Inference   │ ── JAX / NumPyro (NUTS) + celerite2 GP (SHOTerm)
│    (GPU-Accelerated)    │    Kipping (q1, q2) Limb Darkening, Photo-eccentric ρ* Prior
└─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. Joint RV (Optional)  │ ── Transits + Multi-Spectrograph Doppler (HARPS, ESPRESSO)
│                         │    Derives True Mass (Mp), Bulk Density (ρp), Orbit
└─────────────────────────┘
```

---

## Core Scientific Features

### 1. Observational Photometry Ingestion & Detrending (Phase 2)
* **Direct MAST REST API Client:** Automated query and download of official TESS calibrated light curves (`_lc.fits`) and Target Pixel Files (`_tp.fits`) via STScI CAOM endpoints with local disk caching.
* **High-Performance FITS Parser:** Direct extraction of binary tables (`PDCSAP_FLUX`, `SAP_FLUX`, `QUALITY`), 3D pixel cubes, and automated WCS astrometric calibration from `APERTURE` FITS headers.
* **Stellar Detrending (`wotan`):** Preservation of transit geometries and depth using robust iterative filters (biweight, Huber-spline) with cadence-break tolerance.

### 2. Spatial Vetting & Blend Disambiguation
* **Difference Imaging on Real TPFs:** Subtracts in-transit average from out-of-transit baseline on raw Target Pixel Files with Monte Carlo error propagation.
* **Sub-Pixel WCS Astrometric Centroiding:** Maps difference flux deficits directly to catalog celestial coordinates (RA/Dec $\to$ sub-pixel coordinates), validating whether the deficit is on-target ($< 0.5$ TESS pixels) or originating from nearby blended eclipsing binaries (BEBs).
* **Sub-Pixel PRF Fitting:** Fits a 2D analytical Pixel Response Function to the difference flux to localize transit deficits down to sub-arcsecond precision.
* **Gaia DR3 Cross-Match & Blend Limits:** Queries all stars within 1 arcminute and analytically calculates the maximum possible transit depth each neighbor could inject into the aperture ($\Delta m_{\text{max}}$), mathematically ruling out blended eclipsing binaries (BEBs).
* **TRICERATOPS Integration:** Calculates False Positive Probability (FPP) and Nearby False Positive Probability (NFPP) against TRILEGAL galactic stellar population models.

### 3. High-Dimensional Bayesian Modeling
* **Kipping (2013) Triangular Parameterization:** Samples unconstrained uniform parameters $(q_1, q_2) \in [0, 1]^2$ mapping directly to physically stable, monotonically decreasing limb-darkening profiles ($u_1 + u_2 < 1$, $u_1 > 0$, $u_1 + 2u_2 > 0$).
* **Eccentric Orbit Reparameterization:** Samples $h = \sqrt{e}\cos\omega$ and $k = \sqrt{e}\sin\omega$ with uniform disk priors to avoid boundary biases at $e \to 0$.
* **Empirical Stellar Density Prior:** Integrates Gaia DR3 / spectroscopic density priors ($\rho_*$) into the likelihood to break the classical photo-eccentric degeneracy between orbital eccentricity and transit duration.
* **Correlated Stellar Noise (GPs):** Integrates stochastically driven Simple Harmonic Oscillator (SHO) kernels via `celerite2` to absorb stellar granulation and spot modulation without distorting transit depth.
* **Joint Photometry + Radial Velocity:** Simultaneous inference of light curves and Doppler velocities from multiple spectrographs with separate instrumental zero-points and jitter.

---

## Installation

### Standard Installation (Core + MCMC)
```bash
git clone https://github.com/zolhos/Astro-Exo.git
cd Astro-Exo
pip install -e .
```

### Full Installation (GPU / JAX / GP / Vetting)
```bash
pip install -e ".[all]"
```

---

## Quickstart & Usage

### 1. Command-Line Interface (CLI)

#### Environment & Module Diagnostics (Smoke Test)
```bash
astro-exo smoke
```

#### Ingest Pre-Identified Candidates from NASA Exoplanet Archive (with 24h Quota Guard)
```bash
astro-exo fetch-tois --limit 50 --output data/nasa_candidates.csv
```

#### Run Batch Processing on Multiple Targets
```bash
astro-exo batch --input data/pilot_candidates.csv --mode mock --output-dir results/pilot_batch
```

#### Run Full Pipeline on a Target
```bash
astro-exo run --tic 261136679 --sector 1 --period 3.5225 --t0 1325.5 --duration 2.5 --backend emcee
```

#### Run Spatial Vetting Only
```bash
astro-exo vet --tic 261136679 --sector 1 --period 3.5225 --t0 1325.5 --duration 2.5
```

### 2. Interactive Visual Dashboard

Generate and open a standalone, interactive HTML5 Canvas visual report showing multi-day light curves, phase-folded transits, sub-pixel difference imaging heatmaps, and Doppler radial velocity curves:

```bash
python3 examples/generate_visual_report.py
open examples/sample_dashboard.html
```

### 3. Python API

```python
from astro_exo.pipeline import TargetConfig, PipelineConfig, ExoplanetPipelineRunner
from astro_exo.pipeline.batch import BatchProcessor
from astro_exo.ingestion.nasa_archive import fetch_tois_tap

# 1. Fetch live candidates with local cache protection
tois = fetch_tois_tap(limit=10, dispositions=["PC", "CP"])

# 2. Run batch execution
batch = BatchProcessor(input_path="data/pilot_candidates.csv", mode="mock", output_dir="results/batch")
results = batch.run()
summary = batch.export_summary("results/batch/summary.json")

# 3. Single-target execution
target = TargetConfig(
    tic_id=261136679,
    sector=1,
    period_days=3.5225,
    t0_bjd=1325.5,
    duration_hours=2.5
)

config = PipelineConfig(
    run_vetting=True,
    sampler_backend="emcee",  # or "jax_nuts"
    enable_gp=True,
    output_dir="results"
)

runner = ExoplanetPipelineRunner(target, config)
product = runner.run()

print(f"Disposition: {product.disposition}")
print(f"Planet-to-Star Radius Ratio (Rp/Rs): {product.inference.rp_rs:.4f} +/- {product.inference.rp_rs_err:.4f}")
print(f"Centroid Offset: {product.vetting.centroid_offset_arcsec:.2f} arcsec ({product.vetting.centroid_significance_sigma:.1f} sigma)")
```

---

## Repository Structure

```
Astro-Exo/
├── data/                # Curated benchmark datasets (pilot_candidates.csv, NASA cache)
├── docs/                # Architectural manuals and guides (batch_processing_guide.md)
├── examples/            # Visual reports, sample dashboards (sample_dashboard.html)
├── src/astro_exo/
│   ├── ingestion/       # MAST/TESS downloads, NASA TAP client, TLS search, biweight detrending
│   ├── vetting/         # Difference imaging, Gaia DR3 overlay, dilution, PRF, TRICERATOPS
│   ├── models/          # Transforms (Kipping, e-omega), emcee MCMC, JAX/NumPyro NUTS, celerite2 GP
│   ├── pipeline/        # Runner, BatchProcessor, config dataclasses, output schemas
│   ├── smoke.py         # Diagnostic suite for dependencies and numerical stability
│   └── cli.py           # Unified CLI (run, vet, smoke, batch, fetch-tois)
├── tests/               # Automated unit tests (smoke, quick sample, large multi-transit, batch)
├── pyproject.toml       # Modern Python packaging configuration (PEP 621)
├── CITATION.cff         # Academic citation metadata
└── all_turns.json       # Complete 27-turn analytical specification
```

---

## Running Tests

Execute the automated test suite with `pytest` or directly with standard `python3`:

```bash
# Full test suite via pytest
pytest tests/ -v

# Or run tests directly with python3:
python3 tests/test_smoke.py
python3 tests/test_phase2_real_photometry.py
python3 tests/test_quick_sample.py
python3 tests/test_large_sample.py
python3 tests/test_batch_runner.py
```

---

## Citation

If you use **Astro-Exo** in your academic research, please cite:

```bibtex
@software{zolhos2026astroexo,
  author = {Zolhos, Diego},
  title = {Astro-Exo: High-Precision Bayesian Inference and Spatial Vetting Pipeline for Exoplanet Discovery},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zolhos/Astro-Exo}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
