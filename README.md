# Astro-Exo 🪐✨

**High-Precision Bayesian Inference, Spatial Vetting, and GPU-Accelerated Pipeline for Exoplanet Discovery and Transit Characterization.**

[![CI](https://github.com/zolhos/Astro-Exo/actions/workflows/ci.yml/badge.svg)](https://github.com/zolhos/Astro-Exo/actions/workflows/ci.yml)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-blue.svg)](RELEASE_NOTES.md)
[![Web Portal](https://img.shields.io/badge/Web%20Portal-Live%20Dashboard-purple.svg)](https://zolhos.github.io/Astro-Exo/)
[![Paper](https://img.shields.io/badge/Paper-JOSS%2FAAS-orange.svg)](paper/paper.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 54 passing](https://img.shields.io/badge/tests-54%20passing-brightgreen.svg)](tests/)

---

## 🌐 Interactive Scientific Web Portal

Explore our live interactive exoplanet catalog with all **50 real benchmark exoplanets** and **300 diagnostic figures** at:  
👉 **[https://zolhos.github.io/Astro-Exo/](https://zolhos.github.io/Astro-Exo/)**

* **Filterable Catalog:** Instant filtering across 50 real exoplanet systems (12 Hot Jupiters + 38 Sub-Neptunes and Super-Earths) by planetary regime and validation status.
* **Full Diagnostic Inspector:** Phase-folded transits with Mandel-Agol & Keplerian $a/R_*$, 7D MCMC corner plots, 2D PRF difference images, Gaia DR3 cone screening fields, TRICERATOPS hypothesis distributions, and authentic Keplerian Doppler RV curves (HARPS-N, HARPS, CORALIE, SOPHIE).
* **Interactive Mass-Radius-Density ($\rho_p$) Diagram:** Real-time correlation chart with clickable candidates mapped against theoretical EOS tracks (Zeng et al. 2016).

---

## Overview

**Astro-Exo** is an end-to-end astrophysical framework designed to detect, spatially vet, and statistically validate transiting exoplanets observed by NASA space missions (**TESS**, **Kepler**, **K2**). 

Unlike classical 1D transit pipelines that treat photometric points in isolation and underestimate parameter errors, **Astro-Exo** couples **sub-pixel spatial vetting** (Pixel Response Function fitting, difference imaging, and Gaia DR3 blend screening) with **GPU-accelerated Bayesian parameter estimation** (Hamiltonian Monte Carlo / No-U-Turn Sampler via JAX/NumPyro and `celerite2` Gaussian Processes) and **joint Doppler radial velocity (RV) Keplerian dynamics** to derive absolute planetary masses and bulk densities.

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
             │
             ▼
┌─────────────────────────┐
│ 5. Scientific Delivery  │ ── Interactive Web Portal, Standardized Catalogs (CSV/JSON),
│    (Phase 6 / Final)    │    High-Resolution Diagnostic Figures, Paper Draft
└─────────────────────────┘
```

---

## Core Scientific Features

### 1. Observational Photometry Ingestion & Detrending (Phase 2)
* **Direct MAST REST API Client:** Automated query and download of official TESS calibrated light curves (`_lc.fits`) and Target Pixel Files (`_tp.fits`) via STScI CAOM endpoints with local disk caching and offline failover.
* **High-Performance FITS Parser:** Direct extraction of binary tables (`PDCSAP_FLUX`, `SAP_FLUX`, `QUALITY`), 3D pixel cubes, and automated WCS astrometric calibration from `APERTURE` FITS headers.
* **Stellar Detrending (`wotan`):** Preservation of transit geometries and depth using robust iterative filters (biweight, Huber-spline) with cadence-break tolerance.

### 2. Spatial Vetting, Gaia DR3 Dilution & Statistical Validation (Phase 4)
* **Difference Imaging on Real TPFs:** Subtracts in-transit average from out-of-transit baseline on raw Target Pixel Files with Monte Carlo error propagation.
* **Sub-Pixel WCS Astrometric Centroiding:** Maps difference flux deficits directly to catalog celestial coordinates (RA/Dec $\to$ sub-pixel coordinates), validating whether the deficit is on-target ($< 0.5$ TESS pixels) or originating from nearby blended eclipsing binaries (BEBs).
* **Sub-Pixel PRF Fitting:** Fits a 2D analytical Pixel Response Function to the difference flux to localize transit deficits down to sub-arcsecond precision.
* **Gaia DR3 2.5' Multispectral Screening:** Queries all stars within 2.5 arcmin, derives synthetic TESS magnitudes ($T_{\text{mag}}$) from Gaia $G, BP, RP$ colors (TIC v8), and analytically computes critical magnitude bounds ($\Delta m_{\text{crit}} = -2.5 \log_{10} \delta_{\text{obs}}$) ruling out background blends.
* **Analytical De-dilution (Transit Restoration):** Reverses transit depth attenuation in close binary blends (e.g. WASP-77b at 3.3"), restoring true physical transit depth $\delta_{\text{true}} = \delta_{\text{obs}} / D$ and true radius ratio $(R_p/R_\star)_{\text{true}}$.
* **TRICERATOPS Statistical Validation:** Computes marginalized posterior probabilities across 6 astrophysical hypotheses (TP, PTP, EB, EBx2P, HEB, BEB), deriving False Positive Probability (FPP < 1%) and Nearby False Positive Probability (NFPP < 0.1%).

### 3. High-Dimensional Bayesian Modeling (Phase 3)
* **Kipping (2013) Triangular Parameterization:** Samples unconstrained uniform parameters $(q_1, q_2) \in [0, 1]^2$ mapping directly to physically stable, monotonically decreasing limb-darkening profiles ($u_1 + u_2 < 1$, $u_1 > 0$, $u_1 + 2u_2 > 0$).
* **Eccentric Orbit Reparameterization:** Samples $h = \sqrt{e}\cos\omega$ and $k = \sqrt{e}\sin\omega$ with uniform disk priors to avoid boundary biases at $e \to 0$.
* **Empirical Stellar Density Prior:** Integrates Gaia DR3 / spectroscopic density priors ($\rho_*$) into the likelihood to break the classical photo-eccentric degeneracy between orbital eccentricity and transit duration.
* **Correlated Stellar Noise (GPs):** Integrates stochastically driven Simple Harmonic Oscillator (SHO) kernels via `celerite2` (with exact dense-matrix jitter fallback) to absorb stellar granulation and spot modulation without distorting transit depth.

### 4. Multi-Instrument Joint Keplerian Dynamics & Interior Classification (Phase 5)
* **Multi-Spectrograph Ingestion & Vectorization:** Simultaneous ingestion of Doppler time-series from ground-based spectrographs (**HARPS**, **CORALIE**, **ESPRESSO**, **HIRES**), partitioning independent systemic zero-point velocities ($\gamma_k$) and instrumental jitters ($\sigma_{\text{jit}, k}$).
* **Analytical Kepler Solver:** High-precision Newton-Raphson Keplerian solver converging in $< 10^{-12}$ precision across circular and eccentric orbits.
* **True Planetary Mass ($M_p$) & Bulk Density ($\rho_p$):** Couples transit inclination $i$ with Doppler semi-amplitude $K$ to eliminate the $\sin i$ degeneracy, measuring true mass, physical radius, bulk density, surface gravity ($\log g_p$), and escape velocity ($v_{\text{esc}}$).
* **Internal Structure Classification:** Automatically maps characterized exoplanets onto theoretical equation of state (EOS) composition tracks (Iron core, Earth-like rocky, Water worlds, Sub-Neptunes, Hot Jupiters, Inflated giants) based on high-pressure EOS models (Zeng et al.).

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

#### Run Multi-Spectrograph Joint Doppler Radial Velocity (RV)
```bash
# Run benchmark demo on WASP-77b with multi-instrument data (HARPS + CORALIE)
astro-exo joint-rv --demo

# Or run on custom dataset
astro-exo joint-rv --target TOI-1001.01 --rv-file data/rv_data/TOI-1001.01_rv.csv --period 1.9316 --t0 2458500.0 --rp-rs 0.105
```

#### Generate Scientific Interactive Web Dashboard
```bash
# Generate dashboard from novel round results
astro-exo dashboard --results-dir results/rodada_amostras_ineditas --output results/dashboard_astro_exo.html --open
```

### 2. Full Test Round with Novel Samples (Complete Pipeline Verification)

Execute all modules across novel, previously unseen targets and output all diagnostic plots, catalogs, and reports:

```bash
python3 examples/executar_rodada_amostras_ineditas.py
```
This produces the complete suite of 17 individual plots and comprehensive reports in `results/rodada_amostras_ineditas/`.

### 3. Interactive Visual Dashboard

Generate and open a standalone, interactive HTML5 Canvas visual report:

```bash
python3 examples/generate_visual_report.py
open examples/sample_dashboard.html
```

### 4. Python API

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
├── data/                                      # Curated benchmark datasets & RV archives
│   ├── rv_data/                               # Multi-spectrograph radial velocity datasets
│   ├── pilot_candidates.csv                   # Historical validation batch
│   └── nasa_tois_cache.json                   # Offline cache fallback for NASA TAP
├── docs/                                      # Architectural manuals, handouts & scientific reports
│   ├── handout_sessao_fase6.md                # Latest session handout & Phase 6 roadmap
│   ├── handout_sessao_fase5.md                # Joint RV modeling session handout
│   ├── relatorio_cientifico_fase5.md          # Multi-spectrograph physical properties report
│   └── batch_processing_guide.md             # Batch pipeline operational guide
├── examples/                                  # Executable scripts and visual demonstrations
│   ├── executar_rodada_amostras_ineditas.py   # Complete novel-sample execution engine
│   ├── run_phase5_joint_modeling.py           # Joint Transit + RV modeling runner
│   └── generate_visual_report.py              # HTML5 interactive visual dashboard
├── results/                                   # Analysis products & diagnostic figures
│   └── rodada_amostras_ineditas/              # Complete novel-sample validation bundle
│       ├── catalogo_amostras_ineditas.csv     # Final characterization catalog
│       ├── resumo_rodada.json                 # Machine-readable output summary
│       ├── relatorio_completo_rodada_inedita.md # Scientific analytical report
│       ├── painel_geral_rodada.png            # Multi-panel diagnostic overview
│       ├── mass_radius_density_diagram.png    # Mass-Radius-Density EOS diagram
│       └── TIC_*/                             # Individual transit, MCMC, PRF, Gaia & RV plots
├── src/astro_exo/                             # Core Python package
│   ├── ingestion/                             # MAST/TESS client, RV loader, NASA archive TAP
│   ├── vetting/                               # Difference imaging, PRF 2D, Gaia DR3, TRICERATOPS
│   ├── models/                                # Kipping/e-omega transforms, emcee, NUTS, celerite2, joint RV
│   ├── pipeline/                              # Pipeline runner, BatchProcessor, configs, schemas
│   ├── smoke.py                               # Diagnostic environment verifier
│   └── cli.py                                 # Unified CLI entrypoint
├── tests/                                     # Automated test suite (32 tests passing)
│   ├── test_all_modules_novel_samples.py      # Complete novel-sample validation test
│   ├── test_phase5_joint_rv.py                # Keplerian solver & RV likelihood tests
│   ├── test_phase4_vetting.py                 # Difference imaging & Gaia cone tests
│   ├── test_phase3_bayesian.py                # MCMC/NUTS sampler tests
│   ├── test_phase2_real_photometry.py         # FITS & detrending tests
│   └── test_smoke.py                          # Fast sanity check
├── pyproject.toml                             # Modern packaging configuration (PEP 621)
├── CITATION.cff                               # Academic citation metadata
└── LICENSE                                    # MIT License
```

---

## Running Tests

Execute the automated test suite with standard `unittest` or `pytest`:

```bash
# Run all 37 automated tests
python3 -m unittest discover tests

# Or run tests individually:
python3 tests/test_cli_v1.py
python3 tests/test_all_modules_novel_samples.py
python3 tests/test_phase5_joint_rv.py
python3 tests/test_phase4_vetting.py
python3 tests/test_phase3_bayesian.py
python3 tests/test_phase2_real_photometry.py
python3 tests/test_smoke.py
```

---

## Citation

If you use **Astro-Exo** in your academic research, please cite:

```bibtex
@software{zolhos2026astroexo,
  author = {Zolhos, Diego},
  title = {Astro-Exo: High-Precision Bayesian Inference, Spatial Vetting, and Radial Velocity Modeling Pipeline for Exoplanet Discovery},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zolhos/Astro-Exo}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
