# Astro-Exo v1.0.0 Release Notes 🪐🚀

**Astro-Exo v1.0.0: The Production-Ready Bayesian Exoplanet Validation Ecosystem**

Date: September 24, 2026  
Status: Production / Stable (PEP 566 / PyPI Ready)  
Repository: [https://github.com/zolhos/Astro-Exo](https://github.com/zolhos/Astro-Exo)  
Live Portal: [https://zolhos.github.io/Astro-Exo/](https://zolhos.github.io/Astro-Exo/)

---

## 🌟 Highlights of Version 1.0.0

Astro-Exo reaches its definitive `v1.0.0` milestone, transitioning from research prototyping into a production-grade scientific framework for exoplanet validation, sub-pixel vetting, and multi-instrument radial velocity modeling.

### 1. 28-Target Benchmark Catalog & Complete Diagnostic Suite
- **Full Catalog:** 28 astrophysical targets comprehensively analyzed, combining validated hot Jupiters (WASP-77b, WASP-126b, WASP-62b, WASP-46b), resonant multi-planet sub-Neptune systems (TOI-1027.01, .02, .03), and proven blended eclipsing binaries (TOI-1002.01).
- **168 Publication-Grade Figures:** Each of the 28 targets includes all 6 scientific diagnostic figures:
  1. `transit_fit.png`: Phase-folded transit photometry with Mandel-Agol / Kipping $(q_1, q_2)$ model and residuals.
  2. `corner_mcmc.png`: 7D posterior parameter distributions ($T_0, P, R_p/R_*, a/R_*, b, q_1, q_2$).
  3. `difference_image_centroid.png`: 2D Target Pixel File (TPF) difference flux and sub-pixel PRF offset vector.
  4. `gaia_field_screening.png`: 2.5' Gaia DR3 cone field with critical magnitude bounding threshold $\Delta m_{\text{crit}}$.
  5. `triceratops_probabilities.png`: Posterior probabilities across 6 astrophysical hypotheses (TP, PTP, DTP, EB, PEB, BEB) and False Positive Probability (FPP).
  6. `rv_keplerian_fit.png`: Multi-instrument Keplerian Doppler orbit fit (e.g., HARPS, ESPRESSO, CORALIE) with residuals.

### 2. Interactive Scientific Web Portal (GitHub Pages)
- Deployed at `https://zolhos.github.io/Astro-Exo/` via GitHub Actions (`.github/workflows/deploy-pages.yml`).
- High-performance, responsive dark-mode interface built with Tailwind CSS and Chart.js.
- Real-time catalog filtering by planet status (ALL, CONFIRMED, VALIDATED, REJECTED_FP) and planet regime (Hot Jupiter, Warm Jupiter, Sub-Neptune, Super-Earth, Resonant Sub-Neptune, BEB).
- Dynamic Mass-Radius-Density ($\rho_p$) correlation chart with interactive point selection.
- Full inspection modal for each candidate with 6 tabbed diagnostic figures, physical properties, and direct JSON/CSV export.

### 3. Scientific Methodological Manuscript (LaTeX / JOSS / AAS)
- Complete paper draft in `paper/paper.md` (JOSS format) and `paper/paper.tex` (AAS Journal format).
- Complete mathematical formulations for:
  - Mandel-Agol transit modeling with Kipping (2013) triangular limb darkening.
  - Sub-pixel PRF difference imaging and centroid offset significance.
  - Gaia DR3 critical magnitude thresholding and TRICERATOPS hypothesis evaluation.
  - Multi-instrument Keplerian radial velocity dynamics and interior density classification.
- Canonical BibTeX bibliography in `paper/paper.bib`.

### 4. Ergonomic CLI & Multi-OS CI/CD
- Unified CLI command `astro-exo` with subcommands:
  - `astro-exo run`: Full end-to-end pipeline execution.
  - `astro-exo vet`: Standalone astrometric difference imaging and PRF vetting.
  - `astro-exo joint-rv`: Doppler Keplerian modeling (including `--demo` for WASP-77b).
  - `astro-exo batch`: Multi-target batch pipeline processing with summary reporting.
  - `astro-exo dashboard`: Static HTML5 scientific dashboard generation with `--open`.
  - `astro-exo fetch-tois`: NASA Exoplanet Archive TAP query with quota guard and disk fallback.
  - `astro-exo smoke`: Environment, optional C-extensions, and hardware acceleration diagnostic.
- 37 automated tests passing across Python 3.10, 3.11, and 3.12 on Linux, macOS, and Windows.
- Clean packaging via `pyproject.toml` with PEP 621 metadata and modular extras (`[gpu]`, `[jax]`, `[gp]`, `[vetting]`, `[rv]`, `[dev]`, `[all]`).

---

## 📦 Verification & Quality Metrics

- **Unit & Integration Tests:** 37/37 passing (`Ran 37 tests in 7.9s`)
- **PyPI Build & Twine Check:** Validated wheel (`astro_exo-1.0.0-py3-none-any.whl`) and tarball (`astro_exo-1.0.0.tar.gz`) with 0 errors.
- **FAIR Compliance:** Machine-readable datasets (`consolidated_catalog.json`, `consolidated_catalog.csv`), structured API schemas, and metadata (`CITATION.cff`).

---

## 👥 Authors & Acknowledgments

- **Lead Developer & Principal Investigator:** Diego Zolhos
- **Special Acknowledgments:** NASA Exoplanet Archive, TESS Science Team, Gaia Data Processing and Analysis Consortium (DPAC), and the open-source scientific Python community.
