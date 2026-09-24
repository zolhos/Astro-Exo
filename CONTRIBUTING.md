# Contributing to Astro-Exo

Thank you for your interest in contributing to **Astro-Exo**! We welcome contributions from astrophysicists, software engineers, and open-source contributors of all backgrounds.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1). Please review it to understand our community standards.

---

## How Can I Contribute?

- **Reporting Bugs:** Open an issue on GitHub detailing the observed behavior, minimal reproducible code snippet, input data, and system specifications.
- **Suggesting Features:** Propose new physics models, detrending filters, data ingestion sources, or visual analytics tools via GitHub Discussions or Issues.
- **Improving Documentation:** Clarify docstrings, user tutorials, mathematical descriptions in `paper/`, or usage examples.
- **Submitting Code:** Submit pull requests addressing open issues or adding verified capabilities.

---

## Development Workflow & Guidelines

### 1. Environment Setup

Clone the repository and set up a clean Python virtual environment (Python 3.10+ recommended):

```bash
git clone https://github.com/zolhos/Astro-Exo.git
cd Astro-Exo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Code Style and Quality (PEP 8)

We maintain rigorous scientific software standards:
- Adhere strictly to **PEP 8** guidelines.
- Use explicit type annotations (`typing`) across all public functions and class methods.
- Document all classes and functions with **NumPy-style docstrings**, including parameter types, mathematical formulations, and return schemas.
- Keep algorithms modular and numerical routines vectorized using NumPy/SciPy/Astropy whenever possible.

### 3. Running the Test Suite

All contributions must pass the automated test suite before being merged. Ensure that existing tests pass and write new unit tests for any new features or bug fixes.

Run tests using Python's standard `unittest` framework:

```bash
python3 -m unittest discover -s tests
```

To run a specific test module:

```bash
python3 -m unittest tests/test_smoke.py
```

### 4. Git and Pull Request Process

1. **Fork and Branch:** Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit Early and Clearly:** Write clear, descriptive commit messages describing the *why* and *what* of your change:
   ```bash
   git commit -m "feat(vetting): add custom PRF aperture correction"
   ```
3. **Keep PRs Focused:** Avoid bundling unrelated changes or refactors into a single PR.
4. **Update Documentation:** If your change modifies public APIs or adds dependencies, update the relevant documentation, docstrings, and paper metadata if appropriate.
5. **Open Pull Request:** Push to your fork and submit a PR against `main`. Provide a concise summary of changes and reference any associated issue numbers (e.g., `Closes #12`).

---

## Scientific Rigor & Attribution

When implementing transit, astrometric, or radial velocity equations:
- Cite the relevant peer-reviewed astrophysical literature in the module header and docstrings (e.g., Mandel & Agol 2002, Kipping 2013, Giacalone et al. 2021).
- Include mathematical tests validating analytical outputs against benchmark or synthetic analytical cases.

Thank you for helping push the frontier of reproducible exoplanetary astrophysics!
