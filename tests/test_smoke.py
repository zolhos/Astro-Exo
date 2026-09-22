"""
Pytest-compatible wrapper and entrypoint for the Astro-Exo smoke test suite.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from astro_exo.smoke import (
    test_smoke_transforms,
    test_smoke_difference_imaging,
    test_smoke_dilution_screening,
    test_smoke_keplerian_rv,
    test_smoke_pipeline_schemas,
    run_full_diagnostics_and_smoke
)


def test_transforms_smoke():
    ok, msg = test_smoke_transforms()
    assert ok, msg


def test_difference_imaging_smoke():
    ok, msg = test_smoke_difference_imaging()
    assert ok, msg


def test_dilution_screening_smoke():
    ok, msg = test_smoke_dilution_screening()
    assert ok, msg


def test_keplerian_rv_smoke():
    ok, msg = test_smoke_keplerian_rv()
    assert ok, msg


def test_pipeline_schemas_smoke():
    ok, msg = test_smoke_pipeline_schemas()
    assert ok, msg


if __name__ == "__main__":
    sys.exit(run_full_diagnostics_and_smoke())
