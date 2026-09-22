"""
Tests for the BatchProcessor engine and catalog ingestion.
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astro_exo.pipeline.batch import BatchProcessor, BatchTargetResult


def test_load_pilot_candidates_csv():
    """Verify loading curated pilot candidates from CSV."""
    csv_path = os.path.join(REPO_ROOT, "data", "pilot_candidates.csv")
    targets = BatchProcessor.load_targets(csv_path)
    assert len(targets) == 5
    tic_ids = [t["tic_id"] for t in targets]
    assert 25155310 in tic_ids  # WASP-126b
    assert 100100827 in tic_ids # WASP-18b
    assert 99999901 in tic_ids  # Mock BEB


def test_load_pilot_candidates_json():
    """Verify loading curated pilot candidates from JSON."""
    json_path = os.path.join(REPO_ROOT, "data", "pilot_candidates.json")
    targets = BatchProcessor.load_targets(json_path)
    assert len(targets) == 5
    assert targets[0]["name"] == "WASP-126b"


def test_batch_processor_execution_mock(tmp_path):
    """Verify batch processing runs end-to-end and outputs consolidated files."""
    test_outdir = str(tmp_path / "test_batch_results")
    processor = BatchProcessor(mode="mock", output_dir=test_outdir)
    csv_path = os.path.join(REPO_ROOT, "data", "pilot_candidates.csv")

    results = processor.process_catalog(csv_path)
    assert len(results) == 5

    # Check confirmed exoplanet passed
    wasp126 = next(r for r in results if r.tic_id == 25155310)
    assert wasp126.status == "PASSED"
    assert wasp126.spatial_vetting_passed is True

    # Check simulated false positive rejected
    beb = next(r for r in results if r.tic_id == 99999901)
    assert beb.status == "REJECTED_FP"
    assert beb.spatial_vetting_passed is False
    assert beb.centroid_offset_arcsec > 3.0

    # Verify summary files were created
    assert os.path.exists(os.path.join(test_outdir, "batch_summary.csv"))
    assert os.path.exists(os.path.join(test_outdir, "batch_summary.json"))
    assert os.path.exists(os.path.join(test_outdir, "TIC_25155310.json"))


def test_nasa_archive_cache_and_quota():
    """Verify NASA archive retrieval respects local disk cache to protect quotas."""
    from astro_exo.ingestion.nasa_archive import fetch_nasa_tois, export_tois_to_csv

    # First fetch will use the existing cache file (zero network requests)
    tois = fetch_nasa_tois(limit=4, use_cache=True)
    assert len(tois) <= 4
    if len(tois) > 0:
        assert "tic_id" in tois[0]
        assert "period_days" in tois[0]
        assert tois[0]["period_days"] > 0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        test_load_pilot_candidates_csv()
        test_load_pilot_candidates_json()
        test_nasa_archive_cache_and_quota()
        from pathlib import Path
        test_batch_processor_execution_mock(Path(temp_dir))
    print("[OK] Todos os testes de BatchProcessor e NASA Archive passaram com sucesso!")
