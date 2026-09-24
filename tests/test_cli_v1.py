"""
Unit and integration tests for Astro-Exo Production CLI v1.0.0.
Verifies command parsing, help generation, version output, joint-rv demo, and dashboard generation.
"""

import os
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from astro_exo import __version__
from astro_exo.cli import main


class TestCLIProductionV1(unittest.TestCase):
    """Test suite verifying all subcommands of astro-exo v1.0.0 CLI."""

    def test_01_version_flag(self):
        """Verifies that --version outputs the correct v1.0.0 string."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit) as cm:
                main(["--version"])
            self.assertEqual(cm.exception.code, 0)
            self.assertIn(__version__, fake_out.getvalue())
            self.assertIn("1.0.0", fake_out.getvalue())

    def test_02_help_generation(self):
        """Verifies top-level help lists all required subcommands."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
            self.assertEqual(cm.exception.code, 0)
            out = fake_out.getvalue()
            for cmd in ["run", "vet", "joint-rv", "batch", "dashboard", "fetch-tois", "smoke"]:
                self.assertIn(cmd, out)

    def test_03_subcommands_help(self):
        """Verifies individual help pages for each subcommand."""
        for subcmd in ["run", "vet", "joint-rv", "batch", "dashboard", "fetch-tois"]:
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with self.assertRaises(SystemExit) as cm:
                    main([subcmd, "--help"])
                self.assertEqual(cm.exception.code, 0)
                self.assertIn(f"usage: astro-exo {subcmd}", fake_out.getvalue())

    def test_04_joint_rv_demo_execution(self):
        """Verifies joint-rv demo execution on benchmark WASP-77b dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main(["joint-rv", "--demo", "--nwalkers", "20", "--nsteps", "100", "--nburn", "50", "--outdir", tmpdir])
            self.assertEqual(ret, 0)
            expected_json = os.path.join(tmpdir, "wasp_77b_rv_fit.json")
            self.assertTrue(os.path.isfile(expected_json))

    def test_05_dashboard_generator(self):
        """Verifies dashboard compilation from results directory into standalone HTML."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        results_dir = os.path.join(repo_root, "results", "rodada_amostras_ineditas")
        if os.path.isdir(results_dir):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_html = os.path.join(tmpdir, "dashboard.html")
                ret = main(["dashboard", "--results-dir", results_dir, "--output", out_html])
                self.assertEqual(ret, 0)
                self.assertTrue(os.path.isfile(out_html))
                with open(out_html, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("Astro-Exo v1.0.0", content)
                    self.assertIn("Catálogo Físico e Status de Vetting", content)


if __name__ == "__main__":
    unittest.main()
