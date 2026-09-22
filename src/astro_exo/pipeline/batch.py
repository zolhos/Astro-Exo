"""
Batch processing engine for Astro-Exo pipeline.
Executes automated spatial vetting and inference across candidate catalogs (CSV or JSON).
Includes fault-tolerant execution, live/mock execution modes, and consolidated reporting.
"""

import os
import sys
import csv
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from astro_exo.pipeline.config import TargetConfig, PipelineConfig
from astro_exo.pipeline.schemas import VettingReport
from astro_exo.models.transforms import compute_stellar_density, impact_param_to_inclination, kipping_to_quadratic
from astro_exo.models.joint_rv import compute_planetary_mass_density
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.dilution import calculate_max_transit_depth, rule_out_neighbors_as_blends


@dataclass
class BatchTargetResult:
    """Summary metrics and vetting outcome for an individual target in a batch."""
    tic_id: int
    name: str
    toi: str
    sector: Optional[int]
    period_days: float
    t0_bjd: float
    duration_hours: float
    status: str  # 'PASSED', 'REJECTED_FP', 'FAILED'
    spatial_vetting_passed: bool
    centroid_offset_arcsec: float
    centroid_offset_sigma: float
    gaia_neighbors_screened: int
    neighbors_ruled_out: int
    depth_ppm: float
    derived_density_g_cm3: Optional[float]
    elapsed_sec: float
    error_message: Optional[str] = None


class BatchProcessor:
    """Orchestrates batch execution of exoplanet candidate catalogs."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        mode: str = "mock",
        output_dir: str = "results/batch_run"
    ):
        self.config = config or PipelineConfig(output_dir=output_dir)
        self.mode = mode.lower()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_targets(filepath: str) -> List[Dict[str, Any]]:
        """Parses targets from a CSV or JSON candidate catalog."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de candidatos não encontrado: {filepath}")

        targets = []
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    targets.append(item)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    targets.append({
                        "tic_id": int(row["tic_id"]),
                        "name": row.get("name", f"TIC-{row['tic_id']}"),
                        "toi": row.get("toi", "N/A"),
                        "sector": int(row["sector"]) if row.get("sector") else None,
                        "period_days": float(row.get("period_days", 3.5)),
                        "t0_bjd": float(row.get("t0_bjd", 1325.0)),
                        "duration_hours": float(row.get("duration_hours", 3.0)),
                        "depth_ppm": float(row.get("depth_ppm", 5000.0)),
                        "r_star_rsun": float(row.get("r_star_rsun", 1.0)),
                        "m_star_msun": float(row.get("m_star_msun", 1.0)),
                        "author": row.get("author", "SPOC"),
                        "expected_disp": row.get("expected_disp", "PC")
                    })
        return targets

    def _process_single_mock(self, target_dict: Dict[str, Any]) -> BatchTargetResult:
        """Processes a target using controlled astrophysical simulation."""
        t_start = time.perf_counter()
        tic_id = target_dict["tic_id"]
        name = target_dict.get("name", str(tic_id))
        toi = target_dict.get("toi", "N/A")
        period = target_dict["period_days"]
        t0 = target_dict["t0_bjd"]
        dur_h = target_dict["duration_hours"]
        dur_d = dur_h / 24.0
        depth_ppm = target_dict.get("depth_ppm", 5000.0)
        depth_frac = depth_ppm / 1e6

        # Cenário especial: Alvo marcado como Falso Positivo (BEB)
        is_simulated_fp = (target_dict.get("expected_disp") == "FP") or ("BEB" in name)

        # 1. Simula TPF 7x7 com janela dinâmica adaptada à duração do trânsito
        n_cadences = 100
        ny, nx = 7, 7
        half_window = max(0.25, 2.0 * dur_d)
        time_pts = np.linspace(-half_window, half_window, n_cadences) + t0
        flux_tpf = np.full((n_cadences, ny, nx), 2000.0)
        err_tpf = np.full_like(flux_tpf, 3.0)

        phase = (time_pts - t0 + 0.5 * period) % period - 0.5 * period
        in_transit = np.abs(phase) <= (0.5 * dur_d)

        target_cat_pos = (3.0, 3.0)

        if is_simulated_fp:
            # Em um BEB, o eclipse ocorre em um pixel deslocado (ex: vizinho contaminante em 1, 5)
            flux_tpf[in_transit, 1, 5] -= 400.0
            flux_tpf[:, 3, 3] += 5000.0
        else:
            # Em um planeta real, o eclipse ocorre exatamente no pixel central do alvo
            flux_tpf[in_transit, 3, 3] -= 2000.0 * depth_frac * 5.0
            flux_tpf[:, 3, 3] += 5000.0

        i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
            time_pts, flux_tpf, err_tpf, period, t0, dur_d
        )

        cen_res = measure_centroid_offset(
            i_diff, sigma_diff, target_cat_pos, tess_pixel_scale_arcsec=21.0, n_mc_perturbations=300
        )

        offset_arcsec = cen_res["offset_arcsec"]
        offset_sigma = cen_res["offset_significance_sigma"]
        passed_vetting = (offset_sigma < 3.0) and (offset_arcsec < 3.0)

        # 2. Triagem analítica de vizinhos Gaia
        mock_neighbors = [
            {"source_id": tic_id * 10 + 1, "phot_g_mean_mag": 15.5},
            {"source_id": tic_id * 10 + 2, "phot_g_mean_mag": 16.8},
            {"source_id": tic_id * 10 + 3, "phot_g_mean_mag": 17.5},
        ]
        vetted_nb = rule_out_neighbors_as_blends(
            observed_transit_depth_ppm=depth_ppm,
            target_mag=10.0,
            neighbors=mock_neighbors
        )
        ruled_out = sum(1 for n in vetted_nb if n["ruled_out"])

        # 3. Propriedades Físicas
        a_rs = 10.0
        rho_star = compute_stellar_density(period, a_rs)

        planet_phys = compute_planetary_mass_density(
            m_star_msun=target_dict.get("m_star_msun", 1.0),
            r_star_rsun=target_dict.get("r_star_rsun", 1.0),
            period_days=period,
            k_semiamp_ms=60.0,
            rp_rs=np.sqrt(depth_frac)
        )

        elapsed = time.perf_counter() - t_start
        status = "PASSED" if passed_vetting else "REJECTED_FP"

        return BatchTargetResult(
            tic_id=tic_id,
            name=name,
            toi=toi,
            sector=target_dict.get("sector"),
            period_days=period,
            t0_bjd=t0,
            duration_hours=dur_h,
            status=status,
            spatial_vetting_passed=passed_vetting,
            centroid_offset_arcsec=round(offset_arcsec, 2),
            centroid_offset_sigma=round(offset_sigma, 2),
            gaia_neighbors_screened=len(mock_neighbors),
            neighbors_ruled_out=ruled_out,
            depth_ppm=round(depth_ppm, 1),
            derived_density_g_cm3=round(planet_phys["density_g_cm3"], 2) if passed_vetting else None,
            elapsed_sec=round(elapsed, 3),
            error_message=None
        )

    def _process_single_live(self, target_dict: Dict[str, Any]) -> BatchTargetResult:
        """Processes a target by connecting to live NASA MAST servers."""
        from astro_exo.pipeline.runner import ExoplanetPipelineRunner
        t_start = time.perf_counter()
        tic_id = target_dict["tic_id"]

        target_cfg = TargetConfig(
            tic_id=tic_id,
            sector=target_dict.get("sector"),
            period_days=target_dict.get("period_days"),
            t0_bjd=target_dict.get("t0_bjd"),
            duration_hours=target_dict.get("duration_hours"),
            r_star_rsun=target_dict.get("r_star_rsun", 1.0),
            m_star_msun=target_dict.get("m_star_msun", 1.0),
            author=target_dict.get("author", "SPOC")
        )

        runner = ExoplanetPipelineRunner(target_cfg, self.config)
        product = runner.run()
        elapsed = time.perf_counter() - t_start

        passed = product.vetting.passed_spatial_vetting
        status = "PASSED" if passed else "REJECTED_FP"

        return BatchTargetResult(
            tic_id=tic_id,
            name=target_dict.get("name", str(tic_id)),
            toi=target_dict.get("toi", "N/A"),
            sector=target_dict.get("sector"),
            period_days=target_dict.get("period_days", 3.5),
            t0_bjd=target_dict.get("t0_bjd", 0.0),
            duration_hours=target_dict.get("duration_hours", 3.0),
            status=status,
            spatial_vetting_passed=passed,
            centroid_offset_arcsec=round(product.vetting.centroid_offset_arcsec, 2),
            centroid_offset_sigma=round(product.vetting.centroid_significance_sigma, 2),
            gaia_neighbors_screened=product.vetting.gaia_neighbors_count,
            neighbors_ruled_out=product.vetting.neighbors_ruling_out_count,
            depth_ppm=round(product.inference.rp_rs**2 * 1e6, 1) if product.inference else 0.0,
            derived_density_g_cm3=round(product.inference.stellar_density_g_cm3, 2) if product.inference else None,
            elapsed_sec=round(elapsed, 3),
            error_message=None
        )

    def process_catalog(self, filepath: str) -> List[BatchTargetResult]:
        """Executes the batch processing pipeline across all targets in the catalog."""
        targets = self.load_targets(filepath)
        n_total = len(targets)

        print("=" * 76)
        print(f"   ASTRO-EXO: PROCESSAMENTO EM LOTE ({n_total} ALVOS PILOTO)")
        print(f"   Catálogo: {os.path.basename(filepath)} | Modo: {self.mode.upper()}")
        print(f"   Diretório de Saída: {self.output_dir}")
        print("=" * 76)

        results: List[BatchTargetResult] = []
        t_batch_start = time.perf_counter()

        targets_dir = os.path.join(self.output_dir, "targets")
        os.makedirs(targets_dir, exist_ok=True)

        for idx, target_info in enumerate(targets, start=1):
            tic = target_info["tic_id"]
            name = target_info.get("name", f"TIC-{tic}")
            toi = target_info.get("toi", "N/A")
            safe_toi = str(toi).replace(".", "_")
            print(f"[{idx}/{n_total}] Processando {name} (TIC {tic})...", end=" ", flush=True)

            try:
                if self.mode == "live":
                    res = self._process_single_live(target_info)
                else:
                    res = self._process_single_mock(target_info)

                results.append(res)
                icon = "✓" if res.status == "PASSED" else "⚠"
                print(f"{icon} {res.status} [offset={res.centroid_offset_arcsec}\", {res.elapsed_sec:.2f}s]")

                # Salva resultado individual em subdiretório de alvos e na raiz para compatibilidade
                target_json_data = asdict(res)
                detailed_target_path = os.path.join(targets_dir, f"TIC_{tic}_TOI_{safe_toi}.json")
                with open(detailed_target_path, "w", encoding="utf-8") as f_det:
                    json.dump(target_json_data, f_det, indent=2)

                compat_target_path = os.path.join(self.output_dir, f"TIC_{tic}.json")
                with open(compat_target_path, "w", encoding="utf-8") as f_single:
                    json.dump(target_json_data, f_single, indent=2)

            except Exception as e:
                print(f"✗ FALHA: {type(e).__name__}: {e}")
                fail_res = BatchTargetResult(
                    tic_id=tic,
                    name=name,
                    toi=target_info.get("toi", "N/A"),
                    sector=target_info.get("sector"),
                    period_days=float(target_info.get("period_days", 0.0)),
                    t0_bjd=float(target_info.get("t0_bjd", 0.0)),
                    duration_hours=float(target_info.get("duration_hours", 0.0)),
                    status="FAILED",
                    spatial_vetting_passed=False,
                    centroid_offset_arcsec=-1.0,
                    centroid_offset_sigma=-1.0,
                    gaia_neighbors_screened=0,
                    neighbors_ruled_out=0,
                    depth_ppm=0.0,
                    derived_density_g_cm3=None,
                    elapsed_sec=0.0,
                    error_message=f"{type(e).__name__}: {e}"
                )
                results.append(fail_res)

        t_batch_total = time.perf_counter() - t_batch_start

        # Gera relatórios e metadados consolidados
        self._export_summaries(results, filepath, t_batch_total)

        # Imprime sumário final
        self._print_batch_summary(results, t_batch_total)

        return results

    def _export_summaries(self, results: List[BatchTargetResult], catalog_path: Optional[str] = None, total_time: float = 0.0):
        """Exports batch_summary.csv, batch_summary.json, and run_metadata.json."""
        import hashlib
        from datetime import datetime, timezone

        # 1. CSV consolidado
        csv_path = os.path.join(self.output_dir, "batch_summary.csv")
        fieldnames = [
            "tic_id", "name", "toi", "sector", "period_days", "t0_bjd",
            "duration_hours", "status", "spatial_vetting_passed",
            "centroid_offset_arcsec", "centroid_offset_sigma",
            "gaia_neighbors_screened", "neighbors_ruled_out",
            "depth_ppm", "derived_density_g_cm3", "elapsed_sec", "error_message"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

        # 2. JSON consolidado
        json_path = os.path.join(self.output_dir, "batch_summary.json")
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump([asdict(r) for r in results], f_json, indent=2)

        # 3. Metadados de Execução e Rastreabilidade (Provenance)
        sha256 = ""
        if catalog_path and os.path.exists(catalog_path):
            with open(catalog_path, "rb") as f_in:
                sha256 = hashlib.sha256(f_in.read()).hexdigest()

        n_total = len(results)
        n_passed = sum(1 for r in results if r.status == "PASSED")
        n_rejected = sum(1 for r in results if r.status == "REJECTED_FP")
        n_failed = sum(1 for r in results if r.status == "FAILED")

        metadata = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "0.1.0",
            "mode": self.mode,
            "input_catalog": {
                "filepath": os.path.abspath(catalog_path) if catalog_path else None,
                "sha256": sha256,
                "total_targets": n_total
            },
            "execution_metrics": {
                "total_duration_seconds": round(total_time, 4),
                "avg_duration_per_target_seconds": round(total_time / n_total, 4) if n_total > 0 else 0.0,
                "passed_count": n_passed,
                "rejected_fp_count": n_rejected,
                "failed_count": n_failed,
                "validation_rate_percent": round((n_passed / n_total) * 100.0, 1) if n_total > 0 else 0.0
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            },
            "output_files": {
                "summary_csv": os.path.abspath(csv_path),
                "summary_json": os.path.abspath(json_path),
                "targets_dir": os.path.abspath(os.path.join(self.output_dir, "targets"))
            }
        }

        meta_path = os.path.join(self.output_dir, "run_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(metadata, f_meta, indent=2)

    def export_summary(self, output_path: str) -> str:
        """Public helper to export batch results to a specific file path."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if hasattr(self, "results") and self.results:
            data = [asdict(r) for r in self.results]
        else:
            data = []
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def _print_batch_summary(self, results: List[BatchTargetResult], total_time: float):
        """Displays formatted summary table in terminal."""
        n_total = len(results)
        n_passed = sum(1 for r in results if r.status == "PASSED")
        n_rejected = sum(1 for r in results if r.status == "REJECTED_FP")
        n_failed = sum(1 for r in results if r.status == "FAILED")

        print("\n" + "=" * 76)
        print("                  RESUMO DO PROCESSAMENTO EM LOTE")
        print("=" * 76)
        print(f"{'TIC ID':<12} {'NOME':<16} {'PERÍODO':<10} {'STATUS':<14} {'OFFSET':<10} {'TEMPO'}")
        print("-" * 76)
        for r in results:
            offset_str = f"{r.centroid_offset_arcsec:.2f}\"" if r.centroid_offset_arcsec >= 0 else "N/A"
            print(f"{r.tic_id:<12} {r.name[:15]:<16} {r.period_days:<10.3f} {r.status:<14} {offset_str:<10} {r.elapsed_sec:.2f}s")
        print("-" * 76)
        print(f"Total de Alvos Processados : {n_total}")
        print(f"Candidatos Validados       : {n_passed} (Vetting Aprovado)")
        print(f"Falsos Positivos Descartados: {n_rejected} (BEB / Offset Detectado)")
        print(f"Erros de Execução          : {n_failed}")
        print(f"Tempo Total do Lote        : {total_time:.2f} segundos")
        print(f"Arquivos Gerados em        : {self.output_dir}")
        print("=" * 76)
