import os
import sys
import json
import time
from typing import Dict, Any, List

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from astro_exo.ingestion.mast_client import fetch_tess_photometry
from astro_exo.ingestion.fits_reader import read_tess_lightcurve, read_tess_tpf
from astro_exo.vetting.difference_img import vet_target_pixel_file
import wotan


BATCH_2_TARGETS = [
    {
        "tic_id": 144065872,
        "name": "TOI-105.01 (WASP-95b)",
        "toi": "105.01",
        "period_days": 2.184667,
        "t0_btjd": 3904.4127,
        "duration_hours": 2.70,
        "catalog_depth_ppm": 11535.5,
        "expected_disp": "KP"
    },
    {
        "tic_id": 16288184,
        "name": "TOI-1049.01 (WASP-77b)",
        "toi": "1049.01",
        "period_days": 2.180533,
        "t0_btjd": 1626.8015,
        "duration_hours": 2.94,
        "catalog_depth_ppm": 16399.9,
        "expected_disp": "KP"
    },
    {
        "tic_id": 107782586,
        "name": "TOI-1009.01",
        "toi": "1009.01",
        "period_days": 1.960028,
        "t0_btjd": 2229.2309,
        "duration_hours": 2.01,
        "catalog_depth_ppm": 1707.6,
        "expected_disp": "PC"
    },
    {
        "tic_id": 341420329,
        "name": "TOI-1019.01",
        "toi": "1019.01",
        "period_days": 5.234101,
        "t0_btjd": 2232.4469,
        "duration_hours": 3.71,
        "catalog_depth_ppm": 20796.9,
        "expected_disp": "PC"
    }
]


def adjust_epoch_to_dataset(t0_ref: float, period: float, time_arr: np.ndarray) -> float:
    t_mid = float(np.nanmedian(time_arr))
    n_cycles = round((t_mid - t0_ref) / period)
    return float(t0_ref + n_cycles * period)


def process_sample(target_info: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    tic = target_info["tic_id"]
    name = target_info["name"]
    period = target_info["period_days"]
    duration_h = target_info["duration_hours"]
    t0_ref = target_info["t0_btjd"]

    print(f"\n=======================================================")
    print(f"Processando Novo Alvo: {name} (TIC {tic})")
    print(f"Período: {period:.4f} d | Duração: {duration_h:.2f} h")
    print(f"=======================================================")

    t_start = time.perf_counter()

    # 1. Download de Curva de Luz e TPF (com cache local)
    lc_path = fetch_tess_photometry(tic, product="lc")
    tpf_path = fetch_tess_photometry(tic, product="tp")

    # 2. Leitura FITS da Curva de Luz
    lc_data = read_tess_lightcurve(lc_path)
    time_arr = lc_data["time"]
    flux_arr = lc_data["flux"]
    sector = lc_data["sector"]

    # 3. Ajuste de época para o setor observado
    t0_sector = adjust_epoch_to_dataset(t0_ref, period, time_arr)
    print(f"Setor {sector} | t0 ajustado: {t0_sector:.4f} BTJD")

    # 4. Desestacionalização com wotan (biweight)
    flatten_lc, trend_lc = wotan.flatten(
        time_arr,
        flux_arr,
        method="biweight",
        window_length=0.75,
        edge_cutoff=0.5,
        break_tolerance=0.5,
        return_trend=True
    )

    # Medição da profundidade observada
    phase = (time_arr - t0_sector + 0.5 * period) % period - 0.5 * period
    dur_days = duration_h / 24.0
    in_tr = np.abs(phase) <= (0.5 * dur_days)
    out_tr = (np.abs(phase) >= (0.75 * dur_days)) & (np.abs(phase) <= (1.75 * dur_days))

    obs_depth = float(np.nanmedian(flatten_lc[out_tr]) - np.nanmedian(flatten_lc[in_tr]))
    obs_depth_ppm = obs_depth * 1e6

    # 5. Leitura FITS do TPF com WCS
    tpf_data = read_tess_tpf(tpf_path)

    # 6. Vetting Espacial de Imagem de Diferença e Centróide
    vet_res = vet_target_pixel_file(
        tpf_data=tpf_data,
        period=period,
        t0=t0_sector,
        duration_hours=duration_h,
        max_allowed_offset_arcsec=10.0
    )

    dt = time.perf_counter() - t_start

    print(f"Status Vetting: {vet_res['status']}")
    print(f"Offset Astrométrico: {vet_res['offset_arcsec']:.2f}\" ({vet_res['offset_pix']:.3f} pix)")
    print(f"Profundidade Observada: {obs_depth_ppm:.1f} ppm (Catálogo: {target_info['catalog_depth_ppm']:.1f} ppm)")
    print(f"Concluído em: {dt:.2f} s")

    diag_data = {
        "tic_id": tic,
        "name": name,
        "toi": target_info["toi"],
        "sector": sector,
        "period_days": period,
        "t0_sector_btjd": t0_sector,
        "duration_hours": duration_h,
        "catalog_depth_ppm": target_info["catalog_depth_ppm"],
        "observed_depth_ppm": obs_depth_ppm,
        "vetted_status": vet_res["status"],
        "vetted_passed": vet_res["passed"],
        "offset_arcsec": vet_res["offset_arcsec"],
        "offset_pix": vet_res["offset_pix"],
        "sigma_offset_arcsec": vet_res["sigma_offset_arcsec"],
        "offset_significance_sigma": vet_res["offset_significance_sigma"],
        "target_pix": vet_res["target_pix"],
        "diff_centroid_pix": vet_res["diff_centroid_pix"],
        "cadences_count": len(time_arr),
        "execution_time_seconds": dt
    }

    targets_dir = os.path.join(output_dir, "targets")
    os.makedirs(targets_dir, exist_ok=True)
    diag_file = os.path.join(targets_dir, f"TIC_{tic}_TOI_{target_info['toi']}.json")
    with open(diag_file, "w") as f:
        json.dump(diag_data, f, indent=2)

    return {
        "summary": diag_data,
        "time": time_arr,
        "flatten_lc": flatten_lc,
        "trend_lc": trend_lc,
        "flux": flux_arr,
        "phase": phase,
        "t0_sector": t0_sector,
        "tpf": tpf_data,
        "vet_res": vet_res
    }


def main():
    output_dir = "results/phase2_samples_batch_2"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    processed_artifacts = []

    for t in BATCH_2_TARGETS:
        res = process_sample(t, output_dir)
        results.append(res["summary"])
        processed_artifacts.append(res)

    import csv
    summary_json = os.path.join(output_dir, "batch_summary.json")
    summary_csv = os.path.join(output_dir, "batch_summary.csv")

    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)

    fieldnames = list(results[0].keys())
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResumo do lote salvo em: {summary_json}")

    # Painel Científico Comparativo das 4 novas amostras
    print("\nGerando painel comparativo das 4 novas amostras...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), facecolor="#0B132B")
    plt.subplots_adjust(hspace=0.45, wspace=0.25)

    plt.rcParams["text.color"] = "#E0E1DD"
    plt.rcParams["axes.labelcolor"] = "#E0E1DD"
    plt.rcParams["xtick.color"] = "#8D99AE"
    plt.rcParams["ytick.color"] = "#8D99AE"

    for idx, art in enumerate(processed_artifacts):
        s = art["summary"]
        time_arr = art["time"]
        flux_arr = art["flux"]
        trend_arr = art["trend_lc"]
        flatten_arr = art["flatten_lc"]
        phase_arr = art["phase"]
        vet_res = art["vet_res"]
        i_diff = vet_res["images"]["i_diff"]
        target_pix = s["target_pix"]
        diff_cen = s["diff_centroid_pix"]

        # 1. Curva de Luz Bruta + Tendência wotan
        ax1 = axes[idx, 0]
        ax1.set_facecolor("#1C2541")
        ax1.scatter(time_arr, flux_arr, s=1, c="#48CAE4", alpha=0.4, label="PDCSAP Flux")
        ax1.plot(time_arr, trend_arr, c="#F72585", lw=1.5, label="wotan biweight")
        ax1.set_title(f"{s['name']} (TIC {s['tic_id']}, Setor {s['sector']})", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Tempo (BTJD)", fontsize=8)
        ax1.set_ylabel("Fluxo Norm.", fontsize=8)
        ax1.grid(True, color="#2D3A5D", linestyle="--", alpha=0.4)
        if idx == 0:
            ax1.legend(loc="lower right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=7)

        # 2. Trânsito Dobrado em Fase
        ax2 = axes[idx, 1]
        ax2.set_facecolor("#1C2541")
        p_hrs = phase_arr * 24.0
        sort_p = np.argsort(p_hrs)
        ax2.scatter(p_hrs[sort_p], flatten_arr[sort_p], s=1.5, c="#90E0EF", alpha=0.3)

        dur_h = s["duration_hours"]
        window_h = max(dur_h * 1.5, 3.0)
        bins = np.linspace(-window_h, window_h, 30)
        bc = 0.5 * (bins[1:] + bins[:-1])
        bm = np.zeros(len(bc))
        for b_i in range(len(bc)):
            mb = (p_hrs >= bins[b_i]) & (p_hrs < bins[b_i+1])
            bm[b_i] = np.nanmedian(flatten_arr[mb]) if np.sum(mb) > 0 else np.nan

        valid_b = ~np.isnan(bm)
        ax2.plot(bc[valid_b], bm[valid_b], "o-", c="#FFB703", lw=2, ms=3.5, label="Binned")
        ax2.set_xlim(-window_h, window_h)
        ax2.set_title(f"Dobrado em Fase (P = {s['period_days']:.3f} d | Prof: {s['observed_depth_ppm']:.0f} ppm)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Fase (Horas)", fontsize=8)
        ax2.set_ylabel("Fluxo Detrended", fontsize=8)
        ax2.grid(True, color="#2D3A5D", linestyle="--", alpha=0.4)

        # 3. Imagem de Diferença e Vetting WCS
        ax3 = axes[idx, 2]
        ax3.set_facecolor("#1C2541")
        im3 = ax3.imshow(i_diff, origin="lower", cmap="magma")
        ax3.plot(target_pix[0], target_pix[1], "rx", ms=9, mew=2, label="Alvo (WCS)")
        ax3.plot(diff_cen[0], diff_cen[1], "co", ms=7, mew=2, label="Centróide $\Delta I$")
        status_color = "#4CAF50" if s["vetted_passed"] else "#F44336"
        ax3.set_title(f"Offset: {s['offset_arcsec']:.2f}\" ({s['offset_pix']:.3f} pix) - {s['vetted_status']}", fontsize=10, fontweight="bold", color=status_color)
        ax3.set_xlabel("Pixel X", fontsize=8)
        ax3.set_ylabel("Pixel Y", fontsize=8)
        if idx == 0:
            ax3.legend(loc="upper right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=7)

    fig_path = os.path.join(output_dir, "phase2_batch2_diagnostics.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figura comparativa salva em: {fig_path}")


if __name__ == "__main__":
    main()
