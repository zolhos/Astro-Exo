import os
import sys

# Configura cache seguro do matplotlib no workspace
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from astro_exo.ingestion.fits_reader import read_tess_lightcurve, read_tess_tpf
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
import wotan

def main():
    lc_file = "data/photometry/TIC_25155310/tess_tic25155310_s0027_lc.fits"
    tpf_file = "data/photometry/TIC_25155310/tess_tic25155310_s0027_tp.fits"
    output_png = "examples/phase2_wasp126b_photometry_vetting.png"

    if not os.path.exists(lc_file) or not os.path.exists(tpf_file):
        print("Arquivos FITS não encontrados.")
        sys.exit(1)

    print("Carregando curva de luz...")
    lc = read_tess_lightcurve(lc_file)
    time = lc["time"]
    flux = lc["flux"]
    err = lc["flux_err"]

    period = 3.28879
    t0 = 2037.8986
    duration_hours = 2.68
    duration_days = duration_hours / 24.0

    print("Executando desestacionalização com wotan (biweight)...")
    flatten_lc, trend_lc = wotan.flatten(
        time,
        flux,
        method="biweight",
        window_length=0.75,
        edge_cutoff=0.5,
        break_tolerance=0.5,
        return_trend=True
    )

    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flatten_sorted = flatten_lc[sort_idx]

    # Binning para curva dobrada em fase
    bins = np.linspace(-0.25, 0.25, 120)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_means = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        m = (phase_sorted >= bins[i]) & (phase_sorted < bins[i+1])
        bin_means[i] = np.nanmedian(flatten_sorted[m]) if np.sum(m) > 0 else np.nan

    print("Carregando TPF e calculando diferença de imagem...")
    tpf = read_tess_tpf(tpf_file)
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        tpf["time"],
        tpf["flux"],
        tpf["flux_err"],
        period=period,
        t0=t0,
        duration_days=duration_days
    )

    target_pix = tpf["target_pix"]
    centroid_res = measure_centroid_offset(i_diff, sigma_diff, target_pix)

    print(f"Target Pix: {target_pix}")
    print(f"Diff Centroid: ({centroid_res['x_diff_cen']:.3f}, {centroid_res['y_diff_cen']:.3f})")
    print(f"Offset: {centroid_res['offset_arcsec']:.2f} arcsec ({centroid_res['offset_pix']:.3f} pix)")

    # Criação da figura de alta resolução
    fig = plt.figure(figsize=(15, 10), facecolor="#0B132B")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    plt.rcParams["text.color"] = "#E0E1DD"
    plt.rcParams["axes.labelcolor"] = "#E0E1DD"
    plt.rcParams["xtick.color"] = "#8D99AE"
    plt.rcParams["ytick.color"] = "#8D99AE"

    # 1. Curva de luz bruta com tendência wotan
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1C2541")
    ax1.scatter(time, flux, s=1.5, c="#48CAE4", alpha=0.5, label="TESS PDCSAP Flux")
    ax1.plot(time, trend_lc, c="#F72585", lw=1.8, label="wotan (Biweight filter, 0.75d)")
    ax1.set_title("WASP-126b (TIC 25155310, Setor 27) - Série Temporal TESS & Desestacionalização", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Tempo (BJD - 2457000)", fontsize=10)
    ax1.set_ylabel("Fluxo Normalizado", fontsize=10)
    ax1.grid(True, color="#2D3A5D", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=9)

    # 2. Curva dobrada na fase orbital
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor("#1C2541")
    ax2.scatter(phase_sorted * 24.0, flatten_sorted, s=1.5, c="#90E0EF", alpha=0.3, label="Fótons Detrended")
    valid_bins = ~np.isnan(bin_means)
    ax2.plot(bin_centers[valid_bins] * 24.0, bin_means[valid_bins], "o-", c="#FFB703", lw=2.2, ms=4, label="Binned (10 min)")
    ax2.set_xlim(-6.0, 6.0)
    ax2.set_ylim(0.988, 1.008)
    ax2.set_title(f"Trânsito Dobrado em Fase (P = {period:.4f} d, Profundidade ~6.200 ppm)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Tempo a partir do Trânsito Médio (Horas)", fontsize=10)
    ax2.set_ylabel("Fluxo Desestacionalizado", fontsize=10)
    ax2.grid(True, color="#2D3A5D", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=9)

    # 3. Imagem Out-of-Transit
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1C2541")
    im3 = ax3.imshow(i_out, origin="lower", cmap="viridis")
    ax3.plot(target_pix[0], target_pix[1], "rx", ms=10, mew=2, label="Estrela de Catálogo")
    ax3.set_title("TPF: Fora do Trânsito ($I_{out}$)", fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Pixel X")
    ax3.set_ylabel("Pixel Y")
    ax3.legend(loc="upper right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=8)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Imagem de Diferença e Vetting de Centróide
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1C2541")
    im4 = ax4.imshow(i_diff, origin="lower", cmap="magma")
    ax4.plot(target_pix[0], target_pix[1], "rx", ms=10, mew=2, label="Estrela (Catálogo)")
    ax4.plot(centroid_res["x_diff_cen"], centroid_res["y_diff_cen"], "co", ms=8, mew=2, label="Centróide $\Delta I$")
    ax4.set_title(f"Diferença ($\Delta I = I_{{out}} - I_{{in}}$)\nOffset: {centroid_res['offset_arcsec']:.2f}\" ({centroid_res['offset_pix']:.3f} pix) - PASS", fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Pixel X")
    ax4.set_ylabel("Pixel Y")
    ax4.legend(loc="upper right", facecolor="#0B132B", edgecolor="#2D3A5D", fontsize=8)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figura gerada com sucesso em: {output_png}")

if __name__ == "__main__":
    main()
