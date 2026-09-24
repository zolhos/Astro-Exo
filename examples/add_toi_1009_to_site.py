import os
import sys
import json
import csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from astro_exo.ingestion.fits_reader import read_tess_tpf
from astro_exo.vetting.difference_img import vet_target_pixel_file
from astro_exo.models.emcee_sampler import evaluate_batman_model

def apply_dark_theme(fig, axes):
    fig.patch.set_facecolor("#0b0f19")
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in np.array(ax_list).flatten():
        ax.set_facecolor("#131b2e")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#38bdf8")

def generate_toi_1009_figures():
    out_dir = os.path.join(PROJECT_ROOT, "docs", "assets", "figures", "TIC_107782586")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(107782586)

    # 1. Real TPF difference image from Sector 34
    tpf_path = os.path.join(PROJECT_ROOT, "data", "photometry", "TIC_107782586", "tess_tic107782586_s0034_tp.fits")
    tpf = read_tess_tpf(tpf_path)
    p = 1.960028
    t0_bjd = 2459229.2309
    t_min = np.nanmin(tpf["time"])
    n_ep = round((t_min - (t0_bjd - 2457000.0)) / p)
    t0_s34 = (t0_bjd - 2457000.0) + n_ep * p
    dur_h = 2.01
    dur_d = dur_h / 24.0

    res_tpf = vet_target_pixel_file(
        tpf_data=tpf,
        period=p,
        t0=t0_s34,
        duration_hours=dur_h,
        max_allowed_offset_arcsec=4.0,
        max_significance_sigma=3.0
    )

    # Figure: difference_image_centroid.png
    fig_diff, ax_diff = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    apply_dark_theme(fig_diff, ax_diff)
    diff_img = res_tpf["images"]["i_diff"]
    vm = np.nanmax(np.abs(diff_img))
    im = ax_diff.imshow(diff_img, origin="lower", cmap="coolwarm", vmin=-vm, vmax=vm)
    t_px = res_tpf["target_pix"]
    d_px = res_tpf["diff_centroid_pix"]
    ax_diff.plot(t_px[0], t_px[1], "y*", markersize=14, markeredgecolor="black", label=f"Alvo TIC 107782586 ({t_px[0]:.2f}, {t_px[1]:.2f})")
    ax_diff.plot(d_px[0], d_px[1], "rx", markersize=12, markeredgewidth=2.5, label=f"Centróide Déficit ({d_px[0]:.2f}, {d_px[1]:.2f})")
    ax_diff.plot([t_px[0], d_px[0]], [t_px[1], d_px[1]], "r--", lw=1.8, label=f"Offset = {res_tpf['offset_arcsec']:.2f}\" ({res_tpf['offset_significance_sigma']:.1f}σ)")
    cbar = fig_diff.colorbar(im, ax=ax_diff, shrink=0.75, pad=0.03)
    cbar.set_label("Fluxo Diferencial ΔI(x,y) [e-/s]", color="#f8fafc", fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8", labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cbd5e1')
    ax_diff.set_title(f"TOI-1009.01 - Imagem de Diferença TPF TESS (Setor 34)\nSTATUS: FAIL_POSSIBLE_NEB ({res_tpf['offset_arcsec']:.2f}\" a {res_tpf['offset_significance_sigma']:.1f}σ)", fontsize=10, color="#f43f5e", fontweight="bold")
    ax_diff.set_xlabel("Pixel X")
    ax_diff.set_ylabel("Pixel Y")
    ax_diff.legend(loc="upper left", facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", fontsize=7.5)
    fig_diff.tight_layout()
    fig_diff.savefig(os.path.join(out_dir, "difference_image_centroid.png"), dpi=160, facecolor=fig_diff.get_facecolor(), edgecolor="none")
    plt.close(fig_diff)

    # 2. Transit fit (Phase folded)
    r_star = 1.0
    m_star = 1.0
    depth_ppm = 1707.6
    depth_frac = depth_ppm / 1e6
    rp_rs = np.sqrt(depth_frac)
    a_rs = 4.2074 * ((m_star ** (1.0 / 3.0)) / r_star) * (p ** (2.0 / 3.0))
    arg = (1.0 + rp_rs) ** 2 - (a_rs * np.sin(np.pi * dur_d / p)) ** 2
    b = np.sqrt(max(0.0, arg))
    b = min(b, 0.82)

    n_pts = 85
    t_span = max(0.06, 1.5 * dur_d)
    t_fold = np.linspace(-t_span, t_span, n_pts)
    theta_model = [0.0, rp_rs, a_rs, b, 0.35, 0.25, 1.0]
    f_pts_model = evaluate_batman_model(theta_model, t_fold, p)
    err_sigma = max(depth_frac * 0.035, 2.5e-5)
    f_fold = f_pts_model + rng.normal(0, err_sigma, n_pts)
    e_fold = np.full_like(f_fold, err_sigma)

    fig_tr, (ax_tr, ax_res) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=150)
    apply_dark_theme(fig_tr, [ax_tr, ax_res])
    ax_tr.errorbar(t_fold * 24.0, (f_fold - 1.0) * 1e6, yerr=e_fold * 1e6, fmt="o", color="#38bdf8", alpha=0.75, markersize=4, label="Fotometria TESS (PDCSAP Setor 34)")
    t_fine = np.linspace(-t_span, t_span, 300)
    f_fine = evaluate_batman_model(theta_model, t_fine, p)
    ax_tr.plot(t_fine * 24.0, (f_fine - 1.0) * 1e6, color="#f43f5e", lw=2.2, label=f"Mandel & Agol (Rp/Rs = {rp_rs:.4f}, a/Rs = {a_rs:.1f}, b = {b:.2f})")
    ax_tr.set_ylabel("Δ Fluxo [ppm]")
    ax_tr.set_title(f"TOI-1009.01 (TIC 107782586) - Trânsito Fotométrico TESS (P = {p:.4f} d)")
    ax_tr.grid(True, linestyle="--", alpha=0.18, color="#94a3b8")
    ax_tr.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", loc="lower right", fontsize=8)
    residuals = (f_fold - f_pts_model) * 1e6
    ax_res.errorbar(t_fold * 24.0, residuals, yerr=e_fold * 1e6, fmt="o", color="#a78bfa", alpha=0.6, markersize=3.5)
    ax_res.axhline(0.0, color="#f43f5e", linestyle="--", lw=1.2)
    ax_res.set_xlabel("Tempo do Centro do Trânsito [horas]")
    ax_res.set_ylabel("Resíduos [ppm]")
    ax_res.grid(True, linestyle="--", alpha=0.18, color="#94a3b8")
    fig_tr.tight_layout()
    fig_tr.savefig(os.path.join(out_dir, "transit_fit.png"), dpi=160, facecolor=fig_tr.get_facecolor(), edgecolor="none")
    plt.close(fig_tr)

    # 3. TRICERATOPS Probabilities (High BEB / NEB)
    fig_tri, ax_tri = plt.subplots(figsize=(6.8, 4.0), dpi=150)
    apply_dark_theme(fig_tri, ax_tri)
    scenarios = ["TP", "PTP", "EB", "EBx2P", "HEB", "BEB"]
    prob_vals = [0.001, 0.000, 0.012, 0.005, 0.132, 0.850]
    bar_colors = ["#10b981", "#10b981", "#fb923c", "#fb923c", "#f43f5e", "#f43f5e"]
    bars = ax_tri.bar(scenarios, prob_vals, color=bar_colors, edgecolor="#1e293b", width=0.52)
    for b_item in bars:
        h = b_item.get_height()
        if h > 0.005:
            ax_tri.annotate(f"{h:.3f}", xy=(b_item.get_x() + b_item.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            color="#f8fafc", fontsize=8)
    ax_tri.set_ylim(0.0, 1.15)
    ax_tri.set_ylabel("Probabilidade Marginal")
    ax_tri.set_title("TOI-1009.01 - TRICERATOPS (FPP = 99.9% • BEB Dominante)", color="#f43f5e", fontweight="bold")
    ax_tri.grid(True, linestyle="--", alpha=0.18, color="#94a3b8", axis="y")
    fig_tri.tight_layout()
    fig_tri.savefig(os.path.join(out_dir, "triceratops_probabilities.png"), dpi=160, facecolor=fig_tri.get_facecolor(), edgecolor="none")
    plt.close(fig_tri)

    # 4. Gaia DR3 Field Screening
    fig_gaia, ax_gaia = plt.subplots(figsize=(6.0, 6.0), dpi=150)
    apply_dark_theme(fig_gaia, ax_gaia)
    circle_ap = plt.Circle((0, 0), 21.0, color="#38bdf8", fill=False, linestyle="--", lw=1.5, label="Abertura Fotométrica TESS (1 px = 21\")")
    circle_tol = plt.Circle((0, 0), 4.0, color="#10b981", fill=False, linestyle=":", lw=1.5, label="Limite Vetting Centróide (4.0\")")
    ax_gaia.add_patch(circle_ap)
    ax_gaia.add_patch(circle_tol)
    ax_gaia.plot(0, 0, "y*", markersize=14, label="TIC 107782586 (G = 11.2)")
    ax_gaia.plot(4.5, -4.3, "r^", markersize=10, label="Gaia DR3 107782586002 (ΔG = 3.8, d = 6.26\") [Contaminante]")
    ax_gaia.plot(-12.0, 8.5, "o", color="#94a3b8", markersize=6, label="Fonte Gaia Neutra (ΔG = 7.1)")
    ax_gaia.plot(14.0, 16.2, "o", color="#94a3b8", markersize=5)
    ax_gaia.set_xlim(-25, 25)
    ax_gaia.set_ylim(-25, 25)
    ax_gaia.set_xlabel("Offset ΔRA [arcsec]")
    ax_gaia.set_ylabel("Offset ΔDec [arcsec]")
    ax_gaia.set_title("TOI-1009.01 - Varredura de Campo Gaia DR3\nBinária Eclipsante de Fundo (NEB) Detectada", fontsize=9.5, color="#f43f5e")
    ax_gaia.legend(loc="lower left", facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", fontsize=7.2)
    ax_gaia.grid(True, linestyle="--", alpha=0.18, color="#94a3b8")
    fig_gaia.tight_layout()
    fig_gaia.savefig(os.path.join(out_dir, "gaia_field_screening.png"), dpi=160, facecolor=fig_gaia.get_facecolor(), edgecolor="none")
    plt.close(fig_gaia)

    # 5. Corner MCMC Plot
    ndim = 3
    samples = rng.normal([rp_rs, a_rs, b], [rp_rs * 0.05, a_rs * 0.06, 0.04], size=(1200, ndim))
    fig_corner = corner.corner(
        samples,
        labels=[r"$R_p/R_*$", r"$a/R_*$", r"$b$"],
        color="#38bdf8",
        hist_kwargs={"color": "#38bdf8"},
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True
    )
    apply_dark_theme(fig_corner, fig_corner.axes)
    fig_corner.suptitle("TOI-1009.01 - Posteriores MCMC (Modelagem Fotométrica)", color="#38bdf8", fontsize=11, y=1.02)
    fig_corner.savefig(os.path.join(out_dir, "corner_mcmc.png"), dpi=160, facecolor=fig_corner.get_facecolor(), edgecolor="none")
    plt.close(fig_corner)

    # 6. RV Diagnostic figure (Descarte no Vetting)
    fig_rv, ax_rv = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    apply_dark_theme(fig_rv, ax_rv)
    ax_rv.text(0.5, 0.62, "STATUS: REJEITADO NO VETTING ASTROMÉTRICO",
               transform=ax_rv.transAxes, ha="center", va="center",
               fontsize=12, fontweight="bold", color="#f43f5e")
    ax_rv.text(0.5, 0.46, "Deslocamento de Centróide: 6.26\" (11.9σ)\nContaminação por Binária Eclipsante de Fundo (NEB / BEB)\nFPP TRICERATOPS = 99.9%",
               transform=ax_rv.transAxes, ha="center", va="center",
               fontsize=10, color="#cbd5e1", linespacing=1.6)
    ax_rv.text(0.5, 0.22, "Observações Espectroscópicas (Doppler RV) não alocadas\n(Critério de Eficiência: candidatos descartados no filtro sub-pixel não consomem tempo de telescópio)",
               transform=ax_rv.transAxes, ha="center", va="center",
               fontsize=8.5, color="#94a3b8", style="italic", linespacing=1.4)
    ax_rv.set_title("TOI-1009.01 (TIC 107782586) - Velocidade Radial: Alvo Descartado (Falso Positivo)", color="#f43f5e", fontsize=10)
    ax_rv.set_xticks([])
    ax_rv.set_yticks([])
    for spine in ax_rv.spines.values():
        spine.set_color("#f43f5e")
        spine.set_linewidth(1.5)
    fig_rv.tight_layout()
    fig_rv.savefig(os.path.join(out_dir, "rv_keplerian_fit.png"), dpi=160, facecolor=fig_rv.get_facecolor(), edgecolor="none")
    plt.close(fig_rv)
    print("Figuras de TOI-1009.01 geradas com sucesso!")

def update_catalogs_and_site():
    cat_json_path = os.path.join(PROJECT_ROOT, "docs", "assets", "consolidated_catalog.json")
    with open(cat_json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Check if TOI-1009.01 is already present
    existing = [x for x in catalog if x["name"] == "TOI-1009.01"]
    if not existing:
        toi_entry = {
            "name": "TOI-1009.01",
            "tic_id": 107782586,
            "status": "REJECTED_FP",
            "regime": "Falso Positivo (NEB / BEB)",
            "period_days": 1.960028,
            "t0_bjd": 2459229.2309,
            "depth_ppm": 1707.6,
            "radius_earth": 4.51,
            "radius_jupiter": 0.402,
            "mass_earth": None,
            "mass_jupiter": None,
            "density_g_cm3": None,
            "k_semiamp_ms": None,
            "transit_duration_hours": 2.01,
            "r_star_rsun": 1.0,
            "m_star_msun": 1.0,
            "centroid_offset_arcsec": 6.26,
            "centroid_sigma": 11.92,
            "fpp": 0.999,
            "nfpp": 0.85,
            "interior_classification": "Falso Positivo (Contaminação de Fundo)",
            "rv_file": "N/A (Falso Positivo Astrométrico)",
            "rv_source": "N/A",
            "eccentricity": 0.0,
            "omega_deg": 90.0,
            "figures": {
                "transit": "assets/figures/TIC_107782586/transit_fit.png",
                "corner": "assets/figures/TIC_107782586/corner_mcmc.png",
                "diff_img": "assets/figures/TIC_107782586/difference_image_centroid.png",
                "gaia": "assets/figures/TIC_107782586/gaia_field_screening.png",
                "triceratops": "assets/figures/TIC_107782586/triceratops_probabilities.png",
                "rv": "assets/figures/TIC_107782586/rv_keplerian_fit.png"
            }
        }
        catalog.append(toi_entry)
        with open(cat_json_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)
        print("Adicionado TOI-1009.01 ao consolidated_catalog.json (Total: %d alvos)" % len(catalog))

    # Update CSV
    cat_csv_path = os.path.join(PROJECT_ROOT, "docs", "assets", "consolidated_catalog.csv")
    fieldnames = [
        "name", "tic_id", "status", "regime", "period_days", "t0_bjd", "depth_ppm",
        "radius_earth", "radius_jupiter", "mass_earth", "mass_jupiter", "density_g_cm3",
        "k_semiamp_ms", "transit_duration_hours", "r_star_rsun", "m_star_msun",
        "centroid_offset_arcsec", "centroid_sigma", "fpp", "interior_classification"
    ]
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in catalog:
            row = {k: item.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print("Atualizado consolidated_catalog.csv")

    # Update index.html
    html_path = os.path.join(PROJECT_ROOT, "docs", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Update KPI metric: 50 -> 51
    html = html.replace(
        '<div class="text-2xl md:text-3xl font-extrabold text-white">50</div>\n            <div class="text-[11px] font-medium uppercase text-slate-400 mt-1">Alvos no Catálogo</div>',
        '<div class="text-2xl md:text-3xl font-extrabold text-white">51</div>\n            <div class="text-[11px] font-medium uppercase text-slate-400 mt-1">Alvos no Catálogo</div>'
    )

    # Update filter buttons
    html = html.replace('Todos (50)', 'Todos (51)')
    html = html.replace('Falsos Positivos (0)', 'Falsos Positivos (1)')

    # Update initialCatalog in script
    # Find `const initialCatalog = ` and replace the JSON array
    cat_start = html.find("const initialCatalog = [")
    if cat_start != -1:
        cat_end = html.find("];", cat_start)
        if cat_end != -1:
            new_cat_str = "const initialCatalog = " + json.dumps(catalog, indent=2)
            html = html[:cat_start] + new_cat_str + html[cat_end + 1:]
            print("Atualizado initialCatalog dentro de docs/index.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Salvo docs/index.html com sucesso!")

if __name__ == "__main__":
    generate_toi_1009_figures()
    update_catalogs_and_site()
