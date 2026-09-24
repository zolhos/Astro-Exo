"""
Phase 4 Master Execution Script:
Multispectral Gaia DR3 Dilution Screening, Analytical De-dilution,
and TRICERATOPS Bayesian False Positive Statistical Validation across 9 Real TESS Systems.
"""

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
from astropy.io import fits

from astro_exo.vetting.gaia import query_gaia_neighbors, overlay_gaia_on_wcs, estimate_tess_mag_from_gaia
from astro_exo.vetting.dilution import (
    calculate_critical_delta_mag,
    calculate_max_transit_depth,
    estimate_aperture_flux_fraction,
    rule_out_neighbors_as_blends,
    calculate_dilution_factor,
    restore_true_radius_ratio,
    correct_binary_blend_wasp77
)
from astro_exo.vetting.triceratops_vet import BayesianFalsePositiveEngine, run_triceratops_validation


ALL_9_TARGETS = [
    {
        "name": "WASP-126b",
        "tic_id": 25155310,
        "sector": 27,
        "tpf_file": "data/photometry/TIC_25155310/tess_tic25155310_s0027_tp.fits",
        "period_days": 3.28879,
        "rp_rs": 0.0763,
        "rp_rs_err": 0.0011,
        "depth_ppm": 5818.0,
        "b": 0.238,
        "duration_hours": 3.44,
        "centroid_offset_arcsec": 0.35,
        "centroid_sigma_arcsec": 0.65,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "WASP-62b",
        "tic_id": 149603524,
        "sector": 27,
        "tpf_file": "data/photometry/TIC_149603524/tess_tic149603524_s0027_tp.fits",
        "period_days": 4.411937,
        "rp_rs": 0.1116,
        "rp_rs_err": 0.0007,
        "depth_ppm": 12458.0,
        "b": 0.268,
        "duration_hours": 3.78,
        "centroid_offset_arcsec": 0.22,
        "centroid_sigma_arcsec": 0.55,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "HATS-3b",
        "tic_id": 336732616,
        "sector": 1,
        "tpf_file": "data/photometry/TIC_336732616/tess_tic336732616_s0001_tp.fits",
        "period_days": 3.547854,
        "rp_rs": 0.0988,
        "rp_rs_err": 0.0018,
        "depth_ppm": 9761.0,
        "b": 0.575,
        "duration_hours": 3.52,
        "centroid_offset_arcsec": 0.42,
        "centroid_sigma_arcsec": 0.70,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "WASP-95b",
        "tic_id": 144065872,
        "sector": 28,
        "tpf_file": "data/photometry/TIC_144065872/tess_tic144065872_s0028_tp.fits",
        "period_days": 2.18467,
        "rp_rs": 0.1015,
        "rp_rs_err": 0.0018,
        "depth_ppm": 10306.0,
        "b": 0.405,
        "duration_hours": 2.87,
        "centroid_offset_arcsec": 0.28,
        "centroid_sigma_arcsec": 0.60,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "WASP-46b",
        "tic_id": 231663901,
        "sector": 27,
        "tpf_file": "data/photometry/TIC_231663901/tess_tic231663901_s0027_tp.fits",
        "period_days": 1.43037,
        "rp_rs": 0.1406,
        "rp_rs_err": 0.0023,
        "depth_ppm": 19773.0,
        "b": 0.728,
        "duration_hours": 1.67,
        "centroid_offset_arcsec": 0.38,
        "centroid_sigma_arcsec": 0.75,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "TOI-1050.01",
        "tic_id": 66818296,
        "sector": 38,
        "tpf_file": "data/photometry/TIC_66818296/tess_tic66818296_s0038_tp.fits",
        "period_days": 3.735484,
        "rp_rs": 0.1223,
        "rp_rs_err": 0.0012,
        "depth_ppm": 14966.0,
        "b": 0.349,
        "duration_hours": 4.32,
        "centroid_offset_arcsec": 0.30,
        "centroid_sigma_arcsec": 0.68,
        "classification": "PLANET_CANDIDATE"
    },
    {
        "name": "TOI-1009.01",
        "tic_id": 107782586,
        "sector": 34,
        "tpf_file": "data/photometry/TIC_107782586/tess_tic107782586_s0034_tp.fits",
        "period_days": 1.960028,
        "rp_rs": 0.0308,
        "rp_rs_err": 0.0012,
        "depth_ppm": 951.1,
        "b": 0.161,
        "duration_hours": 1.99,
        "centroid_offset_arcsec": 0.45,
        "centroid_sigma_arcsec": 0.85,
        "classification": "SHALLOW_SUB_NEPTUNE"
    },
    {
        "name": "WASP-77b",
        "tic_id": 16288184,
        "sector": 39,
        "tpf_file": "data/photometry/TIC_16288184/tess_tic16288184_s0039_tp.fits",
        "period_days": 2.180533,
        "rp_rs": 0.1186,
        "rp_rs_err": 0.0019,
        "depth_ppm": 14074.0,
        "b": 0.218,
        "duration_hours": 2.97,
        "centroid_offset_arcsec": 0.31,
        "centroid_sigma_arcsec": 0.65,
        "classification": "DILUTED_BINARY_PAIR"
    },
    {
        "name": "TOI-1019.01",
        "tic_id": 341420329,
        "sector": 35,
        "tpf_file": "data/photometry/TIC_341420329/tess_tic341420329_s0035_tp.fits",
        "period_days": 5.234101,
        "rp_rs": 0.1408,
        "rp_rs_err": 0.0016,
        "depth_ppm": 19830.0,
        "b": 0.665,
        "duration_hours": 3.71,
        "centroid_offset_arcsec": 16.36,
        "centroid_sigma_arcsec": 1.45,
        "classification": "FALSE_POSITIVE_NEB"
    }
]


def plot_gaia_field_overlay(
    target_name: str,
    tic_id: int,
    target_ra: float,
    target_dec: float,
    target_tmag: float,
    neighbors: List[Dict[str, Any]],
    delta_m_crit: float,
    output_path: str
):
    """Generates a high-resolution sky map overlay of Gaia DR3 neighbors within 2.5 arcmin."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Center target at (0, 0)
    ax.scatter(0, 0, c="crimson", s=250, marker="*", label=f"Target: TIC {tic_id} (T={target_tmag:.2f})", zorder=5)

    # Plot TESS pixel scale aperture circles (core = 21", boundary = 42")
    core_circle = plt.Circle((0, 0), 21.0, color="gray", fill=False, linestyle="--", lw=1.5, label="1 TESS Pixel (21\")")
    ap_circle = plt.Circle((0, 0), 42.0, color="royalblue", fill=False, linestyle="-", lw=1.8, label="Aperture Radius (42\")")
    ax.add_patch(core_circle)
    ax.add_patch(ap_circle)

    for star in neighbors:
        dist = star.get("dist_arcsec", 0.0)
        if dist < 0.1:
            continue

        ra_diff = (star["ra"] - target_ra) * 3600.0 * np.cos(np.radians(target_dec))
        dec_diff = (star["dec"] - target_dec) * 3600.0

        tmag = star.get("tess_mag", star.get("phot_g_mean_mag", 99.0))
        delta_m = tmag - target_tmag
        can_cause = star.get("can_cause_transit", False)

        size = max(20.0, (20.0 - tmag) * 20.0)
        color = "darkorange" if can_cause else "lightgray"
        edgecolor = "black" if can_cause else "gray"

        ax.scatter(ra_diff, dec_diff, s=size, c=color, edgecolors=edgecolor, alpha=0.85, zorder=4)

        label_txt = f"T={tmag:.1f}\n(Δm={delta_m:+.1f})"
        ax.annotate(label_txt, (ra_diff + 3.0, dec_diff + 3.0), fontsize=8, color="#222")

    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    ax.set_xlabel("ΔRA [arcsec]", fontsize=11)
    ax.set_ylabel("ΔDec [arcsec]", fontsize=11)
    ax.set_title(f"{target_name} - Gaia DR3 Field & Dilution Screen (Δm_crit = {delta_m_crit:.2f} mag)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_triceratops_probabilities(
    target_name: str,
    tic_id: int,
    res: Dict[str, Any],
    output_path: str
):
    """Generates scenario probability bar chart."""
    probs = res["probabilities"]
    scenarios = ["TP", "PTP", "EB", "EBx2P", "HEB", "BEB"]
    values = [probs.get(s, 0.0) for s in scenarios]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(scenarios, values, color=colors, edgecolor="black", width=0.55)

    for bar, val in zip(bars, values):
        if val > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val*100:.2f}%", ha="center", va="bottom", fontsize=10, weight="bold")

    fpp_str = f"{res['fpp']*100:.3f}%"
    nfpp_str = f"{res['nfpp']*100:.4f}%"
    status_str = "VALIDATED" if res["validated"] else "REJECTED (FALSE POSITIVE)"
    status_color = "darkgreen" if res["validated"] else "crimson"

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Posterior Probability", fontsize=11)
    ax.set_title(f"{target_name} (TIC {tic_id}) - TRICERATOPS Scenario Probabilities\nFPP = {fpp_str} | NFPP = {nfpp_str} -> {status_str}", fontsize=11, color=status_color, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_wasp77b_dedilution(
    res_wasp77: Dict[str, Any],
    output_path: str
):
    """Generates transit comparison curve for WASP-77b diluted vs de-diluted."""
    fig, ax = plt.subplots(figsize=(8, 5))

    phases = np.linspace(-0.06, 0.06, 500)  # phase in days
    dur_days = 2.97 / 24.0

    # Trapezoid-like smooth transit profile
    def get_transit_curve(depth):
        in_transit = np.abs(phases) < (0.5 * dur_days)
        ingress = (np.abs(phases) >= (0.35 * dur_days)) & (np.abs(phases) <= (0.5 * dur_days))
        curve = np.ones_like(phases)
        curve[in_transit] -= depth
        # smooth ingress/egress
        ramp = (0.5 * dur_days - np.abs(phases[ingress])) / (0.15 * dur_days)
        curve[ingress] = 1.0 - depth * ramp
        return curve

    curve_diluted = get_transit_curve(res_wasp77["observed_depth_ppm"] * 1e-6)
    curve_true = get_transit_curve(res_wasp77["true_depth_ppm"] * 1e-6)

    ax.plot(phases * 24.0, curve_diluted, color="#3498db", lw=2.5, linestyle="--", label=f"Diluted Observed (TESS): Rp/Rs = {res_wasp77['observed_rp_rs']:.4f} (Depth: {res_wasp77['observed_depth_ppm']:.0f} ppm)")
    ax.plot(phases * 24.0, curve_true, color="#e74c3c", lw=2.5, label=f"De-diluted True (WASP-77A): Rp/Rs = {res_wasp77['true_rp_rs']:.4f} (Depth: {res_wasp77['true_depth_ppm']:.0f} ppm)")

    ax.set_xlabel("Time from Mid-Transit [hours]", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    ax.set_title("WASP-77b Transit De-dilution (Correcting WASP-77B at 3.3\" - Dilution Factor D = 0.8022)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def process_phase4_target(target: Dict[str, Any], output_base_dir: str) -> Dict[str, Any]:
    tic = target["tic_id"]
    name = target["name"]
    tpf_file = target["tpf_file"]

    print(f"\n{'='*75}")
    print(f"[PHASE 4] Vetting & Validation: {name} (TIC {tic}, Sector {target['sector']})")
    print(f"Context: {target['classification']}")
    print(f"{'='*75}")

    t_start = time.time()

    # Read FITS header for target coordinates & magnitude
    with fits.open(tpf_file) as hdul:
        h = hdul[0].header
        ra_target = float(h.get("RA_OBJ", 0.0))
        dec_target = float(h.get("DEC_OBJ", 0.0))
        tmag_target = float(h.get("TESSMAG", 10.0))

    print(f"[*] Target Coordinates: RA={ra_target:.5f}°, Dec={dec_target:.5f}°, Tmag={tmag_target:.2f}")

    # 1. Query Gaia DR3 neighbors (2.5 arcmin cone search)
    neighbors = query_gaia_neighbors(
        ra_deg=ra_target,
        dec_deg=dec_target,
        radius_arcmin=2.5,
        target_tic=tic
    )
    print(f"[*] Retrieved {len(neighbors)} Gaia DR3 neighbors within 2.5 arcmin")

    # 2. Critical Delta Magnitude & Screening
    delta_m_crit = calculate_critical_delta_mag(target["depth_ppm"])
    vetted_neighbors = rule_out_neighbors_as_blends(
        observed_transit_depth_ppm=target["depth_ppm"],
        target_mag=tmag_target,
        neighbors=neighbors
    )
    ruled_out_count = sum(1 for n in vetted_neighbors if n.get("ruled_out", False))
    print(f"[*] Critical Delta Mag: Δm_crit = {delta_m_crit:.2f} mag (Threshold for 100% eclipse)")
    print(f"[*] Screened neighbors: {ruled_out_count}/{len(vetted_neighbors)} ruled out as potential blends")

    # 3. Dilution Factor
    dilution_info = calculate_dilution_factor(
        target_mag=tmag_target,
        neighbors=vetted_neighbors
    )
    d_factor = dilution_info["dilution_factor"]
    print(f"[*] Aperture Dilution Factor: D = {d_factor:.4f} (Contamination ratio: {dilution_info['contamination_ratio']*100:.2f}%)")

    # 4. De-dilution check (special solver for WASP-77b)
    wasp77_details = None
    if tic == 16288184:
        wasp77_details = correct_binary_blend_wasp77(
            rp_rs_diluted=target["rp_rs"],
            rp_rs_err=target["rp_rs_err"],
            delta_tmag=1.52
        )
        print(f"  [✓] WASP-77b De-dilution Restored:")
        print(f"      - Diluted Rp/Rs: {wasp77_details['observed_rp_rs']:.4f} +/- {wasp77_details['observed_rp_rs_err']:.4f}")
        print(f"      - True Rp/Rs:    {wasp77_details['true_rp_rs']:.4f} +/- {wasp77_details['true_rp_rs_err']:.4f}")
        print(f"      - True Depth:    {wasp77_details['true_depth_ppm']:.1f} ppm (Depth correction: +{wasp77_details['depth_correction_percent']:.1f}%)")

    # 5. TRICERATOPS Bayesian Validation
    val_res = run_triceratops_validation(
        tic_id=tic,
        sectors=[target["sector"]],
        period=target["period_days"],
        depth=target["depth_ppm"],
        duration_days=target["duration_hours"] / 24.0,
        rp_rs=target["rp_rs"],
        impact_parameter_b=target["b"],
        target_tmag=tmag_target,
        centroid_offset_arcsec=target["centroid_offset_arcsec"],
        centroid_sigma_arcsec=target["centroid_sigma_arcsec"],
        neighbors=vetted_neighbors
    )

    fpp = val_res["fpp"]
    nfpp = val_res["nfpp"]
    is_valid = val_res["validated"]
    probs = val_res["probabilities"]

    print(f"[*] TRICERATOPS Validation Results:")
    print(f"  • P(TP)  = {probs['TP']*100:.2f}% | P(PTP) = {probs['PTP']*100:.2f}%")
    print(f"  • P(EB)  = {probs['EB']*100:.2f}% | P(EBx2P) = {probs['EBx2P']*100:.2f}%")
    print(f"  • P(HEB) = {probs['HEB']*100:.2f}% | P(BEB) = {probs['BEB']*100:.2f}%")
    print(f"  • FPP    = {fpp*100:.4f}% (Threshold < 1.0%)")
    print(f"  • NFPP   = {nfpp*100:.4f}% (Threshold < 0.1%)")
    print(f"  • Status : {'VALIDATED EXOPLANET' if is_valid else 'FALSE POSITIVE / REJECTED'}")

    # 6. Generate Figures
    target_out_dir = os.path.join(output_base_dir, f"TIC_{tic}")
    os.makedirs(target_out_dir, exist_ok=True)

    gaia_plot_path = os.path.join(target_out_dir, f"gaia_field_{target['name'].split()[0].lower()}.png")
    plot_gaia_field_overlay(
        target_name=name,
        tic_id=tic,
        target_ra=ra_target,
        target_dec=dec_target,
        target_tmag=tmag_target,
        neighbors=vetted_neighbors,
        delta_m_crit=delta_m_crit,
        output_path=gaia_plot_path
    )

    prob_plot_path = os.path.join(target_out_dir, f"triceratops_prob_{target['name'].split()[0].lower()}.png")
    plot_triceratops_probabilities(
        target_name=name,
        tic_id=tic,
        res=val_res,
        output_path=prob_plot_path
    )

    if wasp77_details:
        wasp_plot_path = os.path.join(target_out_dir, "wasp77b_dedilution.png")
        plot_wasp77b_dedilution(wasp77_details, wasp_plot_path)

    elapsed = time.time() - t_start

    record = {
        "target": target,
        "runtime_seconds": elapsed,
        "target_coordinates": {"ra": ra_target, "dec": dec_target, "tmag": tmag_target},
        "gaia_vetting": {
            "neighbors_count": len(neighbors),
            "ruled_out_count": ruled_out_count,
            "delta_m_crit": delta_m_crit,
            "dilution_factor": d_factor,
            "contamination_ratio": dilution_info["contamination_ratio"]
        },
        "wasp77_dedilution": wasp77_details,
        "triceratops_validation": val_res,
        "figures": {
            "gaia_field": os.path.abspath(gaia_plot_path),
            "triceratops_probabilities": os.path.abspath(prob_plot_path)
        }
    }
    return record


def main():
    output_dir = "results/phase4_vetting"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ASTRO-EXO: FASE 4 - TRIAGEM GAIA DR3 & VALIDAÇÃO ESTATÍSTICA TRICERATOPS")
    print(f"Alvos Analisados: {len(ALL_9_TARGETS)} sistemas reais do TESS")
    print("=" * 80)

    results = []
    for tgt in ALL_9_TARGETS:
        rec = process_phase4_target(tgt, output_dir)
        results.append(rec)

    # Export consolidated summary JSON
    summary_path = os.path.join(output_dir, "phase4_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"FASE 4 CONCLUÍDA COM SUCESSO! Relatório consolidado salvo em:")
    print(f"  -> {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
