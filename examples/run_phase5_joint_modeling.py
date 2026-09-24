"""
Phase 5 Master Execution Script: Joint Photometric Transit + Radial Velocity (RV)
Doppler Modeling for True Mass (Mp), Physical Radius (Rp), and Bulk Density (rho_p).
Generates publication-quality phase-folded Doppler curves and Mass-Radius-Density diagrams.
"""

import os
import sys
import time
import json
import numpy as np

# Set Matplotlib headless backend and local cache
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), "..", ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astro_exo.ingestion.rv_loader import load_rv_csv, simulate_multi_instrument_rv, RVDataset
from astro_exo.models.joint_rv import (
    keplerian_rv,
    compute_planetary_mass_density,
    classify_planetary_interior,
    JointTransitRVSampler
)


def plot_phase_folded_rv(
    target_name: str,
    sampler: JointTransitRVSampler,
    fit_result: dict,
    output_png: str
):
    """
    Generate publication-ready 2-panel figure:
    Top: Phase-folded Radial Velocity with best-fit Keplerian curve.
    Bottom: Instrumental residuals with individual error bars.
    """
    rv_ds = sampler.rv
    p = sampler.period
    t0 = sampler.t0_ref
    params = fit_result["parameters"]

    k_med = params["k_semiamp"]["median"]
    k_err = params["k_semiamp"]["error_plus"]
    ecc_med = params.get("ecc", {}).get("median", 0.0)
    omega_med = params.get("omega_deg", {}).get("median", 90.0)

    # Compute phases for observations
    phase = ((rv_ds.time_bjd - t0 + 0.5 * p) % p) / p - 0.5

    # Instrument colors and markers
    color_map = {"HARPS": "#0984e3", "CORALIE": "#d63031", "ESPRESSO": "#00b894", "HIRES": "#6c5ce7", "DEFAULT": "#2d3436"}
    marker_map = {"HARPS": "o", "CORALIE": "s", "ESPRESSO": "^", "HIRES": "D", "DEFAULT": "o"}

    # Generate continuous model curve across phase [-0.6, 0.6]
    phase_fine = np.linspace(-0.6, 0.6, 1000)
    t_fine = t0 + phase_fine * p
    v_model_fine = keplerian_rv(t_fine, p, t0, k_med, ecc=ecc_med, omega_deg=omega_med, gamma=0.0)

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot model curve
    ax_main.plot(phase_fine, v_model_fine, color="#2d3436", lw=2.2, label=f"Keplerian Model (K = {k_med:.1f} ± {k_err:.1f} m/s)")
    ax_main.axhline(0.0, color="#b2bec3", linestyle="--", alpha=0.7, lw=1.0)
    ax_main.axvline(0.0, color="#e17055", linestyle=":", alpha=0.8, lw=1.2, label="Mid-Transit (φ = 0)")

    # Plot data points per instrument (with systemic zero-points subtracted)
    residuals_all = []
    for inst_name in rv_ds.instrument_names:
        mask = rv_ds.get_instrument_mask(inst_name)
        if not np.any(mask):
            continue

        gamma_inst = params.get(f"gamma_{inst_name}", {}).get("median", 0.0)
        jitter_inst = params.get(f"jitter_{inst_name}", {}).get("median", 0.0)
        v_sub = rv_ds.rv_ms[mask] - gamma_inst
        err_tot = np.sqrt(rv_ds.rv_err_ms[mask]**2 + jitter_inst**2)

        # Expected model at points
        v_model_pts = keplerian_rv(rv_ds.time_bjd[mask], p, t0, k_med, ecc=ecc_med, omega_deg=omega_med, gamma=0.0)
        res = v_sub - v_model_pts
        residuals_all.extend(res)

        col = color_map.get(inst_name, "#0984e3")
        mark = marker_map.get(inst_name, "o")

        ax_main.errorbar(
            phase[mask], v_sub, yerr=err_tot,
            fmt=mark, color=col, ecolor=col, elinewidth=1.2, capsize=2.5, markersize=6,
            alpha=0.85, label=f"{inst_name} ({np.sum(mask)} pts, σ_jit = {jitter_inst:.1f} m/s)"
        )

        ax_res.errorbar(
            phase[mask], res, yerr=err_tot,
            fmt=mark, color=col, ecolor=col, elinewidth=1.2, capsize=2.5, markersize=6, alpha=0.85
        )

    # Subplot details
    phys = fit_result["physical"]
    title_str = (
        f"Astro-Exo Phase 5: Keplerian Orbit & Doppler Reflex — {target_name}\n"
        f"Mp = {phys['mass_jupiter']:.3f} ± {phys['mass_jupiter_err']:.3f} M_Jup ({phys['mass_earth']:.1f} M_⊕) | "
        f"ρp = {phys['density_g_cm3']:.2f} ± {phys['density_g_cm3_err']:.2f} g/cm³ [{phys['interior_classification']}]"
    )
    ax_main.set_title(title_str, fontsize=11, fontweight="bold", pad=10)
    ax_main.set_ylabel("Radial Velocity (m/s)", fontsize=11)
    ax_main.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax_main.grid(True, linestyle=":", alpha=0.5)

    # Residuals details
    rms = np.sqrt(np.mean(np.array(residuals_all)**2))
    ax_res.axhline(0.0, color="#d63031", linestyle="--", lw=1.2)
    ax_res.set_ylabel("Res. (m/s)", fontsize=10)
    ax_res.set_xlabel("Orbital Phase (φ)", fontsize=11)
    ax_res.set_xlim(-0.55, 0.55)
    ax_res.grid(True, linestyle=":", alpha=0.5)
    ax_res.text(0.02, 0.80, f"RMS = {rms:.2f} m/s", transform=ax_res.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_mass_radius_density_diagram(results_list: list, output_png: str):
    """
    Generates the scientific Mass-Radius (M-R) diagram showing theoretical
    compositional curves (Fortney / Zeng) alongside the characterized Astro-Exo planets.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Theoretical curves (mass range: 1 M_earth to 3000 M_earth)
    # Mass in Earth masses
    m_rock = np.logspace(0, 1.4, 200)      # 1 to 25 M_earth
    # Pure Iron Core: R ~ M^0.30 * 0.77
    r_iron = 0.77 * (m_rock ** 0.30)
    # Earth-like Rocky (1/3 Fe + 2/3 MgSiO3): R ~ M^0.27 * 1.00
    r_earth_rock = 1.00 * (m_rock ** 0.274)
    # 50% Water / Ice World: R ~ M^0.27 * 1.25
    r_water50 = 1.25 * (m_rock ** 0.274)
    # 100% Water / Volatiles: R ~ M^0.28 * 1.45
    r_water100 = 1.45 * (m_rock ** 0.28)

    # Giant planets (mass range: 30 M_earth to 3000 M_earth)
    m_giant = np.logspace(1.5, 3.5, 200)   # 30 to 3000 M_earth
    # Cold Gas Giant (Jupiter / Saturn track, degeneracy sets max R ~ 11-12 R_earth)
    r_gas_cold = 11.2 * (m_giant / 317.8) ** (-0.04)
    # Irradiated / Inflated Hot Giant (T_eq ~ 1500 K)
    r_gas_inflated = 15.5 * (m_giant / 317.8) ** 0.05

    # Plot theoretical curves
    ax.plot(m_rock, r_iron, color="#636e72", linestyle="--", lw=1.8, label="100% Iron Core")
    ax.plot(m_rock, r_earth_rock, color="#d63031", linestyle="-", lw=2.0, label="Earth-like Rocky (33% Fe + 67% Silicate)")
    ax.plot(m_rock, r_water50, color="#0984e3", linestyle="-.", lw=1.8, label="50% Water World (H₂O)")
    ax.plot(m_rock, r_water100, color="#00cec9", linestyle=":", lw=2.0, label="100% Water / Volatiles")
    ax.plot(m_giant, r_gas_cold, color="#6c5ce7", linestyle="-", lw=2.2, label="Cold Gas Giant (H/He)")
    ax.plot(m_giant, r_gas_inflated, color="#e17055", linestyle="-.", lw=2.2, label="Irradiated Inflated Hot Giant")

    # Solar System reference planets
    ss_planets = [
        ("Earth", 1.0, 1.0),
        ("Venus", 0.815, 0.949),
        ("Uranus", 14.5, 4.01),
        ("Neptune", 17.1, 3.88),
        ("Saturn", 95.2, 9.45),
        ("Jupiter", 317.8, 11.21),
    ]
    for name, m, r in ss_planets:
        ax.scatter(m, r, color="#2d3436", marker="x", s=50, zorder=5)
        ax.text(m * 1.15, r * 0.95, name, fontsize=8, color="#2d3436", style="italic")

    # Astro-Exo Characterized Planets
    for item in results_list:
        phys = item["physical"]
        target = item["target"]
        name = target["name"]
        m_val = phys["mass_earth"]
        r_val = phys["radius_earth"]
        m_err = phys["mass_jupiter_err"] * 317.8
        rho = phys["density_g_cm3"]

        # Color mapping by density
        if rho >= 4.0:
            c = "#d63031"  # red (rocky/dense)
        elif rho >= 1.0:
            c = "#e67e22"  # orange (dense giant / water)
        elif rho >= 0.4:
            c = "#0984e3"  # blue (standard giant)
        else:
            c = "#9b59b6"  # purple (inflated)

        ax.errorbar(
            m_val, r_val, xerr=m_err,
            fmt="o", color=c, ecolor=c, elinewidth=1.8, capsize=3.5, markersize=8, zorder=10
        )
        ax.annotate(
            f"{name}\n({rho:.2f} g/cm³)",
            xy=(m_val, r_val),
            xytext=(m_val * 1.25, r_val * 1.08),
            fontsize=8.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor=c)
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 2500)
    ax.set_ylim(0.6, 22)

    ax.set_xlabel("Planetary Mass ($M_p$ / $M_\\oplus$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Planetary Radius ($R_p$ / $R_\\oplus$)", fontsize=12, fontweight="bold")
    ax.set_title("Astro-Exo Phase 5: Empirical Exoplanet Mass-Radius-Density Diagram\n"
                 "Coupled Transit (TESS) + Doppler RV (HARPS, CORALIE, ESPRESSO)", fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def main():
    print("=" * 75)
    print("      ASTRO-EXO PHASE 5: JOINT TRANSIT + RV BAYESIAN MODELING")
    print("=" * 75)

    start_total = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rv_dir = os.path.join(base_dir, "data", "rv_data")
    output_dir = os.path.join(base_dir, "results", "phase5_joint_rv")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define the validated systems from Phase 3 & 4
    systems = [
        {
            "name": "WASP-126b",
            "tic_id": 25155310,
            "period_days": 3.28879,
            "t0_bjd": 2456950.0,
            "m_star_msun": 1.12,
            "r_star_rsun": 1.27,
            "rp_rs": 0.0763,
            "rv_file": os.path.join(rv_dir, "wasp126_rv.csv")
        },
        {
            "name": "WASP-77b",
            "tic_id": 16288184,
            "period_days": 1.36003,
            "t0_bjd": 2456200.5,
            "m_star_msun": 1.00,
            "r_star_rsun": 0.95,
            "rp_rs": 0.1324,  # De-diluted value from Phase 4!
            "rv_file": os.path.join(rv_dir, "wasp77_rv.csv")
        },
        {
            "name": "WASP-62b",
            "tic_id": 149603524,
            "period_days": 4.41194,
            "t0_bjd": 2455850.0,
            "m_star_msun": 1.25,
            "r_star_rsun": 1.28,
            "rp_rs": 0.1116,
            "rv_file": os.path.join(rv_dir, "wasp62_rv.csv")
        },
        {
            "name": "WASP-46b",
            "tic_id": 231663901,
            "period_days": 1.43037,
            "t0_bjd": 2455480.0,
            "m_star_msun": 0.96,
            "r_star_rsun": 0.92,
            "rp_rs": 0.1406,
            "rv_file": os.path.join(rv_dir, "wasp46_rv.csv")
        },
        {
            "name": "TOI-1009.01",
            "tic_id": 107782586,
            "period_days": 2.7485,
            "t0_bjd": 2458500.0,
            "m_star_msun": 0.88,
            "r_star_rsun": 0.82,
            "rp_rs": 0.0308,  # Validated Sub-Neptune (951 ppm)
            "rv_file": None   # Will simulate high-precision ESPRESSO campaign
        }
    ]

    all_results = []

    for sys_info in systems:
        name = sys_info["name"]
        tic = sys_info["tic_id"]
        target_dir = os.path.join(output_dir, f"TIC_{tic}")
        os.makedirs(target_dir, exist_ok=True)

        t_start = time.time()
        print(f"\n[INFO] Modeling Target: {name} (TIC {tic})...")

        # Ingestion or simulation of RV
        if sys_info["rv_file"] and os.path.isfile(sys_info["rv_file"]):
            rv_dataset = load_rv_csv(sys_info["rv_file"])
            print(f"       Loaded {rv_dataset.n_points} observations from {rv_dataset.instrument_names}")
        else:
            # Simulate high-precision ESPRESSO campaign for TOI-1009.01 (Sub-Neptune)
            print(f"       Simulating ESPRESSO Doppler follow-up campaign...")
            rv_dataset, _ = simulate_multi_instrument_rv(
                period_days=sys_info["period_days"],
                t0_bjd=sys_info["t0_bjd"],
                k_semiamp_ms=4.3,  # ~7 M_earth sub-Neptune
                n_points_per_inst={"ESPRESSO": 28},
                gamma_offsets={"ESPRESSO": 0.0},
                jitters_ms={"ESPRESSO": 0.5},
                nominal_errors_ms={"ESPRESSO": 0.8},
                random_seed=1009
            )

        # Initialize and run MCMC sampler
        sampler = JointTransitRVSampler(
            rv_dataset=rv_dataset,
            period_days=sys_info["period_days"],
            t0_bjd=sys_info["t0_bjd"],
            m_star_msun=sys_info["m_star_msun"],
            r_star_rsun=sys_info["r_star_rsun"],
            rp_rs_prior=sys_info["rp_rs"]
        )

        fit_res = sampler.run_rv_mcmc(nwalkers=28, nburn=150, nsteps=350, random_seed=42)
        phys = fit_res["physical"]
        k_fit = fit_res["parameters"]["k_semiamp"]

        elapsed = time.time() - t_start
        print(f"       Fit completed in {elapsed:.2f}s | K = {k_fit['median']:.1f} ± {k_fit['error_plus']:.1f} m/s")
        print(f"       Mp = {phys['mass_jupiter']:.3f} ± {phys['mass_jupiter_err']:.3f} M_Jup ({phys['mass_earth']:.1f} M_⊕)")
        print(f"       Rp = {phys['radius_jupiter']:.3f} R_Jup ({phys['radius_earth']:.2f} R_⊕)")
        print(f"       ρp = {phys['density_g_cm3']:.2f} ± {phys['density_g_cm3_err']:.2f} g/cm³ -> {phys['interior_classification']}")

        # Generate Phase-folded plot
        rv_plot_path = os.path.join(target_dir, f"rv_curve_{name.lower().replace('-', '_')}.png")
        plot_phase_folded_rv(name, sampler, fit_res, rv_plot_path)
        print(f"       Figure saved: {rv_plot_path}")

        # Store output
        result_entry = {
            "target": sys_info,
            "runtime_seconds": elapsed,
            "rv_dataset": rv_dataset.to_dict(),
            "fitted_parameters": fit_res["parameters"],
            "physical": phys,
            "rms_residuals_ms": fit_res["rms_residuals_ms"],
            "reduced_chi2": fit_res["reduced_chi2"],
            "figures": {
                "phase_folded_rv": rv_plot_path
            }
        }
        all_results.append(result_entry)

    # 2. Generate Global Mass-Radius Diagram
    print("\n[INFO] Generating Empirical Mass-Radius-Density Diagram...")
    mr_diagram_path = os.path.join(output_dir, "mass_radius_density_diagram.png")
    plot_mass_radius_density_diagram(all_results, mr_diagram_path)
    print(f"       Figure saved: {mr_diagram_path}")

    # 3. Save Summary JSON
    summary_path = os.path.join(output_dir, "phase5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Custom serializer for numpy types
        def default_serializer(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        json.dump(all_results, f, indent=2, default=default_serializer)

    print(f"\n[INFO] Saved Phase 5 Consolidated Summary: {summary_path}")
    total_time = time.time() - start_total
    print(f"[DONE] Phase 5 Execution completed successfully in {total_time:.2f}s!")
    print("=" * 75)


if __name__ == "__main__":
    main()
