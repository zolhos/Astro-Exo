"""
Phase 3 Execution Pipeline: Bayesian Transit Parameter Estimation on Real TESS Light Curves.
Ingests detrended PDCSAP photometry for confirmed exoplanet targets, executes Nelder-Mead MAP
optimization, runs affine-invariant MCMC (emcee) sampling, derives physical properties
(Rp/Rs, a/Rs, b, inclination, stellar density, limb darkening), evaluates convergence (R-hat, tau),
and outputs publication-quality corner plots, transit fit figures, and a JSON catalog.
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

from astro_exo.ingestion.fits_reader import read_tess_lightcurve
from astro_exo.models.emcee_sampler import EmceeTransitFitter
import wotan


PHASE3_TARGETS = [
    {
        "name": "WASP-126b",
        "tic_id": 25155310,
        "sector": 27,
        "lc_file": "data/photometry/TIC_25155310/tess_tic25155310_s0027_lc.fits",
        "period_days": 3.28879,
        "t0_btjd": 2037.8986,
        "duration_hours": 2.68,
        "literature_rp_rs": 0.078,
        "literature_depth_ppm": 6100.0,
    },
    {
        "name": "WASP-46b (TOI-101.01)",
        "tic_id": 231663901,
        "sector": 27,
        "lc_file": "data/photometry/TIC_231663901/tess_tic231663901_s0027_lc.fits",
        "period_days": 1.43037,
        "t0_btjd": 1326.0091,
        "duration_hours": 1.62,
        "literature_rp_rs": 0.134,
        "literature_depth_ppm": 18000.0,
    },
    {
        "name": "WASP-62b (TOI-102.01)",
        "tic_id": 149603524,
        "sector": 27,
        "lc_file": "data/photometry/TIC_149603524/tess_tic149603524_s0027_lc.fits",
        "period_days": 4.411937,
        "t0_btjd": 3982.0655,
        "duration_hours": 3.63,
        "literature_rp_rs": 0.113,
        "literature_depth_ppm": 12800.0,
    },
    {
        "name": "WASP-95b (TOI-105.01)",
        "tic_id": 144065872,
        "sector": 28,
        "lc_file": "data/photometry/TIC_144065872/tess_tic144065872_s0028_lc.fits",
        "period_days": 2.18467,
        "t0_btjd": 1326.3312,
        "duration_hours": 2.45,
        "literature_rp_rs": 0.103,
        "literature_depth_ppm": 10600.0,
    },
]


def process_phase3_target(target: Dict[str, Any], output_base_dir: str) -> Dict[str, Any]:
    tic = target["tic_id"]
    name = target["name"]
    lc_file = target["lc_file"]
    period = target["period_days"]
    t0 = target["t0_btjd"]
    dur_days = target["duration_hours"] / 24.0

    print(f"\n{'='*70}")
    print(f"[PHASE 3] Ingesting & Modeling: {name} (TIC {tic}, Sector {target['sector']})")
    print(f"{'='*70}")

    t_start = time.time()

    # 1. Read calibrated FITS light curve
    lc = read_tess_lightcurve(lc_file)
    time_raw = lc["time"]
    flux_raw = lc["flux"]
    err_raw = lc["flux_err"]

    print(f"[*] Read {len(time_raw):,} cadences from {os.path.basename(lc_file)}")

    # 2. Robust detrending with wotan (biweight)
    print(f"[*] Detrending light curve using wotan biweight (window=0.75d)...")
    flat_flux = wotan.flatten(
        time_raw, flux_raw,
        method="biweight",
        window_length=0.75,
        edge_cutoff=0.5,
        break_tolerance=0.5
    )

    # 3. Windowing: Slice data to transit windows (+/- 2.5 * duration)
    valid_pts = np.isfinite(flat_flux) & np.isfinite(time_raw) & np.isfinite(err_raw) & (err_raw > 0)
    time_raw = time_raw[valid_pts]
    flat_flux = flat_flux[valid_pts]
    err_raw = err_raw[valid_pts]

    from astro_exo.models.transforms import align_t0_to_dataset
    t0_local = align_t0_to_dataset(t0, period, time_raw)
    phase = (time_raw - t0_local + 0.5 * period) % period - 0.5 * period
    transit_mask = np.abs(phase) < (2.5 * dur_days)
    t_fit = time_raw[transit_mask]
    f_fit = flat_flux[transit_mask]
    e_fit = err_raw[transit_mask]

    n_events = int(round((np.nanmax(t_fit) - np.nanmin(t_fit)) / period)) + 1
    print(f"[*] Reference epoch aligned to local sector: T0 = {t0_local:.4f} BTJD")
    print(f"[*] Transit windowing: isolated {len(t_fit):,} clean points across ~{n_events} transit events")

    # 4. Initialize MCMC Transit Fitter
    fitter = EmceeTransitFitter(
        time=t_fit,
        flux=f_fit,
        flux_err=e_fit,
        period=period,
        t0_expected=t0
    )

    # 5. Nelder-Mead Maximum A Posteriori (MAP) Optimization
    print(f"[*] Finding Maximum A Posteriori (MAP) starting parameters...")
    theta_map = fitter.fit_map()
    print(f"    MAP: T0={theta_map[0]:.4f}, Rp/Rs={theta_map[1]:.4f}, a/Rs={theta_map[2]:.2f}, b={theta_map[3]:.2f}")

    # 6. Affine-invariant MCMC Ensemble Sampling (emcee)
    nwalkers = 32
    nburn = 350
    nprod = 1000
    print(f"[*] Running MCMC: {nwalkers} walkers x ({nburn} burn-in + {nprod} production) steps...")
    t_mcmc_start = time.time()
    samples = fitter.run_mcmc(nwalkers=nwalkers, nburn=nburn, nprod=nprod, seed=42)
    mcmc_duration = time.time() - t_mcmc_start
    print(f"[*] MCMC sampling finished in {mcmc_duration:.1f}s ({len(samples):,} flat posterior draws)")

    # 7. Posterior Summary & Diagnostics
    summary = fitter.get_summary()
    diagnostics = fitter.get_diagnostics()

    rp_med = summary["rp"]["median"]
    rp_e1s = summary["rp"]["err_plus_1s"]
    depth_med = summary["depth_ppm"]["median"]
    depth_e1s = summary["depth_ppm"]["err_plus_1s"]
    inc_med = summary["inc_deg"]["median"]
    inc_e1s = summary["inc_deg"]["err_plus_1s"]
    dur_med = summary["duration_hours"]["median"]
    rho_med = summary["rho_star_g_cm3"]["median"]

    rhat_vals = [v for v in diagnostics["r_hat"].values() if np.isfinite(v)]
    rhat_max = max(rhat_vals) if rhat_vals else 1.0

    print(f"\n[RESULTS] Posterior Median & 68.3% Credible Intervals (1-sigma):")
    print(f"  • Radius Ratio (Rp/Rs):  {rp_med:.4f} +{rp_e1s:.4f} / -{summary['rp']['err_minus_1s']:.4f} (Lit: {target['literature_rp_rs']:.4f})")
    print(f"  • Transit Depth:         {depth_med:.1f} +{depth_e1s:.1f} / -{summary['depth_ppm']['err_minus_1s']:.1f} ppm (Lit: {target['literature_depth_ppm']:.1f} ppm)")
    print(f"  • Semi-major axis a/Rs:  {summary['a_rs']['median']:.2f} +{summary['a_rs']['err_plus_1s']:.2f} / -{summary['a_rs']['err_minus_1s']:.2f}")
    print(f"  • Impact Parameter b:    {summary['b']['median']:.3f} +{summary['b']['err_plus_1s']:.3f} / -{summary['b']['err_minus_1s']:.3f}")
    print(f"  • Inclination:           {inc_med:.2f}° +{inc_e1s:.2f}° / -{summary['inc_deg']['err_minus_1s']:.2f}°")
    print(f"  • Duration T_14:         {dur_med:.2f} h +{summary['duration_hours']['err_plus_1s']:.2f} / -{summary['duration_hours']['err_minus_1s']:.2f} h")
    print(f"  • Stellar Density rho_*: {rho_med:.3f} g/cm^3")
    print(f"  • Limb Darkening (u1,u2):({summary['u1']['median']:.3f}, {summary['u2']['median']:.3f})")
    print(f"  • Convergence:           Gelman-Rubin R-hat max = {rhat_max:.3f}, Acceptance Rate = {diagnostics['acceptance_fraction_mean']*100:.1f}%")

    # 8. Plot Corner & Transit Fit
    target_out_dir = os.path.join(output_base_dir, f"TIC_{tic}")
    os.makedirs(target_out_dir, exist_ok=True)

    corner_path = os.path.join(target_out_dir, f"corner_{target['name'].split()[0].lower()}.png")
    fit_path = os.path.join(target_out_dir, f"fit_{target['name'].split()[0].lower()}.png")

    print(f"[*] Generating corner plot -> {corner_path}...")
    fitter.plot_corner(corner_path, title=f"{name} (TIC {tic}) - Bayesian Posteriors")

    print(f"[*] Generating transit fit with 1-sigma & 3-sigma envelopes -> {fit_path}...")
    fitter.plot_fit(
        fit_path,
        title=f"{name} (P = {period:.4f} d) - TESS Sector {target['sector']} Transit Light Curve",
        n_bins=45
    )

    total_time = time.time() - t_start
    print(f"[✓] Completed {name} in {total_time:.1f}s")

    record = {
        "target": target,
        "runtime_seconds": total_time,
        "mcmc_duration_seconds": mcmc_duration,
        "posterior_summary": summary,
        "diagnostics": diagnostics,
        "figures": {
            "corner_plot": os.path.abspath(corner_path),
            "transit_fit": os.path.abspath(fit_path)
        }
    }
    return record


def main():
    output_dir = "results/phase3_mcmc"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ASTRO-EXO: FASE 3 - INFERÊNCIA BAYESIANA & MODELAGEM MCMC EM DADOS REAIS TESS")
    print(f"Processador: Apple M2 (ARM64) • Alvos: {len(PHASE3_TARGETS)} exoplanetas confirmados")
    print("=" * 80)

    results = []
    for tgt in PHASE3_TARGETS:
        rec = process_phase3_target(tgt, output_dir)
        results.append(rec)

    # Save consolidated summary
    summary_path = os.path.join(output_dir, "phase3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"FASE 3 CONCLUÍDA COM SUCESSO! Relatório consolidado salvo em:")
    print(f"  -> {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
