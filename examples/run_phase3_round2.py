"""
Phase 3 Execution Pipeline - Round 2:
Bayesian Transit MCMC Parameter Estimation on 5 Real TESS Targets:
1. HATS-3b (TOI-103.01) - Confirmed hot Jupiter (Sector 1)
2. TOI-1050.01 - Deep candidate (Sector 38)
3. TOI-1009.01 - Shallow small-planet candidate (~1700 ppm, Sector 34)
4. WASP-77b (TOI-1049.01) - Known binary blend (Sector 39)
5. TOI-1019.01 - Astrobiological False Positive NEB (Sector 35)
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
from astro_exo.models.transforms import align_t0_to_dataset
import wotan


ROUND2_TARGETS = [
    {
        "name": "HATS-3b (TOI-103.01)",
        "tic_id": 336732616,
        "sector": 1,
        "lc_file": "data/photometry/TIC_336732616/tess_tic336732616_s0001_lc.fits",
        "period_days": 3.547854,
        "t0_btjd": 1327.2526,
        "duration_hours": 3.49,
        "catalog_depth_ppm": 10424.4,
        "classification": "CONFIRMED_PLANET"
    },
    {
        "name": "TOI-1050.01",
        "tic_id": 66818296,
        "sector": 38,
        "lc_file": "data/photometry/TIC_66818296/tess_tic66818296_s0038_lc.fits",
        "period_days": 3.735484,
        "t0_btjd": 2355.5456,
        "duration_hours": 4.14,
        "catalog_depth_ppm": 17030.0,
        "classification": "PLANET_CANDIDATE"
    },
    {
        "name": "TOI-1009.01",
        "tic_id": 107782586,
        "sector": 34,
        "lc_file": "data/photometry/TIC_107782586/tess_tic107782586_s0034_lc.fits",
        "period_days": 1.960028,
        "t0_btjd": 2229.2309,
        "duration_hours": 2.01,
        "catalog_depth_ppm": 1707.6,
        "classification": "SHALLOW_CANDIDATE"
    },
    {
        "name": "WASP-77b (TOI-1049.01)",
        "tic_id": 16288184,
        "sector": 39,
        "lc_file": "data/photometry/TIC_16288184/tess_tic16288184_s0039_lc.fits",
        "period_days": 2.180533,
        "t0_btjd": 1626.8015,
        "duration_hours": 2.94,
        "catalog_depth_ppm": 16399.9,
        "classification": "STELLAR_BLEND_BINARY"
    },
    {
        "name": "TOI-1019.01",
        "tic_id": 341420329,
        "sector": 35,
        "lc_file": "data/photometry/TIC_341420329/tess_tic341420329_s0035_lc.fits",
        "period_days": 5.234101,
        "t0_btjd": 2232.4469,
        "duration_hours": 3.71,
        "catalog_depth_ppm": 20796.9,
        "classification": "FALSE_POSITIVE_NEB"
    }
]


def process_round2_target(target: Dict[str, Any], output_base_dir: str) -> Dict[str, Any]:
    tic = target["tic_id"]
    name = target["name"]
    lc_file = target["lc_file"]
    period = target["period_days"]
    t0 = target["t0_btjd"]
    dur_days = target["duration_hours"] / 24.0

    print(f"\n{'='*75}")
    print(f"[ROUND 2] Processing: {name} (TIC {tic}, Sector {target['sector']})")
    print(f"Classification Context: {target['classification']}")
    print(f"{'='*75}")

    t_start = time.time()

    # 1. Read calibrated FITS light curve
    lc = read_tess_lightcurve(lc_file)
    time_raw = lc["time"]
    flux_raw = lc["flux"]
    err_raw = lc["flux_err"]

    print(f"[*] Loaded {len(time_raw):,} raw cadences from {os.path.basename(lc_file)}")

    # 2. Robust detrending with wotan (biweight)
    print(f"[*] Detrending using wotan biweight filter (window=0.75d)...")
    flat_flux = wotan.flatten(
        time_raw, flux_raw,
        method="biweight",
        window_length=0.75,
        edge_cutoff=0.5,
        break_tolerance=0.5
    )

    # 3. Clean non-finite data points
    valid_pts = np.isfinite(flat_flux) & np.isfinite(time_raw) & np.isfinite(err_raw) & (err_raw > 0)
    time_clean = time_raw[valid_pts]
    flat_clean = flat_flux[valid_pts]
    err_clean = err_raw[valid_pts]

    # 4. Reference epoch alignment & transit window isolation (+/- 2.5 * duration)
    t0_local = align_t0_to_dataset(t0, period, time_clean)
    phase = (time_clean - t0_local + 0.5 * period) % period - 0.5 * period
    transit_mask = np.abs(phase) < (2.5 * dur_days)

    t_fit = time_clean[transit_mask]
    f_fit = flat_clean[transit_mask]
    e_fit = err_clean[transit_mask]

    n_events = int(round((np.nanmax(t_fit) - np.nanmin(t_fit)) / period)) + 1
    print(f"[*] Local Sector Epoch: T0 = {t0_local:.4f} BTJD")
    print(f"[*] Windowing: isolated {len(t_fit):,} transit cadences across ~{n_events} events")

    # 5. Initialize EmceeTransitFitter
    fitter = EmceeTransitFitter(
        time=t_fit,
        flux=f_fit,
        flux_err=e_fit,
        period=period,
        t0_expected=t0_local
    )

    # 6. Nelder-Mead Maximum A Posteriori (MAP) Optimization
    print(f"[*] Finding Maximum A Posteriori (MAP) starting estimate...")
    theta_map = fitter.fit_map()
    print(f"    MAP: T0={theta_map[0]:.4f}, Rp/Rs={theta_map[1]:.4f}, a/Rs={theta_map[2]:.2f}, b={theta_map[3]:.2f}")

    # 7. Affine-invariant MCMC Ensemble Sampling (emcee)
    nwalkers = 32
    nburn = 350
    nprod = 1000
    print(f"[*] Running MCMC: {nwalkers} walkers x ({nburn} burn-in + {nprod} production) steps...")
    t_mcmc_start = time.time()
    samples = fitter.run_mcmc(nwalkers=nwalkers, nburn=nburn, nprod=nprod, seed=42)
    mcmc_duration = time.time() - t_mcmc_start
    print(f"[*] MCMC sampling finished in {mcmc_duration:.1f}s ({len(samples):,} flat draws)")

    # 8. Posterior Summary & Diagnostics
    summary = fitter.get_summary()
    diagnostics = fitter.get_diagnostics()

    rp_med = summary["rp"]["median"]
    depth_med = summary["depth_ppm"]["median"]
    inc_med = summary["inc_deg"]["median"]
    dur_med = summary["duration_hours"]["median"]
    rho_med = summary["rho_star_g_cm3"]["median"]

    rhat_vals = [v for v in diagnostics["r_hat"].values() if np.isfinite(v)]
    rhat_max = max(rhat_vals) if rhat_vals else 1.0

    print(f"\n[RESULTS] Posterior Median & 68.3% Credible Intervals (1-sigma):")
    print(f"  • Radius Ratio (Rp/Rs):  {rp_med:.4f} +{summary['rp']['err_plus_1s']:.4f} / -{summary['rp']['err_minus_1s']:.4f}")
    print(f"  • Transit Depth:         {depth_med:.1f} +{summary['depth_ppm']['err_plus_1s']:.1f} / -{summary['depth_ppm']['err_minus_1s']:.1f} ppm (Cat: {target['catalog_depth_ppm']:.1f} ppm)")
    print(f"  • Semi-major axis a/Rs:  {summary['a_rs']['median']:.2f} +{summary['a_rs']['err_plus_1s']:.2f} / -{summary['a_rs']['err_minus_1s']:.2f}")
    print(f"  • Impact Parameter b:    {summary['b']['median']:.3f} +{summary['b']['err_plus_1s']:.3f} / -{summary['b']['err_minus_1s']:.3f}")
    print(f"  • Inclination:           {inc_med:.2f}° +{summary['inc_deg']['err_plus_1s']:.2f}° / -{summary['inc_deg']['err_minus_1s']:.2f}°")
    print(f"  • Duration T_14:         {dur_med:.2f} h +{summary['duration_hours']['err_plus_1s']:.2f} / -{summary['duration_hours']['err_minus_1s']:.2f} h")
    print(f"  • Stellar Density rho_*: {rho_med:.3f} g/cm^3")
    print(f"  • Limb Darkening (u1,u2):({summary['u1']['median']:.3f}, {summary['u2']['median']:.3f})")
    print(f"  • Convergence:           Gelman-Rubin R-hat max = {rhat_max:.3f}, Acceptance Rate = {diagnostics['acceptance_fraction_mean']*100:.1f}%")

    # 9. Plot Figures
    slug = target["name"].split()[0].replace("-", "_").lower()
    target_out_dir = os.path.join(output_base_dir, f"TIC_{tic}")
    os.makedirs(target_out_dir, exist_ok=True)

    corner_path = os.path.join(target_out_dir, f"corner_{slug}.png")
    fit_path = os.path.join(target_out_dir, f"fit_{slug}.png")

    print(f"[*] Generating corner plot -> {corner_path}...")
    fitter.plot_corner(corner_path, title=f"{name} (TIC {tic}) - Bayesian Posteriors")

    print(f"[*] Generating transit fit plot with 1-sigma & 3-sigma envelopes -> {fit_path}...")
    fitter.plot_fit(
        fit_path,
        title=f"{name} (P = {period:.4f} d) - TESS Sector {target['sector']} Transit Light Curve",
        n_bins=45
    )

    total_time = time.time() - t_start
    print(f"[✓] Target completed in {total_time:.1f}s")

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
    output_dir = "results/phase3_mcmc_round2"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ASTRO-EXO: FASE 3 (RODADA 2) - INFERÊNCIA BAYESIANA EM 5 NOVOS ALVOS REAIS TESS")
    print("Processador: Apple M2 (ARM64) • Amostras: 5 sistemas astrofísicos heterogêneos")
    print("=" * 80)

    results = []
    for tgt in ROUND2_TARGETS:
        rec = process_round2_target(tgt, output_dir)
        results.append(rec)

    # Save summary
    summary_path = os.path.join(output_dir, "round2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RODADA 2 CONCLUÍDA COM SUCESSO! Relatório consolidado salvo em:")
    print(f"  -> {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
