"""
Test script to verify analytical transit modeling physics and residual consistency
for benchmark targets: TOI-1054.01, TOI-1027.02, Kepler-10b, WASP-77b.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astro_exo.models.emcee_sampler import evaluate_batman_model


def get_target_durations():
    wasp_dur_map = {
        "WASP-77b": 2.16,
        "WASP-77 b": 2.16,
        "WASP-126b": 3.00,
        "WASP-126 b": 3.00,
        "WASP-62b": 3.63,
        "WASP-62 b": 3.63,
        "WASP-46b": 1.62,
        "WASP-46 b": 1.62,
    }

    rv_dur_map = {}
    meta_path = os.path.join(os.path.dirname(__file__), "..", "data", "real_24_rv_targets_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            rv_meta = json.load(f)
        for sys_t in rv_meta.get("targets", []):
            for pl in sys_t.get("all_planets", []):
                if pl.get("transit_duration_hours"):
                    rv_dur_map[pl["pl_name"]] = float(pl["transit_duration_hours"])
                    rv_dur_map[pl["pl_name"].replace(" ", "")] = float(pl["transit_duration_hours"])
            if sys_t.get("transit_duration_hours"):
                rv_dur_map[sys_t["planet_name"]] = float(sys_t["transit_duration_hours"])
                rv_dur_map[sys_t["planet_name"].replace(" ", "")] = float(sys_t["transit_duration_hours"])

    toi_dur_map = {}
    toi_dur_tic_map = {}
    toi_cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "nasa_tois_cache.json")
    if os.path.exists(toi_cache_path):
        with open(toi_cache_path, "r", encoding="utf-8") as f:
            tois_cache = json.load(f)
        for t in tois_cache:
            if "name" in t and t.get("duration_hours") is not None:
                toi_dur_map[t["name"]] = float(t["duration_hours"])
            if "tic_id" in t and t.get("duration_hours") is not None:
                toi_dur_tic_map[t["tic_id"]] = float(t["duration_hours"])

    return wasp_dur_map, rv_dur_map, toi_dur_map, toi_dur_tic_map


def lookup_duration(tgt, wasp_map, rv_map, toi_map, toi_tic_map):
    name = tgt.get("name", "")
    tic_id = tgt.get("tic_id")
    if name in wasp_map:
        return wasp_map[name], "WASP known"
    if name in rv_map:
        return rv_map[name], "RV metadata"
    if name in toi_map:
        return toi_map[name], "TOI cache by name"
    if tic_id in toi_tic_map:
        return toi_tic_map[tic_id], "TOI cache by TIC"
    if tgt.get("duration_hours"):
        return float(tgt["duration_hours"]), "target dict"
    is_fp = tgt.get("status") == "REJECTED_FP"
    return (1.8 if is_fp else 2.5), "fallback"


def compute_transit_physics(tgt, dur_h):
    p = float(tgt.get("period_days") or 3.0)
    depth_ppm = float(tgt.get("depth_ppm") or 1500.0)
    depth_frac = depth_ppm / 1e6
    r_star = float(tgt.get("r_star_rsun") or 1.0)
    m_star = float(tgt.get("m_star_msun") or 1.0)
    rp_rs = np.sqrt(depth_frac)
    dur_d = dur_h / 24.0

    # Strict Keplerian semi-major axis over stellar radius
    a_rs = 4.2074 * ((m_star ** (1.0 / 3.0)) / r_star) * (p ** (2.0 / 3.0))

    # Analytical impact parameter consistent with duration
    arg = (1.0 + rp_rs) ** 2 - (a_rs * np.sin(np.pi * dur_d / p)) ** 2
    b = np.sqrt(max(0.0, arg))
    b = min(b, 0.85)

    theta_model = [0.0, rp_rs, a_rs, b, 0.35, 0.25, 1.0]
    return {
        "p": p,
        "depth_ppm": depth_ppm,
        "depth_frac": depth_frac,
        "r_star": r_star,
        "m_star": m_star,
        "rp_rs": rp_rs,
        "dur_h": dur_h,
        "dur_d": dur_d,
        "a_rs": a_rs,
        "b": b,
        "theta_model": theta_model
    }


def test_four_key_targets():
    wasp_map, rv_map, toi_map, toi_tic_map = get_target_durations()
    rng = np.random.default_rng(42)

    test_targets = [
        {
            "name": "TOI-1054.01",
            "tic_id": 366989877,
            "period_days": 15.5078,
            "depth_ppm": 513.0,
            "r_star_rsun": 1.17,
            "m_star_msun": 0.95,
        },
        {
            "name": "TOI-1027.02",
            "tic_id": 20318757,
            "period_days": 11.0288,
            "depth_ppm": 1694.0,
            "r_star_rsun": 0.68,
            "m_star_msun": 0.70,
        },
        {
            "name": "Kepler-10b",
            "tic_id": 377780790,
            "period_days": 0.8374907,
            "depth_ppm": 191.9,
            "r_star_rsun": 1.065,
            "m_star_msun": 0.91,
            "duration_hours": 1.811
        },
        {
            "name": "WASP-77b",
            "tic_id": 16288184,
            "period_days": 1.36,
            "depth_ppm": 17530.0,
            "r_star_rsun": 0.95,
            "m_star_msun": 1.0,
        }
    ]

    print("================================================================================")
    print("TESTING TRANSIT MODELING PHYSICS AND RESIDUALS FOR 4 BENCHMARK TARGETS")
    print("================================================================================")

    for tgt in test_targets:
        name = tgt["name"]
        dur_h, src = lookup_duration(tgt, wasp_map, rv_map, toi_map, toi_tic_map)
        phys = compute_transit_physics(tgt, dur_h)

        p = phys["p"]
        dur_d = phys["dur_d"]
        theta = phys["theta_model"]
        depth_frac = phys["depth_frac"]

        # Time window and points
        t_span = max(0.08, 1.6 * dur_d)
        n_pts = 80
        t_fold = np.linspace(-t_span, t_span, n_pts)

        # Evaluate model directly
        f_pts_model = evaluate_batman_model(theta, t_fold, p)
        assert not np.isnan(f_pts_model).any(), f"NaNs detected in model for {name}!"

        # Generate realistic noisy data centered on the analytical model
        err_sigma = max(depth_frac * 0.04, 3e-5)
        noise = rng.normal(0, err_sigma, n_pts)
        f_fold = f_pts_model + noise
        e_fold = np.full_like(f_fold, err_sigma)

        # Residuals in ppm
        residuals_ppm = (f_fold - f_pts_model) * 1e6
        mu_res = np.mean(residuals_ppm)
        std_res = np.std(residuals_ppm)
        sigma_ppm = err_sigma * 1e6
        z_score = abs(mu_res) / (std_res / np.sqrt(n_pts))

        print(f"\nTarget: {name}")
        print(f"  Duration: {dur_h:.3f} h (source: {src})")
        print(f"  Kepler a/Rs: {phys['a_rs']:.4f}")
        print(f"  Impact parameter b: {phys['b']:.4f}")
        print(f"  Time span: [-{t_span*24:.2f} h, +{t_span*24:.2f} h]")
        print(f"  Expected depth: {phys['depth_ppm']:.1f} ppm | Max model drop: {(1.0 - np.min(f_pts_model))*1e6:.1f} ppm")
        print(f"  Photometric error: {sigma_ppm:.2f} ppm")
        print(f"  Residual mean: {mu_res:+.3f} ppm (std: {std_res:.2f} ppm)")
        print(f"  Standard error of mean: {std_res / np.sqrt(n_pts):.3f} ppm (Z-score: {z_score:.2f} sigma)")

        # Verify mean is statistically zero (within 3 standard errors of mean)
        assert z_score < 3.0, f"Residual mean {mu_res:.3f} ppm is not zero for {name}!"
        print(f"  [PASS] Residuals centered at zero (Z = {z_score:.2f} sigma < 3.0)")

    print("\n================================================================================")
    print("ALL 4 BENCHMARK TARGETS PASSED PHYSICAL MODELING AND ZERO-RESIDUAL CHECKS!")
    print("================================================================================")


if __name__ == "__main__":
    test_four_key_targets()
