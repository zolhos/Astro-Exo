"""
Script to extract 22 real observational RV datasets from VizieR catalogs and NASA Exoplanet Archive parameters.
Outputs:
  - 22 RV CSV files in data/rv_data/
  - data/real_22_new_targets_metadata.json
"""

import os
import glob
import json
import pickle
import io
import numpy as np
import pandas as pd
from astropy.io.votable import parse

REPO_ROOT = "/Users/diegozolhos/Projects/Astro Exo Planets"
RV_DIR = os.path.join(REPO_ROOT, "data", "rv_data")
METADATA_OUT = os.path.join(REPO_ROOT, "data", "real_22_new_targets_metadata.json")
CACHE_DIR = "/Users/diegozolhos/.cache/astropy/astroquery/Vizier"
NASA_JSON_PATH = "/Users/diegozolhos/.gemini/antigravity/brain/b8b57836-a87d-4d35-9ed3-6ea6ec4a15d9/.system_generated/steps/106/content.md"

os.makedirs(RV_DIR, exist_ok=True)

# 1. Load NASA pscomppars data
with open(NASA_JSON_PATH, "r", encoding="utf-8") as f:
    text = f.read()
json_text = text[text.find('['):text.rfind(']')+1]
nasa_planets = json.loads(json_text)

nasa_by_host = {}
for p in nasa_planets:
    nasa_by_host.setdefault(p['hostname'], []).append(p)

print(f"Loaded {len(nasa_planets)} planets across {len(nasa_by_host)} host systems from NASA pscomppars.")

# 2. Extract 14 Bonomo et al. 2023 (HARPS-N) RV datasets
# Catalog: J/A+A/677/A33 Table 2
bonomo_pickle = os.path.join(CACHE_DIR, "5c535969264b7380f6141df648f714a64d1c5bb309c9b3e2e4db450d.pickle")
with open(bonomo_pickle, "rb") as f:
    resp = pickle.load(f)
vot = parse(io.BytesIO(resp.content))
tables = [t.to_table() for t in vot.iter_tables()]
bonomo_t3 = tables[3] # System, Time (BJD - 2450000), RV (m/s), e_RV (m/s)

bonomo_targets = [
    {
        "system_name": "Kepler-37",
        "primary_planet": "Kepler-37 d",
        "nasa_hostname": "Kepler-37",
        "csv_basename": "kepler37_rv.csv",
        "alt_names": ["kepler_37_rv.csv", "kepler-37_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "Kepler-68",
        "primary_planet": "Kepler-68 b",
        "nasa_hostname": "Kepler-68",
        "csv_basename": "kepler68_rv.csv",
        "alt_names": ["kepler_68_rv.csv", "kepler-68_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "Kepler-323",
        "primary_planet": "Kepler-323 b",
        "nasa_hostname": "Kepler-323",
        "csv_basename": "kepler323_rv.csv",
        "alt_names": ["kepler_323_rv.csv", "kepler-323_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "Kepler-1876",
        "primary_planet": "Kepler-1876 b",
        "nasa_hostname": "Kepler-1876",
        "csv_basename": "kepler1876_rv.csv",
        "alt_names": ["kepler_1876_rv.csv", "kepler-1876_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-12",
        "primary_planet": "K2-12 b",
        "nasa_hostname": "K2-12",
        "csv_basename": "k2_12_rv.csv",
        "alt_names": ["k2-12_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-79",
        "primary_planet": "K2-79 b",
        "nasa_hostname": "K2-79",
        "csv_basename": "k2_79_rv.csv",
        "alt_names": ["k2-79_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-96",
        "primary_planet": "HD 3167 b",
        "nasa_hostname": "HD 3167",
        "csv_basename": "k2_96_rv.csv",
        "alt_names": ["k2-96_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-106",
        "primary_planet": "EPIC 220674823 b",
        "nasa_hostname": "EPIC 220674823",
        "csv_basename": "k2_106_rv.csv",
        "alt_names": ["k2-106_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-135",
        "primary_planet": "GJ 9827 b",
        "nasa_hostname": "GJ 9827",
        "csv_basename": "k2_135_rv.csv",
        "alt_names": ["k2-135_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-167",
        "primary_planet": "K2-167 b",
        "nasa_hostname": "K2-167",
        "csv_basename": "k2_167_rv.csv",
        "alt_names": ["k2-167_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-262",
        "primary_planet": "Wolf 503 b",
        "nasa_hostname": "Wolf 503",
        "csv_basename": "k2_262_rv.csv",
        "alt_names": ["k2-262_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-263",
        "primary_planet": "K2-263 b",
        "nasa_hostname": "K2-263",
        "csv_basename": "k2_263_rv.csv",
        "alt_names": ["k2-263_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-312",
        "primary_planet": "HD 80653 b",
        "nasa_hostname": "HD 80653",
        "csv_basename": "k2_312_rv.csv",
        "alt_names": ["k2-312_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    },
    {
        "system_name": "K2-418",
        "primary_planet": "EPIC 229004835 b",
        "nasa_hostname": "EPIC 229004835",
        "csv_basename": "k2_418_rv.csv",
        "alt_names": ["k2-418_rv.csv"],
        "source_rv": "VizieR J/A+A/677/A33 Table 2 (Bonomo et al. 2023, HARPS-N)"
    }
]

datasets_summary = []

for tgt in bonomo_targets:
    sname = tgt["system_name"]
    mask = (bonomo_t3["System"] == sname)
    sub = bonomo_t3[mask]
    
    times = np.array(sub["Time"], dtype=np.float64) + 2450000.0
    rvs = np.array(sub["RV"], dtype=np.float64)
    errs = np.array(sub["e_RV"], dtype=np.float64)
    insts = np.array(["HARPS-N"] * len(times), dtype=str)
    
    # Sort chronologically
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    rvs = rvs[sort_idx]
    errs = errs[sort_idx]
    insts = insts[sort_idx]
    
    df = pd.DataFrame({
        "bjd": [f"{t:.6f}" for t in times],
        "rv_ms": [f"{v:.3f}" for v in rvs],
        "rv_err_ms": [f"{e:.3f}" for e in errs],
        "instrument": insts
    })
    
    csv_path = os.path.join(RV_DIR, tgt["csv_basename"])
    df.to_csv(csv_path, index=False)
    
    # Create symlinks
    for alt in tgt["alt_names"]:
        alt_path = os.path.join(RV_DIR, alt)
        if os.path.islink(alt_path) or os.path.exists(alt_path):
            os.remove(alt_path)
        os.symlink(tgt["csv_basename"], alt_path)
        
    datasets_summary.append({
        "system_name": sname,
        "csv_file": f"data/rv_data/{tgt['csv_basename']}",
        "n_obs": len(times),
        "bjd_min": float(times.min()),
        "bjd_max": float(times.max()),
        "baseline_days": float(times.max() - times.min()),
        "rv_err_median_ms": float(np.median(errs)),
        "instrument": "HARPS-N",
        "primary_planet": tgt["primary_planet"],
        "nasa_hostname": tgt["nasa_hostname"],
        "source_rv": tgt["source_rv"]
    })
    print(f"Bonomo {sname}: {len(times)} RV points written to {tgt['csv_basename']}")

# 3. Extract 8 WASP RV datasets from VizieR pickles
wasp_configs = [
    {
        "system_name": "WASP-8",
        "primary_planet": "WASP-8 b",
        "nasa_hostname": "WASP-8",
        "csv_basename": "wasp8_rv.csv",
        "alt_names": ["wasp-8_rv.csv", "wasp_8_rv.csv"],
        "pickle_name": "4f7797182da36d92f4fbeddfd7bd5ecd6464d679a2e9d919230193a0.pickle",
        "table_index": 0,
        "time_col": "JD",
        "time_offset": 0.0,
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": "Inst",
        "inst_map": lambda x: str(x).strip(),
        "source_rv": "VizieR J/A+A/517/L1 Table 0 (Queloz et al. 2010, HARPS/CORALIE)"
    },
    {
        "system_name": "WASP-23",
        "primary_planet": "WASP-23 b",
        "nasa_hostname": "WASP-23",
        "csv_basename": "wasp23_rv.csv",
        "alt_names": ["wasp-23_rv.csv", "wasp_23_rv.csv"],
        "pickle_name": "ac0bc2d45f0b99c7c3d7c81488b6c94c2678a25ef5dc55bf80cf2184.pickle",
        "table_index": 1,
        "time_col": "BJD",
        "time_offset": 0.0,
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": "Obs",
        "inst_map": lambda x: "CORALIE" if str(x).strip().upper() == "C" else "HARPS",
        "source_rv": "VizieR J/A+A/531/A24 Table 1 (Triaud et al. 2011, CORALIE)"
    },
    {
        "system_name": "WASP-31",
        "primary_planet": "WASP-31 b",
        "nasa_hostname": "WASP-31",
        "csv_basename": "wasp31_rv.csv",
        "alt_names": ["wasp-31_rv.csv", "wasp_31_rv.csv"],
        "pickle_name": "acc59df17649c42c90001642e7caaad950ac6e89f0c925379978299b.pickle",
        "table_index": 1,
        "time_col": "JD",
        "time_offset": 0.0,
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": "Inst",
        "inst_map": lambda x: str(x).strip() if str(x).strip() else "CORALIE",
        "source_rv": "VizieR J/A+A/531/A60 Table 1 (Anderson et al. 2011, CORALIE)"
    },
    {
        "system_name": "WASP-34",
        "primary_planet": "WASP-34 b",
        "nasa_hostname": "WASP-34",
        "csv_basename": "wasp34_rv.csv",
        "alt_names": ["wasp-34_rv.csv", "wasp_34_rv.csv"],
        "pickle_name": "c887234639584b762b4cdfef41270b4118cdb876e9d9e8981f397269.pickle",
        "table_index": 0,
        "time_col": "HJD",
        "time_offset": 0.0,
        "rv_col": "HRV",
        "rv_scale": 1000.0,
        "err_col": "e_HRV",
        "err_scale": 1000.0,
        "inst_col": None,
        "inst_map": lambda x: "CORALIE",
        "source_rv": "VizieR J/A+A/526/A130 Table 0 (Smalley et al. 2011, CORALIE)"
    },
    {
        "system_name": "WASP-50",
        "primary_planet": "WASP-50 b",
        "nasa_hostname": "WASP-50",
        "csv_basename": "wasp50_rv.csv",
        "alt_names": ["wasp-50_rv.csv", "wasp_50_rv.csv"],
        "pickle_name": "0669daededa983428ec3bfc4a29539260844ecedcd20dbaa1a7f7278.pickle",
        "table_index": 3,
        "time_col": "HJD",
        "time_offset": 0.0,
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1.0,  # e_RV is already in m/s (8.5 - 14.5 m/s)
        "inst_col": None,
        "inst_map": lambda x: "CORALIE",
        "source_rv": "VizieR J/A+A/533/A88 Table 3 (Gillon et al. 2011, CORALIE)"
    },
    {
        "system_name": "WASP-54",
        "primary_planet": "WASP-54 b",
        "nasa_hostname": "WASP-54",
        "csv_basename": "wasp54_rv.csv",
        "alt_names": ["wasp-54_rv.csv", "wasp_54_rv.csv"],
        "pickle_name": "8e53c4b3edaff9884b29898dee7a62403907e49302f3b0a0690c70e3.pickle",
        "table_index": 0,
        "time_col": "BJD",
        "time_offset": 0.0,
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": "Inst",
        "inst_map": lambda x: str(x).strip() if str(x).strip() else "CORALIE",
        "source_rv": "VizieR J/A+A/551/A73 Table 0 (Faedi et al. 2013, CORALIE)"
    },
    {
        "system_name": "WASP-80",
        "primary_planet": "WASP-80 b",
        "nasa_hostname": "WASP-80",
        "csv_basename": "wasp80_rv.csv",
        "alt_names": ["wasp-80_rv.csv", "wasp_80_rv.csv"],
        "pickle_name": "0e2f2efc8c1e4d3316d349d981c79df5f1a7e78391a45e5a0638c0cb.pickle",
        "table_index": 2,
        "time_col": "BJD",
        "time_offset": 2400000.0,  # MJD to BJD
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": "Obs",
        "inst_map": lambda x: "CORALIE" if str(x).strip().upper() == "C" else "HARPS",
        "source_rv": "VizieR J/A+A/551/A80 Table 2 (Triaud et al. 2013, HARPS/CORALIE)"
    },
    {
        "system_name": "WASP-103",
        "primary_planet": "WASP-103 b",
        "nasa_hostname": "WASP-103",
        "csv_basename": "wasp103_rv.csv",
        "alt_names": ["wasp-103_rv.csv", "wasp_103_rv.csv"],
        "pickle_name": "5810c90c1fcdabf091d7506b90a9bdaf67ab0661ef09aae668c09b2e.pickle",
        "table_index": 3,
        "time_col": "HJD",
        "time_offset": 2450000.0,  # HJD-2450000 to full HJD/BJD
        "rv_col": "RV",
        "rv_scale": 1000.0,
        "err_col": "e_RV",
        "err_scale": 1000.0,
        "inst_col": None,
        "inst_map": lambda x: "CORALIE",
        "source_rv": "VizieR J/A+A/562/L3 Table 3 (Gillon et al. 2014, CORALIE)"
    }
]

for wcfg in wasp_configs:
    sname = wcfg["system_name"]
    p_path = os.path.join(CACHE_DIR, wcfg["pickle_name"])
    with open(p_path, "rb") as f:
        resp = pickle.load(f)
    vot = parse(io.BytesIO(resp.content))
    tables = [t.to_table() for t in vot.iter_tables()]
    tab = tables[wcfg["table_index"]]
    
    times = np.array(tab[wcfg["time_col"]], dtype=np.float64) + wcfg["time_offset"]
    rvs = np.array(tab[wcfg["rv_col"]], dtype=np.float64) * wcfg["rv_scale"]
    errs = np.array(tab[wcfg["err_col"]], dtype=np.float64) * wcfg["err_scale"]
    
    if wcfg["inst_col"] and wcfg["inst_col"] in tab.colnames:
        insts = np.array([wcfg["inst_map"](x) for x in tab[wcfg["inst_col"]]], dtype=str)
    else:
        insts = np.array([wcfg["inst_map"](None) for _ in times], dtype=str)
        
    # Sort chronologically
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    rvs = rvs[sort_idx]
    errs = errs[sort_idx]
    insts = insts[sort_idx]
    
    df = pd.DataFrame({
        "bjd": [f"{t:.6f}" for t in times],
        "rv_ms": [f"{v:.3f}" for v in rvs],
        "rv_err_ms": [f"{e:.3f}" for e in errs],
        "instrument": insts
    })
    
    csv_path = os.path.join(RV_DIR, wcfg["csv_basename"])
    df.to_csv(csv_path, index=False)
    
    # Create symlinks
    for alt in wcfg["alt_names"]:
        alt_path = os.path.join(RV_DIR, alt)
        if os.path.islink(alt_path) or os.path.exists(alt_path):
            os.remove(alt_path)
        os.symlink(wcfg["csv_basename"], alt_path)
        
    unique_inst = ", ".join(sorted(list(set(insts))))
    datasets_summary.append({
        "system_name": sname,
        "csv_file": f"data/rv_data/{wcfg['csv_basename']}",
        "n_obs": len(times),
        "bjd_min": float(times.min()),
        "bjd_max": float(times.max()),
        "baseline_days": float(times.max() - times.min()),
        "rv_err_median_ms": float(np.median(errs)),
        "instrument": unique_inst,
        "primary_planet": wcfg["primary_planet"],
        "nasa_hostname": wcfg["nasa_hostname"],
        "source_rv": wcfg["source_rv"]
    })
    print(f"WASP {sname}: {len(times)} RV points written to {wcfg['csv_basename']} (inst: {unique_inst})")

# 4. Assemble real_22_new_targets_metadata.json
metadata_targets = []

for summary in datasets_summary:
    sname = summary["system_name"]
    host = summary["nasa_hostname"]
    prim_name = summary["primary_planet"]
    
    host_pls = nasa_by_host.get(host, [])
    # Find primary planet
    prim_p = next((p for p in host_pls if p["pl_name"].lower() == prim_name.lower()), None)
    if prim_p is None and host_pls:
        # fallback to first
        prim_p = host_pls[0]
        
    tic_id = prim_p.get("tic_id", "") if prim_p else ""
    tic_num = None
    if tic_id and "TIC" in str(tic_id):
        try:
            tic_num = int(str(tic_id).replace("TIC", "").strip())
        except ValueError:
            pass
            
    # Duration, depth ppm, etc.
    p_days = float(prim_p["pl_orbper"]) if prim_p and prim_p.get("pl_orbper") is not None else None
    t0 = float(prim_p["pl_tranmid"]) if prim_p and prim_p.get("pl_tranmid") is not None else None
    rp = float(prim_p["pl_rade"]) if prim_p and prim_p.get("pl_rade") is not None else None
    rp_jup = float(prim_p["pl_radj"]) if prim_p and prim_p.get("pl_radj") is not None else None
    mp = float(prim_p["pl_bmasse"]) if prim_p and prim_p.get("pl_bmasse") is not None else None
    mp_jup = float(prim_p["pl_bmassj"]) if prim_p and prim_p.get("pl_bmassj") is not None else None
    density = float(prim_p["pl_dens"]) if prim_p and prim_p.get("pl_dens") is not None else None
    k_rv = float(prim_p["pl_rvamp"]) if prim_p and prim_p.get("pl_rvamp") is not None else None
    dur_h = float(prim_p["pl_trandur"]) if prim_p and prim_p.get("pl_trandur") is not None else None
    depth_pct = float(prim_p["pl_trandep"]) if prim_p and prim_p.get("pl_trandep") is not None else None
    depth_ppm = depth_pct * 10000.0 if depth_pct is not None else None
    r_star = float(prim_p["st_rad"]) if prim_p and prim_p.get("st_rad") is not None else None
    m_star = float(prim_p["st_mass"]) if prim_p and prim_p.get("st_mass") is not None else None
    teff = float(prim_p["st_teff"]) if prim_p and prim_p.get("st_teff") is not None else None
    
    # All planets in the system
    all_pls = []
    for pl in host_pls:
        all_pls.append({
            "pl_name": pl.get("pl_name"),
            "pl_letter": pl.get("pl_letter"),
            "period_days": float(pl["pl_orbper"]) if pl.get("pl_orbper") is not None else None,
            "t0_bjd": float(pl["pl_tranmid"]) if pl.get("pl_tranmid") is not None else None,
            "rp_rearth": float(pl["pl_rade"]) if pl.get("pl_rade") is not None else None,
            "rp_rjup": float(pl["pl_radj"]) if pl.get("pl_radj") is not None else None,
            "mp_mearth": float(pl["pl_bmasse"]) if pl.get("pl_bmasse") is not None else None,
            "mp_mjup": float(pl["pl_bmassj"]) if pl.get("pl_bmassj") is not None else None,
            "density_g_cm3": float(pl["pl_dens"]) if pl.get("pl_dens") is not None else None,
            "rv_semiamplitude_ms": float(pl["pl_rvamp"]) if pl.get("pl_rvamp") is not None else None,
            "transit_duration_hours": float(pl["pl_trandur"]) if pl.get("pl_trandur") is not None else None,
            "transit_depth_percent": float(pl["pl_trandep"]) if pl.get("pl_trandep") is not None else None,
            "transit_depth_ppm": float(pl["pl_trandep"])*10000.0 if pl.get("pl_trandep") is not None else None
        })

    target_entry = {
        "system_name": sname,
        "nasa_hostname": host,
        "planet_name": prim_p.get("pl_name", f"{sname} b") if prim_p else f"{sname} b",
        "tic_id": tic_id,
        "tic_number": tic_num,
        "rv_file": summary["csv_file"],
        "instrument": summary["instrument"],
        "n_rv_observations": summary["n_obs"],
        "baseline_days": summary["baseline_days"],
        "bjd_min": summary["bjd_min"],
        "bjd_max": summary["bjd_max"],
        "rv_err_median_ms": summary["rv_err_median_ms"],
        "period_days": p_days,
        "t0_bjd": t0,
        "rp_rearth": rp,
        "rp_rjup": rp_jup,
        "mp_mearth": mp,
        "mp_mjup": mp_jup,
        "density_g_cm3": density,
        "transit_duration_hours": dur_h,
        "transit_depth_percent": depth_pct,
        "transit_depth_ppm": depth_ppm,
        "m_star_msun": m_star,
        "r_star_rsun": r_star,
        "st_teff_k": teff,
        "k_rv_semiamplitude_ms": k_rv,
        "num_planets_in_system": len(all_pls),
        "all_planets": all_pls,
        "source_rv": summary["source_rv"],
        "source_params": "NASA Exoplanet Archive (pscomppars via TAP API)"
    }
    metadata_targets.append(target_entry)

full_metadata = {
    "catalog_source": "VizieR (Bonomo et al. 2023 J/A+A/677/A33 & WASP literature) and NASA Exoplanet Archive (pscomppars)",
    "n_systems": len(metadata_targets),
    "total_rv_observations": sum(t["n_rv_observations"] for t in metadata_targets),
    "targets": metadata_targets
}

with open(METADATA_OUT, "w", encoding="utf-8") as f:
    json.dump(full_metadata, f, indent=2)

print(f"\nSuccessfully created {METADATA_OUT}:")
print(f"Total systems: {full_metadata['n_systems']}")
print(f"Total RV observations: {full_metadata['total_rv_observations']}")
