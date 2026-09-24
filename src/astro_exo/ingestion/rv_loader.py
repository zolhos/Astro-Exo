"""
Radial Velocity (RV) Data Ingestion and Multi-Instrument Preprocessing.
Supports standard astronomical CSV formats, instrument mapping, phase folding,
and high-fidelity Doppler simulation with instrumental zero-points and jitter.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import os
import numpy as np


@dataclass
class RVDataset:
    """
    Structured container for Doppler radial velocity time-series.
    """
    time_bjd: np.ndarray
    rv_ms: np.ndarray
    rv_err_ms: np.ndarray
    instruments: np.ndarray  # String array of instrument names
    instrument_names: List[str] = field(init=False)
    inst_indices: np.ndarray = field(init=False)

    def __post_init__(self):
        self.time_bjd = np.asarray(self.time_bjd, dtype=np.float64)
        self.rv_ms = np.asarray(self.rv_ms, dtype=np.float64)
        self.rv_err_ms = np.asarray(self.rv_err_ms, dtype=np.float64)
        self.instruments = np.asarray(self.instruments, dtype=str)

        # Unique instrument identifiers (sorted deterministically)
        unique_insts = [str(x) for x in sorted(list(set(self.instruments)))]
        self.instrument_names = unique_insts

        # Map each observation to its integer instrument index [0, N_inst - 1]
        inst_map = {name: i for i, name in enumerate(unique_insts)}
        self.inst_indices = np.array([inst_map[name] for name in self.instruments], dtype=np.int32)

    @property
    def n_points(self) -> int:
        return len(self.time_bjd)

    def __len__(self) -> int:
        return self.n_points

    @property
    def n_instruments(self) -> int:
        return len(self.instrument_names)

    @property
    def baseline_days(self) -> float:
        if self.n_points == 0:
            return 0.0
        return float(np.nanmax(self.time_bjd) - np.nanmin(self.time_bjd))

    def phase_fold(self, period_days: float, t0_bjd: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates orbital phase phi in [-0.5, 0.5] and sorts all arrays.
        """
        phase = ((self.time_bjd - t0_bjd + 0.5 * period_days) % period_days) / period_days - 0.5
        sort_idx = np.argsort(phase)
        return phase[sort_idx], sort_idx

    def get_instrument_mask(self, inst_name: str) -> np.ndarray:
        return self.instruments == inst_name

    def to_dict(self) -> Dict[str, Union[int, float, List[str]]]:
        return {
            "n_points": self.n_points,
            "n_instruments": self.n_instruments,
            "instruments": self.instrument_names,
            "baseline_days": self.baseline_days,
            "rv_mean_ms": float(np.nanmean(self.rv_ms)) if self.n_points > 0 else 0.0,
            "rv_std_ms": float(np.nanstd(self.rv_ms)) if self.n_points > 0 else 0.0,
            "rv_err_median_ms": float(np.nanmedian(self.rv_err_ms)) if self.n_points > 0 else 0.0,
        }


def load_rv_csv(file_path: str) -> RVDataset:
    """
    Load an RV dataset from a CSV file.
    Expected columns: bjd (or time), rv_ms (or rv), rv_err_ms (or rv_err), instrument (optional, default 'DEFAULT')
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"RV data file not found: {file_path}")

    # Read header and data
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        raise ValueError(f"Empty RV data file: {file_path}")

    header = [col.lower().strip() for col in lines[0].split(",")]

    # Column resolution
    col_time = next((i for i, c in enumerate(header) if c in ("bjd", "time", "hjd", "rjd", "bjd_tdb")), 0)
    col_rv = next((i for i, c in enumerate(header) if c in ("rv_ms", "rv", "radial_velocity", "vrad")), 1)
    col_err = next((i for i, c in enumerate(header) if c in ("rv_err_ms", "rv_err", "err", "e_rv", "sigma")), 2)
    col_inst = next((i for i, c in enumerate(header) if c in ("instrument", "inst", "telescope")), None)

    times = []
    rvs = []
    errs = []
    insts = []

    for row_idx, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= max(col_time, col_rv, col_err):
            continue
        try:
            t = float(parts[col_time])
            v = float(parts[col_rv])
            e = float(parts[col_err])
            inst = parts[col_inst] if col_inst is not None and len(parts) > col_inst else "DEFAULT"
            times.append(t)
            rvs.append(v)
            errs.append(e)
            insts.append(inst)
        except ValueError:
            continue

    if not times:
        raise ValueError(f"No valid numerical RV data rows found in {file_path}")

    return RVDataset(
        time_bjd=np.array(times, dtype=np.float64),
        rv_ms=np.array(rvs, dtype=np.float64),
        rv_err_ms=np.array(errs, dtype=np.float64),
        instruments=np.array(insts, dtype=str)
    )


def simulate_multi_instrument_rv(
    period_days: float,
    t0_bjd: float,
    k_semiamp_ms: float,
    ecc: float = 0.0,
    omega_deg: float = 90.0,
    n_points_per_inst: Optional[Dict[str, int]] = None,
    gamma_offsets: Optional[Dict[str, float]] = None,
    jitters_ms: Optional[Dict[str, float]] = None,
    nominal_errors_ms: Optional[Dict[str, float]] = None,
    baseline_days: float = 45.0,
    random_seed: int = 42
) -> Tuple[RVDataset, Dict[str, float]]:
    """
    Simulates high-fidelity multi-instrument Doppler data for validation & testing.

    Returns
    -------
    dataset : RVDataset
    ground_truth : dict
        Contains the simulated ground truth parameters.
    """
    from astro_exo.models.joint_rv import keplerian_rv

    rng = np.random.default_rng(random_seed)

    if n_points_per_inst is None:
        n_points_per_inst = {"HARPS": 25, "CORALIE": 18}
    if gamma_offsets is None:
        gamma_offsets = {"HARPS": 0.0, "CORALIE": -15.4}
    if jitters_ms is None:
        jitters_ms = {"HARPS": 1.2, "CORALIE": 4.5}
    if nominal_errors_ms is None:
        nominal_errors_ms = {"HARPS": 2.5, "CORALIE": 8.0}

    all_times = []
    all_rvs = []
    all_errs = []
    all_insts = []

    for inst_name, count in n_points_per_inst.items():
        # Random observations within baseline
        t_obs = t0_bjd + np.sort(rng.uniform(0.0, baseline_days, size=count))
        gamma = gamma_offsets.get(inst_name, 0.0)
        jitter = jitters_ms.get(inst_name, 0.0)
        nom_err = nominal_errors_ms.get(inst_name, 3.0)

        # Pure Keplerian reflex motion
        v_pure = keplerian_rv(
            time=t_obs,
            period=period_days,
            t0=t0_bjd,
            k_semiamp=k_semiamp_ms,
            ecc=ecc,
            omega_deg=omega_deg,
            gamma=gamma
        )

        # Error and jitter perturbation
        err = np.full(count, nom_err)
        total_sigma = np.sqrt(err**2 + jitter**2)
        noise = rng.normal(0.0, total_sigma)

        v_obs = v_pure + noise

        all_times.extend(t_obs)
        all_rvs.extend(v_obs)
        all_errs.extend(err)
        all_insts.extend([inst_name] * count)

    # Sort globally by observation time
    sort_idx = np.argsort(all_times)
    time_arr = np.array(all_times, dtype=np.float64)[sort_idx]
    rv_arr = np.array(all_rvs, dtype=np.float64)[sort_idx]
    err_arr = np.array(all_errs, dtype=np.float64)[sort_idx]
    inst_arr = np.array(all_insts, dtype=str)[sort_idx]

    dataset = RVDataset(
        time_bjd=time_arr,
        rv_ms=rv_arr,
        rv_err_ms=err_arr,
        instruments=inst_arr
    )

    ground_truth = {
        "period_days": period_days,
        "t0_bjd": t0_bjd,
        "k_semiamp_ms": k_semiamp_ms,
        "ecc": ecc,
        "omega_deg": omega_deg,
        "gamma_offsets": gamma_offsets,
        "jitters_ms": jitters_ms
    }

    return dataset, ground_truth
