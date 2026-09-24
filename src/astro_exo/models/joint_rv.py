"""
Joint Bayesian inference of Photometric Transits and Radial Velocity (RV) Doppler measurements.
Solves for planetary mass, radius, bulk density, surface gravity, and orbital parameters
simultaneously across multiple spectrographs with separate instrumental zero-points and jitter.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    import emcee
except ImportError:
    emcee = None

from astro_exo.models.transforms import (
    kipping_to_quadratic,
    impact_param_to_inclination,
    compute_stellar_density,
)
from astro_exo.ingestion.rv_loader import RVDataset


# Physical constants (IAU 2015 / CODATA 2018 standard)
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
M_SUN_KG = 1.98847e30     # kg
R_SUN_M = 6.957e8         # m
M_JUP_KG = 1.89813e27     # kg
R_JUP_M = 7.1492e7        # m
M_EARTH_KG = 5.9722e24    # kg
R_EARTH_M = 6.3781e6      # m
AU_M = 1.495978707e11     # m


def keplerian_rv(
    time: np.ndarray,
    period: float,
    t0: float,
    k_semiamp: float,
    ecc: float = 0.0,
    omega_deg: float = 90.0,
    gamma: float = 0.0
) -> np.ndarray:
    """
    Computes Keplerian Radial Velocity curve for eccentric or circular orbits.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps (days, BJD).
    period : float
        Orbital period (days).
    t0 : float
        Time of mid-transit / inferior conjunction (days, BJD).
    k_semiamp : float
        Radial velocity semi-amplitude (m/s).
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    omega_deg : float
        Argument of periastron in degrees.
    gamma : float
        Systemic velocity / instrument zero point (m/s).

    Returns
    -------
    rv : np.ndarray
        Stellar radial velocity at timestamps (m/s).
    """
    time = np.asarray(time, dtype=np.float64)

    # For circular orbit (e = 0):
    # Mid-transit occurs at phi = 0, where RV passes gamma with negative slope:
    # v(t) = gamma - K * sin(2*pi*(t - t0)/P)
    if ecc <= 1e-7:
        mean_anom = 2.0 * np.pi * (time - t0) / period
        return gamma - k_semiamp * np.sin(mean_anom)

    # Eccentric orbit:
    # Transit occurs at true anomaly f_tra = pi/2 - omega
    omega_rad = np.radians(omega_deg)
    f_tra = np.pi / 2.0 - omega_rad

    # Eccentric anomaly at transit E_tra
    tan_e_tra_2 = np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(f_tra / 2.0)
    e_tra = 2.0 * np.arctan(tan_e_tra_2)
    m_tra = e_tra - ecc * np.sin(e_tra)

    # Mean anomaly at time t
    mean_anom = ((2.0 * np.pi * (time - t0) / period) + m_tra) % (2.0 * np.pi)

    # Solve Kepler's equation E - e*sin(E) = M via Newton-Raphson iteration
    e_anom = mean_anom.copy()
    for _ in range(7):
        f_eval = e_anom - ecc * np.sin(e_anom) - mean_anom
        f_prime = 1.0 - ecc * np.cos(e_anom)
        e_anom -= f_eval / f_prime

    # True anomaly f
    true_anom = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(e_anom / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(e_anom / 2.0)
    )

    return gamma + k_semiamp * (np.cos(true_anom + omega_rad) + ecc * np.cos(omega_rad))


def classify_planetary_interior(mass_earth: float, density_g_cm3: float) -> str:
    """
    Classifies planetary internal structure based on mass-radius-density regimes
    (Fortney et al. 2007; Zeng et al. 2016, 2019).
    """
    if mass_earth < 2.0:
        if density_g_cm3 >= 5.0:
            return "Terrestrial / Iron-Silicate (Rocky)"
        return "Low-Density Terrestrial / Volatiles"
    elif mass_earth < 10.0:
        if density_g_cm3 >= 5.0:
            return "Super-Earth (Dense Rocky)"
        elif density_g_cm3 >= 2.0:
            return "Water World / Volatile-Rich Super-Earth"
        else:
            return "Mini-Neptune / Low-Density Super-Earth"
    elif mass_earth < 50.0:
        if density_g_cm3 >= 2.5:
            return "Water Giant / Ocean Planet"
        else:
            return "Sub-Neptune / Ice Giant"
    else:  # Giant planets (Saturn / Jupiter mass range)
        if density_g_cm3 >= 0.90:
            return "Dense / Massive Hot Jupiter"
        elif density_g_cm3 >= 0.35:
            return "Standard Gas Giant / Hot Jupiter"
        else:
            return "Inflated / Low-Density Hot Jupiter"


def compute_planetary_mass_density(
    m_star_msun: float,
    r_star_rsun: float,
    period_days: float,
    k_semiamp_ms: float,
    rp_rs: float,
    ecc: float = 0.0,
    inc_deg: float = 90.0
) -> Dict[str, Union[float, str]]:
    """
    Derive physical planetary mass, radius, bulk density, surface gravity,
    semi-major axis, and interior structure classification.
    """
    p_sec = period_days * 86400.0
    m_star_kg = m_star_msun * M_SUN_KG
    r_star_m = r_star_rsun * R_SUN_M

    sin_i = np.sin(np.radians(inc_deg))
    if sin_i <= 0:
        sin_i = 1e-4

    sqrt_1_e2 = np.sqrt(np.clip(1.0 - ecc**2, 1e-6, 1.0))
    c_factor = (k_semiamp_ms * sqrt_1_e2 / sin_i) * ((p_sec / (2.0 * np.pi * G_SI)) ** (1.0 / 3.0))

    # Exact iterative solution for M_p: M_p = C * (M_* + M_p)^(2/3)
    m_p_kg = c_factor * (m_star_kg ** (2.0 / 3.0))
    for _ in range(5):
        m_p_kg = c_factor * ((m_star_kg + m_p_kg) ** (2.0 / 3.0))

    # Planetary radius
    r_p_m = rp_rs * r_star_m

    # Planetary mass in Earth and Jupiter units
    m_p_earth = m_p_kg / M_EARTH_KG
    m_p_jup = m_p_kg / M_JUP_KG

    # Planetary radius in Earth and Jupiter units
    r_p_earth = r_p_m / R_EARTH_M
    r_p_jup = r_p_m / R_JUP_M

    # Volume and mean density (g / cm^3)
    vol_m3 = (4.0 / 3.0) * np.pi * (r_p_m ** 3)
    rho_kg_m3 = m_p_kg / vol_m3
    rho_g_cm3 = rho_kg_m3 / 1000.0

    # Surface gravity: g_p = G * M_p / R_p^2 (cgs: cm/s^2)
    g_p_si = G_SI * m_p_kg / (r_p_m ** 2)
    g_p_cgs = g_p_si * 100.0
    log_g_cgs = np.log10(g_p_cgs)

    # Escape velocity (km / s)
    v_esc_kms = np.sqrt(2.0 * G_SI * m_p_kg / r_p_m) / 1000.0

    # Semi-major axis (AU)
    a_m = ((G_SI * (m_star_kg + m_p_kg) * (p_sec ** 2)) / (4.0 * (np.pi ** 2))) ** (1.0 / 3.0)
    a_au = a_m / AU_M

    interior_class = classify_planetary_interior(m_p_earth, rho_g_cm3)

    return {
        "mass_earth": float(m_p_earth),
        "mass_jupiter": float(m_p_jup),
        "radius_earth": float(r_p_earth),
        "radius_jupiter": float(r_p_jup),
        "density_g_cm3": float(rho_g_cm3),
        "log_g_cgs": float(log_g_cgs),
        "v_esc_kms": float(v_esc_kms),
        "a_au": float(a_au),
        "interior_classification": interior_class
    }


class JointTransitRVSampler:
    """
    Joint Bayesian MCMC sampler for transit photometry and multi-instrument radial velocities.
    """
    def __init__(
        self,
        rv_dataset: RVDataset,
        phot_time: Optional[np.ndarray] = None,
        phot_flux: Optional[np.ndarray] = None,
        phot_err: Optional[np.ndarray] = None,
        period_days: float = 3.0,
        t0_bjd: float = 0.0,
        m_star_msun: float = 1.0,
        r_star_rsun: float = 1.0,
        rp_rs_prior: float = 0.1,
        fit_eccentricity: bool = False
    ):
        self.rv = rv_dataset
        self.phot_time = phot_time
        self.phot_flux = phot_flux
        self.phot_err = phot_err

        self.period = float(period_days)
        self.t0_ref = float(t0_bjd)
        self.m_star = float(m_star_msun)
        self.r_star = float(r_star_rsun)
        self.rp_rs_init = float(rp_rs_prior)
        self.fit_eccentricity = fit_eccentricity

        self.instruments = self.rv.instrument_names
        self.n_inst = len(self.instruments)

        # Precompute target stellar density from host star properties
        vol_star_m3 = (4.0 / 3.0) * np.pi * ((self.r_star * R_SUN_M) ** 3)
        self.rho_star_target_cgs = (self.m_star * M_SUN_KG / vol_star_m3) / 1000.0

    def evaluate_rv_model(self, theta_rv: Dict[str, Any], time: np.ndarray, inst_indices: np.ndarray) -> np.ndarray:
        """
        Evaluates multi-instrument Keplerian RV model.
        """
        k_semiamp = theta_rv["k_semiamp"]
        t0 = theta_rv.get("t0", self.t0_ref)
        period = theta_rv.get("period", self.period)
        ecc = theta_rv.get("ecc", 0.0)
        omega = theta_rv.get("omega_deg", 90.0)
        gammas = theta_rv["gammas"]

        # Base Keplerian reflex motion (centered at 0)
        v_base = keplerian_rv(time, period, t0, k_semiamp, ecc, omega, gamma=0.0)

        # Add instrument-specific zero-points
        inst_shifts = np.array([gammas[idx] for idx in inst_indices])
        return v_base + inst_shifts

    def run_rv_mcmc(
        self,
        nwalkers: int = 32,
        nburn: int = 400,
        nsteps: int = 800,
        k_guess_ms: Optional[float] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs MCMC sampling for the radial velocity parameters (semi-amplitude K,
        systemic velocity offsets gamma_k, and instrument jitters sigma_k).
        """
        if emcee is None:
            raise ImportError("emcee is required for Bayesian sampling.")

        # Estimate initial K if not provided: peak-to-peak / 2
        if k_guess_ms is None:
            k_guess = float(np.nanpercentile(self.rv.rv_ms, 90) - np.nanpercentile(self.rv.rv_ms, 10)) / 2.0
            k_guess = max(5.0, k_guess)
        else:
            k_guess = float(k_guess_ms)

        # Parameter vector: [K, ecc (if enabled), omega (if enabled), gamma_0, ..., gamma_{N-1}, jitter_0, ..., jitter_{N-1}]
        param_names = ["k_semiamp"]
        init_vals = [k_guess]
        bounds = [(0.1, 2000.0)]

        if self.fit_eccentricity:
            param_names.extend(["ecc", "omega_deg"])
            init_vals.extend([0.05, 90.0])
            bounds.extend([(0.0, 0.85), (0.0, 360.0)])

        for inst in self.instruments:
            param_names.append(f"gamma_{inst}")
            inst_mask = self.rv.get_instrument_mask(inst)
            mean_v = float(np.nanmedian(self.rv.rv_ms[inst_mask])) if np.any(inst_mask) else 0.0
            init_vals.append(mean_v)
            bounds.append((mean_v - 200.0, mean_v + 200.0))

        for inst in self.instruments:
            param_names.append(f"jitter_{inst}")
            init_vals.append(2.0)
            bounds.append((0.0, 100.0))

        ndim = len(param_names)

        def unpack(p):
            idx = 0
            k_amp = p[idx]
            idx += 1
            if self.fit_eccentricity:
                ecc = p[idx]
                idx += 1
                omega = p[idx]
                idx += 1
            else:
                ecc = 0.0
                omega = 90.0
            gammas = p[idx:idx + self.n_inst]
            idx += self.n_inst
            jitters = p[idx:idx + self.n_inst]
            return k_amp, ecc, omega, gammas, jitters

        def log_prior(p):
            for val, (low, high) in zip(p, bounds):
                if not (low <= val <= high):
                    return -np.inf
            return 0.0

        def log_likelihood(p):
            k_amp, ecc, omega, gammas, jitters = unpack(p)
            theta_dict = {
                "k_semiamp": k_amp,
                "ecc": ecc,
                "omega_deg": omega,
                "gammas": gammas,
                "t0": self.t0_ref,
                "period": self.period
            }
            v_model = self.evaluate_rv_model(theta_dict, self.rv.time_bjd, self.rv.inst_indices)
            inst_jitters = np.array([jitters[i] for i in self.rv.inst_indices])
            sigma2 = (self.rv.rv_err_ms ** 2) + (inst_jitters ** 2)
            residuals = self.rv.rv_ms - v_model
            chi2 = np.sum((residuals ** 2) / sigma2)
            log_det = np.sum(np.log(2.0 * np.pi * sigma2))
            return -0.5 * (chi2 + log_det)

        def log_posterior(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(p)

        # Optimize MAP starting point
        nll = lambda p: -log_likelihood(p) if np.isfinite(log_prior(p)) else 1e12
        if minimize is not None:
            opt = minimize(nll, init_vals, method="Nelder-Mead", options={"maxiter": 1500})
            if opt.success and np.isfinite(log_prior(opt.x)):
                start_p = opt.x
            else:
                start_p = np.array(init_vals)
        else:
            start_p = np.array(init_vals)

        # Initialize walkers
        rng = np.random.default_rng(random_seed)
        pos = start_p + 1e-3 * rng.standard_normal((nwalkers, ndim))
        # Ensure within bounds
        for i in range(nwalkers):
            for j, (low, high) in enumerate(bounds):
                pos[i, j] = np.clip(pos[i, j], low + 1e-4, high - 1e-4)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        # Burn-in
        state = sampler.run_mcmc(pos, nburn, progress=False)
        sampler.reset()
        # Production
        sampler.run_mcmc(state, nsteps, progress=False)

        samples = sampler.get_chain(flat=True)

        # Extract 16th, 50th, 84th percentiles
        pcts = np.percentile(samples, [16, 50, 84], axis=0)
        p16, p50, p84 = pcts[0], pcts[1], pcts[2]

        param_results = {}
        for idx, name in enumerate(param_names):
            med = float(p50[idx])
            err_pos = float(p84[idx] - med)
            err_neg = float(med - p16[idx])
            param_results[name] = {
                "median": med,
                "error_plus": err_pos,
                "error_minus": err_neg,
                "std": float(np.std(samples[:, idx]))
            }

        k_med = param_results["k_semiamp"]["median"]
        ecc_med = param_results["ecc"]["median"] if self.fit_eccentricity else 0.0

        # Compute physical planetary mass and bulk density
        physical = compute_planetary_mass_density(
            m_star_msun=self.m_star,
            r_star_rsun=self.r_star,
            period_days=self.period,
            k_semiamp_ms=k_med,
            rp_rs=self.rp_rs_init,
            ecc=ecc_med,
            inc_deg=90.0
        )

        # Compute uncertainties on Mp and density via sample propagation
        k_chain = samples[:, 0]
        ecc_chain = samples[:, 1] if self.fit_eccentricity else np.zeros_like(k_chain)
        m_jup_chain = []
        rho_chain = []

        for k_val, e_val in zip(k_chain[::5], ecc_chain[::5]):
            phys_sample = compute_planetary_mass_density(
                m_star_msun=self.m_star,
                r_star_rsun=self.r_star,
                period_days=self.period,
                k_semiamp_ms=k_val,
                rp_rs=self.rp_rs_init,
                ecc=e_val,
                inc_deg=90.0
            )
            m_jup_chain.append(phys_sample["mass_jupiter"])
            rho_chain.append(phys_sample["density_g_cm3"])

        m_jup_pcts = np.percentile(m_jup_chain, [16, 50, 84])
        rho_pcts = np.percentile(rho_chain, [16, 50, 84])

        physical["mass_jupiter_err"] = float((m_jup_pcts[2] - m_jup_pcts[0]) / 2.0)
        physical["density_g_cm3_err"] = float((rho_pcts[2] - rho_pcts[0]) / 2.0)

        # Residuals calculation with median parameters
        k_amp_med, ecc_med, omega_med, gammas_med, jitters_med = unpack(p50)
        theta_med = {
            "k_semiamp": k_amp_med,
            "ecc": ecc_med,
            "omega_deg": omega_med,
            "gammas": gammas_med,
            "t0": self.t0_ref,
            "period": self.period
        }
        v_model_med = self.evaluate_rv_model(theta_med, self.rv.time_bjd, self.rv.inst_indices)
        residuals = self.rv.rv_ms - v_model_med
        rms_ms = float(np.sqrt(np.mean(residuals ** 2)))

        return {
            "parameters": param_results,
            "physical": physical,
            "rms_residuals_ms": rms_ms,
            "reduced_chi2": float(np.sum((residuals / self.rv.rv_err_ms)**2) / max(1, self.rv.n_points - ndim)),
            "n_samples": len(samples),
            "samples": samples,
            "param_names": param_names
        }

    run_mcmc = run_rv_mcmc
