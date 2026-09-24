"""
MCMC transit fitting engine using emcee and analytical transit models (batman).
Includes Nelder-Mead MAP finding, affine-invariant MCMC ensemble sampling,
Gelman-Rubin convergence diagnostics, full credible intervals (1-sigma and 3-sigma),
and publication-quality corner and light curve visualization.
"""

import os
from typing import Optional, Dict, Any, Tuple, List
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
    compute_transit_duration,
    align_t0_to_dataset,
)
from astro_exo.models.diagnostics import compute_gelman_rubin, compute_effective_sample_size


def evaluate_batman_model(
    theta: np.ndarray,
    t: np.ndarray,
    period: float,
    supersample_factor: int = 1,
    exp_time: float = 0.0
) -> np.ndarray:
    """
    Evaluates the batman transit forward model.
    theta = [t0, rp, a_rs, b, q1, q2, f0]
    """
    import batman

    t0, rp, a_rs, b, q1, q2, f0 = theta
    u1, u2 = kipping_to_quadratic(q1, q2)

    cos_inc = b / a_rs
    if cos_inc < 0.0 or cos_inc > 1.0:
        return np.full_like(t, np.nan)
    inc = np.degrees(np.arccos(cos_inc))

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    if supersample_factor > 1 and exp_time > 0:
        model = batman.TransitModel(
            params, t, supersample_factor=supersample_factor, exp_time=exp_time
        )
    else:
        model = batman.TransitModel(params, t)

    return f0 * model.light_curve(params)


def log_uniform_prior(theta: np.ndarray, t0_ref: float, period: float = 3.5) -> float:
    """
    Standard physically uninformative uniform priors.
    theta = [t0, rp, a_rs, b, q1, q2, f0]
    """
    t0, rp, a_rs, b, q1, q2, f0 = theta

    t0_tol = max(0.12, 0.08 * period)
    if not (t0_ref - t0_tol < t0 < t0_ref + t0_tol):
        return -np.inf
    if not (0.005 < rp < 0.35):
        return -np.inf
    if not (1.5 < a_rs < 80.0):
        return -np.inf
    if not (0.0 <= b < (1.0 + rp)):
        return -np.inf
    if not (0.0 < q1 < 1.0):
        return -np.inf
    if not (0.0 < q2 < 1.0):
        return -np.inf
    if not (0.90 < f0 < 1.10):
        return -np.inf

    if b / a_rs >= 1.0:
        return -np.inf

    return 0.0


def log_likelihood(
    theta: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    period: float,
    supersample_factor: int = 1,
    exp_time: float = 0.0
) -> float:
    """Gaussian log likelihood."""
    model = evaluate_batman_model(theta, t, period, supersample_factor, exp_time)
    if not np.all(np.isfinite(model)):
        return -np.inf
    residuals = (y - model) / yerr
    chi2 = np.sum(residuals ** 2)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def log_posterior(
    theta: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    period: float,
    t0_ref: float,
    supersample_factor: int = 1,
    exp_time: float = 0.0
) -> float:
    """Combined log posterior."""
    lp = log_uniform_prior(theta, t0_ref, period)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, y, yerr, period, supersample_factor, exp_time)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


class EmceeTransitFitter:
    """
    Orchestrates Maximum A Posteriori (MAP) finding, affine-invariant
    MCMC ensemble sampling, convergence diagnostics, and physical parameter derivation.
    """

    PARAM_NAMES = ["t0", "rp", "a_rs", "b", "q1", "q2", "f0"]
    PARAM_LABELS = [r"$T_0$ (BTJD)", r"$R_p/R_\star$", r"$a/R_\star$", r"$b$", r"$q_1$", r"$q_2$", r"$F_0$"]

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        period: float,
        t0_expected: float,
        supersample_factor: int = 1,
        exp_time: float = 0.0
    ):
        # Strict sanitization against NaNs, infinities, and non-positive errors
        valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
        self.time = np.ascontiguousarray(time[valid], dtype=np.float64)
        self.flux = np.ascontiguousarray(flux[valid], dtype=np.float64)
        self.flux_err = np.ascontiguousarray(flux_err[valid], dtype=np.float64)
        self.period = float(period)

        # Align reference epoch to the dataset observation window
        self.t0_expected = align_t0_to_dataset(float(t0_expected), self.period, self.time)

        self.supersample_factor = supersample_factor
        self.exp_time = exp_time

        self.theta_map: Optional[np.ndarray] = None
        self.sampler: Optional[Any] = None
        self.samples: Optional[np.ndarray] = None
        self.chain: Optional[np.ndarray] = None
        self.derived_samples: Optional[Dict[str, np.ndarray]] = None

    def fit_map(self, theta_initial: Optional[np.ndarray] = None) -> np.ndarray:
        """Find Maximum A Posteriori (MAP) starting position."""
        if minimize is None:
            raise ImportError(
                "scipy is required for Nelder-Mead optimization in EmceeTransitFitter."
            )

        if theta_initial is None:
            # Estimate initial depth from transit center dip
            phase = (self.time - self.t0_expected + 0.5 * self.period) % self.period - 0.5 * self.period
            in_center = np.abs(phase) < 0.02
            if np.any(in_center):
                approx_depth = max(0.001, float(1.0 - np.nanmedian(self.flux[in_center])))
                approx_rp = np.clip(np.sqrt(approx_depth), 0.02, 0.25)
            else:
                approx_rp = 0.08

            theta_initial = np.array([self.t0_expected, approx_rp, 12.0, 0.3, 0.35, 0.30, 1.0])

        def nll(p):
            lp = log_uniform_prior(p, self.t0_expected, self.period)
            if not np.isfinite(lp):
                # Penalty that guides the optimizer back into the prior
                dist_t0 = (p[0] - self.t0_expected) ** 2
                return 1e6 + 1e5 * dist_t0
            ll = log_likelihood(
                p, self.time, self.flux, self.flux_err, self.period,
                self.supersample_factor, self.exp_time
            )
            return -(lp + ll)

        res = minimize(nll, theta_initial, method="Nelder-Mead", options={"maxiter": 2500})
        # If Nelder-Mead converged inside the prior, use it; otherwise fallback to initial
        if np.isfinite(log_uniform_prior(res.x, self.t0_expected, self.period)):
            self.theta_map = res.x
        else:
            self.theta_map = theta_initial
        return self.theta_map

    def run_mcmc(
        self,
        nwalkers: int = 32,
        nburn: int = 400,
        nprod: int = 1200,
        pool: Optional[Any] = None,
        seed: int = 42,
        progress: bool = False
    ) -> np.ndarray:
        """Execute burn-in and production sampling with emcee."""
        if emcee is None:
            raise ImportError("emcee is required for MCMC sampling. Install with `pip install emcee`.")

        if self.theta_map is None:
            self.fit_map()

        ndim = len(self.theta_map)
        rng = np.random.default_rng(seed)

        # Scale perturbations appropriately for each parameter to ensure linear independence
        param_scales = np.array([
            0.0015,  # t0 (~2 minutes)
            0.004,   # rp
            0.4,     # a_rs
            0.04,    # b
            0.04,    # q1
            0.04,    # q2
            0.0008   # f0
        ])

        pos = np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            valid = False
            for attempt in range(200):
                candidate = self.theta_map + (param_scales * (0.8 + 0.1 * attempt)) * rng.standard_normal(ndim)
                if np.isfinite(log_uniform_prior(candidate, self.t0_expected, self.period)):
                    pos[i] = candidate
                    valid = True
                    break
            if not valid:
                # Safe jitter fallback
                pos[i] = self.theta_map + (param_scales * 0.05 * (i + 1))

        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            args=(
                self.time, self.flux, self.flux_err, self.period, self.t0_expected,
                self.supersample_factor, self.exp_time
            ),
            pool=pool
        )

        state = self.sampler.run_mcmc(pos, nburn, progress=progress)
        self.sampler.reset()
        self.sampler.run_mcmc(state, nprod, progress=progress)

        self.chain = self.sampler.get_chain()  # Shape: (nprod, nwalkers, ndim)
        self.samples = self.sampler.get_chain(flat=True)  # Shape: (nprod * nwalkers, ndim)
        self._compute_derived_parameters()
        return self.samples

    def _compute_derived_parameters(self) -> None:
        """Computes posterior chains for derived physical quantities."""
        if self.samples is None:
            return

        rp = self.samples[:, 1]
        a_rs = self.samples[:, 2]
        b = self.samples[:, 3]
        q1 = self.samples[:, 4]
        q2 = self.samples[:, 5]

        # 1. Quadratic limb darkening u1, u2
        sqrt_q1 = np.sqrt(q1)
        u1 = 2.0 * sqrt_q1 * q2
        u2 = sqrt_q1 * (1.0 - 2.0 * q2)

        # 2. Orbital inclination
        cos_i = np.clip(b / a_rs, 0.0, 1.0)
        inc_deg = np.degrees(np.arccos(cos_i))

        # 3. Transit depth (ppm)
        depth_ppm = (rp ** 2) * 1e6

        # 4. Stellar density (g/cm^3)
        G_ASTRO = 2940.457
        rho_ratio_sun = (4.0 * (np.pi ** 2) / (G_ASTRO * (self.period ** 2))) * (a_rs ** 3)
        rho_star = rho_ratio_sun * 1.408

        # 5. Transit duration T_14 (hours)
        sin_i = np.sqrt(np.maximum(0.0, 1.0 - cos_i**2))
        arg = (1.0 / np.maximum(a_rs, 1.0)) * np.sqrt(np.maximum(0.0, (1.0 + rp)**2 - b**2)) / np.maximum(sin_i, 1e-6)
        arg = np.clip(arg, 0.0, 1.0)
        t14_hours = (self.period / np.pi) * np.arcsin(arg) * 24.0

        self.derived_samples = {
            "u1": u1,
            "u2": u2,
            "inc_deg": inc_deg,
            "depth_ppm": depth_ppm,
            "rho_star_g_cm3": rho_star,
            "duration_hours": t14_hours,
        }

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return comprehensive posterior summary including 1-sigma and 3-sigma
        credible intervals for primary and derived physical parameters.
        """
        if self.samples is None:
            raise RuntimeError("MCMC has not been run yet.")

        all_params = {}
        for i, name in enumerate(self.PARAM_NAMES):
            all_params[name] = self.samples[:, i]

        if self.derived_samples:
            all_params.update(self.derived_samples)

        results = {}
        for name, vals in all_params.items():
            q0015, q16, med, q84, q9985 = np.percentile(vals, [0.15, 16.0, 50.0, 84.0, 99.85])
            results[name] = {
                "median": float(med),
                "err_minus_1s": float(med - q16),
                "err_plus_1s": float(q84 - med),
                "err_minus_3s": float(med - q0015),
                "err_plus_3s": float(q9985 - med),
                "q16": float(q16),
                "q50": float(med),
                "q84": float(q84),
            }

        return results

    def get_diagnostics(self) -> Dict[str, Any]:
        """Calculates Gelman-Rubin R-hat, autocorrelation time, and acceptance rates."""
        if self.chain is None or self.sampler is None:
            raise RuntimeError("MCMC has not been run yet.")

        # Gelman-Rubin on production chains (nprod, nwalkers, ndim)
        r_hat = compute_gelman_rubin(self.chain)

        # Autocorrelation time
        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            tau_dict = {name: float(tau[i]) for i, name in enumerate(self.PARAM_NAMES)}
        except Exception:
            tau_dict = {name: np.nan for name in self.PARAM_NAMES}

        acc_frac = self.sampler.acceptance_fraction
        return {
            "r_hat": {name: float(r_hat[i]) for i, name in enumerate(self.PARAM_NAMES)},
            "tau": tau_dict,
            "acceptance_fraction_mean": float(np.mean(acc_frac)),
            "acceptance_fraction_min": float(np.min(acc_frac)),
            "acceptance_fraction_max": float(np.max(acc_frac)),
        }

    def plot_corner(self, filepath: str, title: Optional[str] = None) -> None:
        """Generates a publication-quality corner plot of primary transit parameters."""
        if self.samples is None:
            raise RuntimeError("MCMC has not been run yet.")

        import corner
        import matplotlib
        import matplotlib.pyplot as plt

        fig = corner.corner(
            self.samples,
            labels=self.PARAM_LABELS,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 11},
            color="#2563eb",
            hist_kwargs={"density": True, "color": "#1d4ed8"},
            plot_datapoints=False,
            plot_density=True,
            fill_contours=True,
        )

        if title:
            fig.suptitle(title, fontsize=15, y=1.02, fontweight="bold")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_fit(
        self,
        filepath: str,
        title: Optional[str] = None,
        n_bins: int = 40
    ) -> None:
        """
        Plots phase-folded light curve with binned data, median model,
        1-sigma and 3-sigma credible envelopes, and residuals.
        """
        if self.samples is None:
            raise RuntimeError("MCMC has not been run yet.")

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        summary = self.get_summary()
        t0_med = summary["t0"]["median"]

        # Phase fold in hours
        phase_hours = ((self.time - t0_med + 0.5 * self.period) % self.period - 0.5 * self.period) * 24.0
        sort_idx = np.argsort(phase_hours)
        phase_sorted = phase_hours[sort_idx]
        flux_sorted = self.flux[sort_idx]
        err_sorted = self.flux_err[sort_idx]

        # Model evaluation on fine phase grid
        fine_phase_hours = np.linspace(np.min(phase_sorted), np.max(phase_sorted), 400)
        fine_time = t0_med + (fine_phase_hours / 24.0)

        # Median parameters
        theta_med = np.array([
            summary["t0"]["median"],
            summary["rp"]["median"],
            summary["a_rs"]["median"],
            summary["b"]["median"],
            summary["q1"]["median"],
            summary["q2"]["median"],
            summary["f0"]["median"],
        ])
        model_median = evaluate_batman_model(theta_med, fine_time, self.period)

        # Sample posterior models for 1-sigma & 3-sigma envelopes
        rng = np.random.default_rng(42)
        random_indices = rng.choice(len(self.samples), size=min(150, len(self.samples)), replace=False)
        posterior_models = np.zeros((len(random_indices), len(fine_time)))
        for idx_draw, idx_samp in enumerate(random_indices):
            th = self.samples[idx_samp]
            posterior_models[idx_draw] = evaluate_batman_model(th, fine_time, self.period)

        env_q0015, env_q16, env_q84, env_q9985 = np.percentile(
            posterior_models, [0.15, 16.0, 84.0, 99.85], axis=0
        )

        # Residuals relative to median model
        model_at_data = evaluate_batman_model(theta_med, self.time, self.period)
        residuals = self.flux - model_at_data
        residuals_sorted = residuals[sort_idx]

        # Binning for visual clarity
        bin_edges = np.linspace(np.min(phase_sorted), np.max(phase_sorted), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned_flux = np.zeros(n_bins)
        binned_err = np.zeros(n_bins)
        binned_res = np.zeros(n_bins)
        valid_bins = []

        for i in range(n_bins):
            in_bin = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i + 1])
            if np.sum(in_bin) > 0:
                binned_flux[i] = np.mean(flux_sorted[in_bin])
                binned_err[i] = np.std(flux_sorted[in_bin]) / np.sqrt(np.sum(in_bin))
                binned_res[i] = np.mean(residuals_sorted[in_bin])
                valid_bins.append(i)

        valid_bins = np.array(valid_bins)

        # Matplotlib layout
        fig = plt.figure(figsize=(10, 7), facecolor="white")
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

        # Upper: Light curve & model
        ax0.scatter(phase_sorted, flux_sorted, c="#94a3b8", s=6, alpha=0.35, label="TESS PDCSAP Flux", edgecolors="none")
        if len(valid_bins) > 0:
            ax0.errorbar(
                bin_centers[valid_bins], binned_flux[valid_bins], yerr=binned_err[valid_bins],
                fmt="o", c="#0f172a", ms=5, capsize=2.5, lw=1.2, label="Binned Photometry", zorder=4
            )

        ax0.fill_between(
            fine_phase_hours, env_q0015, env_q9985, color="#38bdf8", alpha=0.3,
            label=r"$3\sigma$ Credible Envelope (99.7%)", zorder=2
        )
        ax0.fill_between(
            fine_phase_hours, env_q16, env_q84, color="#0284c7", alpha=0.5,
            label=r"$1\sigma$ Credible Envelope (68.3%)", zorder=3
        )
        ax0.plot(fine_phase_hours, model_median, c="#b91c1c", lw=2.2, label="Median Transit Model", zorder=5)

        ax0.set_ylabel("Relative Flux", fontsize=11, fontweight="bold")
        ax0.grid(True, linestyle="--", alpha=0.4)
        ax0.legend(loc="lower right", fontsize=9, framealpha=0.9)
        plt.setp(ax0.get_xticklabels(), visible=False)

        if title:
            ax0.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Lower: Residuals
        ax1.scatter(phase_sorted, residuals_sorted * 1e6, c="#94a3b8", s=6, alpha=0.35, edgecolors="none")
        if len(valid_bins) > 0:
            ax1.errorbar(
                bin_centers[valid_bins], binned_res[valid_bins] * 1e6, yerr=binned_err[valid_bins] * 1e6,
                fmt="o", c="#0f172a", ms=5, capsize=2.5, lw=1.2, zorder=4
            )
        ax1.axhline(0, color="#b91c1c", linestyle="--", lw=1.5)
        ax1.set_xlabel("Time from Mid-Transit (hours)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Residuals (ppm)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
