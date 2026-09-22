"""
MCMC transit fitting engine using emcee and analytical transit models (batman).
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    import emcee
except ImportError:
    emcee = None

from astro_exo.models.transforms import kipping_to_quadratic


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


def log_uniform_prior(theta: np.ndarray, t0_ref: float) -> float:
    """
    Standard physically uninformative uniform priors.
    theta = [t0, rp, a_rs, b, q1, q2, f0]
    """
    t0, rp, a_rs, b, q1, q2, f0 = theta

    if not (t0_ref - 0.1 < t0 < t0_ref + 0.1):
        return -np.inf
    if not (0.005 < rp < 0.3):
        return -np.inf
    if not (1.5 < a_rs < 50.0):
        return -np.inf
    if not (0.0 <= b < (1.0 + rp)):
        return -np.inf
    if not (0.0 < q1 < 1.0):
        return -np.inf
    if not (0.0 < q2 < 1.0):
        return -np.inf
    if not (0.95 < f0 < 1.05):
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
    if np.any(np.isnan(model)):
        return -np.inf
    chi2 = np.sum(((y - model) / yerr) ** 2)
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
    lp = log_uniform_prior(theta, t0_ref)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr, period, supersample_factor, exp_time)


class EmceeTransitFitter:
    """
    Orchestrates Maximum A Posteriori (MAP) finding and affine-invariant
    MCMC ensemble sampling for exoplanet transit light curves.
    """

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
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.period = period
        self.t0_expected = t0_expected
        self.supersample_factor = supersample_factor
        self.exp_time = exp_time

        self.theta_map: Optional[np.ndarray] = None
        self.sampler: Optional[Any] = None
        self.samples: Optional[np.ndarray] = None

    def fit_map(self, theta_initial: Optional[np.ndarray] = None) -> np.ndarray:
        """Find Maximum A Posteriori (MAP) starting position."""
        if minimize is None:
            raise ImportError(
                "scipy is required for Nelder-Mead optimization in EmceeTransitFitter. "
                "Install with `pip install scipy`."
            )

        if theta_initial is None:
            # Default initial parameter vector: [t0, rp, a_rs, b, q1, q2, f0]
            theta_initial = np.array([self.t0_expected, 0.08, 10.0, 0.5, 0.35, 0.30, 1.0])

        def nll(p):
            lp = log_uniform_prior(p, self.t0_expected)
            if not np.isfinite(lp):
                return 1e12
            ll = log_likelihood(
                p, self.time, self.flux, self.flux_err, self.period,
                self.supersample_factor, self.exp_time
            )
            return -(lp + ll)

        res = minimize(nll, theta_initial, method="Nelder-Mead")
        self.theta_map = res.x
        return self.theta_map

    def run_mcmc(
        self,
        nwalkers: int = 32,
        nburn: int = 400,
        nprod: int = 1200,
        seed: int = 42
    ) -> np.ndarray:
        """Execute burn-in and production sampling with emcee."""
        if emcee is None:
            raise ImportError(
                "emcee is required for MCMC sampling. "
                "Install with `pip install emcee`."
            )

        if self.theta_map is None:
            self.fit_map()

        ndim = len(self.theta_map)
        rng = np.random.default_rng(seed)
        pos = self.theta_map + 1e-4 * rng.standard_normal((nwalkers, ndim))

        for i in range(nwalkers):
            while not np.isfinite(log_uniform_prior(pos[i], self.t0_expected)):
                pos[i] = self.theta_map + 1e-3 * rng.standard_normal(ndim)

        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            args=(
                self.time, self.flux, self.flux_err, self.period, self.t0_expected,
                self.supersample_factor, self.exp_time
            )
        )

        state = self.sampler.run_mcmc(pos, nburn, progress=False)
        self.sampler.reset()
        self.sampler.run_mcmc(state, nprod, progress=False)

        self.samples = self.sampler.get_chain(flat=True)
        return self.samples

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return posterior median and 68% credible intervals."""
        if self.samples is None:
            raise RuntimeError("MCMC has not been run yet.")

        labels = ["t0", "rp", "a_rs", "b", "q1", "q2", "f0"]
        q16, med, q84 = np.percentile(self.samples, [16, 50, 84], axis=0)

        results = {}
        for i, lbl in enumerate(labels):
            results[lbl] = {
                "median": float(med[i]),
                "err_minus": float(med[i] - q16[i]),
                "err_plus": float(q84[i] - med[i])
            }
        return results
