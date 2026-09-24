"""
Correlated stellar noise modeling via Gaussian Processes (celerite2).
Includes stochastically driven Simple Harmonic Oscillator (SHOTerm) kernels.
"""

from typing import Tuple, Optional
import numpy as np


class CeleriteGPNoiseModel:
    """
    Gaussian Process noise wrapper using celerite2 for fast O(N) evaluation.
    Couples stellar activity and instrumental drifts with transit forward models.
    """

    def __init__(
        self,
        time: np.ndarray,
        yerr: np.ndarray,
        transit_duration: Optional[float] = None
    ):
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        self.yerr = np.ascontiguousarray(yerr, dtype=np.float64)
        self.transit_duration = float(transit_duration) if transit_duration is not None else None

    def _apply_rho_safeguard(
        self,
        rho_gp: float,
        transit_duration: Optional[float] = None
    ) -> float:
        """
        Safeguard rho_gp against absorbing planetary transit signals.
        Enforces rho_gp >= 2 * transit_duration if provided,
        or a reasonable minimum threshold (0.01) to protect transit dips.
        """
        dur = transit_duration if transit_duration is not None else self.transit_duration
        if dur is not None and dur > 0:
            min_rho = 2.0 * float(dur)
        else:
            min_rho = 0.01  # Safe physical lower bound to avoid absorbing transit signals

        return max(float(rho_gp), float(min_rho))

    def build_sho_term(
        self,
        sigma_gp: float,
        rho_gp: float,
        q0: float = 1.0 / np.sqrt(2.0),
        transit_duration: Optional[float] = None
    ):
        """
        Build a celerite2 SHOTerm kernel representing stellar granulation.

        Parameters
        ----------
        sigma_gp : float
            Standard deviation / amplitude of the GP variability.
        rho_gp : float
            Characteristic timescale (e.g. granulation timescale).
        q0 : float
            Quality factor (default critically damped 1/sqrt(2)).
        transit_duration : Optional[float]
            Typical transit duration to safeguard rho_gp.
        """
        rho_gp = self._apply_rho_safeguard(rho_gp, transit_duration)

        try:
            import celerite2
            from celerite2 import terms

            w0 = 2.0 * np.pi / rho_gp
            s0 = (sigma_gp ** 2) / (w0 * q0)
            kernel = terms.SHOTerm(S0=s0, w0=w0, Q=q0)
            gp = celerite2.GaussianProcess(kernel, mean=0.0)
            gp.compute(self.time, yerr=self.yerr)
            return gp
        except ImportError:
            raise ImportError(
                "celerite2 is required for Gaussian Process noise modeling. "
                "Install with `pip install celerite2` or `pip install astro-exo[gp]`."
            )

    def compute_gp_log_likelihood(
        self,
        residuals: np.ndarray,
        sigma_gp: float,
        rho_gp: float,
        transit_duration: Optional[float] = None
    ) -> float:
        """
        Evaluate exact marginal log-likelihood for residuals given GP hyper-parameters.
        ln L = -0.5 * (r^T K^-1 r + ln|K| + N ln(2pi))
        Uses celerite2 if available, or dense exact Cholesky/solver fallback for small samples
        with analytical critically damped SHO kernel (Q = 1/sqrt(2)).
        """
        rho_gp = self._apply_rho_safeguard(rho_gp, transit_duration)

        try:
            gp = self.build_sho_term(sigma_gp, rho_gp, transit_duration=transit_duration)
            return float(gp.log_likelihood(residuals))
        except ImportError:
            # Analytical critically damped SHO kernel (Q = 1 / sqrt(2)):
            # k(dt) = sigma_gp^2 * exp(-w0 * dt / sqrt(2)) * [cos(w0 * dt / sqrt(2)) + sin(w0 * dt / sqrt(2))]
            # where w0 = 2 * pi / rho_gp
            dt = np.abs(self.time[:, None] - self.time[None, :])
            w0 = 2.0 * np.pi / rho_gp
            eta = (w0 * dt) / np.sqrt(2.0)
            k = (sigma_gp ** 2) * np.exp(-eta) * (np.cos(eta) + np.sin(eta))
            k += np.diag(self.yerr ** 2 + 1e-14)
            sign, logdet = np.linalg.slogdet(k)
            if sign <= 0:
                return -np.inf
            try:
                quad = residuals @ np.linalg.solve(k, residuals)
                n = len(residuals)
                return float(-0.5 * (quad + logdet + n * np.log(2.0 * np.pi)))
            except np.linalg.LinAlgError:
                return -np.inf
