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

    def __init__(self, time: np.ndarray, yerr: np.ndarray):
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        self.yerr = np.ascontiguousarray(yerr, dtype=np.float64)

    def build_sho_term(self, sigma_gp: float, rho_gp: float, q0: float = 1.0 / np.sqrt(2.0)):
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
        """
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
        rho_gp: float
    ) -> float:
        """
        Evaluate exact marginal log-likelihood for residuals given GP hyper-parameters.
        ln L = -0.5 * (r^T K^-1 r + ln|K| + N ln(2pi))
        """
        gp = self.build_sho_term(sigma_gp, rho_gp)
        return float(gp.log_likelihood(residuals))
