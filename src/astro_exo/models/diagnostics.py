"""
Statistical diagnostics for MCMC convergence and autocorrelation.
Includes Gelman-Rubin R-hat, effective sample size (ESS), and autocorrelation time.
"""

from typing import Dict, Any, Optional
import warnings
import numpy as np


class ConvergenceWarning(UserWarning):
    """Warning emitted when MCMC chains fail convergence criteria."""
    pass



def compute_gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """
    Computes Gelman-Rubin potential scale reduction factor (R-hat).

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (n_steps, n_chains, n_params) or (n_steps, n_chains).

    Returns
    -------
    r_hat : np.ndarray
        R-hat metric for each parameter. Values close to 1.0 (< 1.05) indicate convergence.
    """
    if chains.ndim == 2:
        chains = chains[:, :, np.newaxis]

    n_steps, n_chains, n_params = chains.shape

    if n_chains < 2 or n_steps < 4:
        return np.ones(n_params)

    # Chain means and overall mean
    chain_means = np.mean(chains, axis=0)  # (n_chains, n_params)
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # Between-chain variance B/n
    b_div_n = np.sum((chain_means - overall_mean) ** 2, axis=0) / (n_chains - 1)

    # Within-chain variance W
    chain_vars = np.var(chains, axis=0, ddof=1)  # (n_chains, n_params)
    w = np.mean(chain_vars, axis=0)  # (n_params,)

    # Avoid division by zero
    w = np.where(w == 0, 1e-12, w)

    # Estimated marginal posterior variance
    var_est = ((n_steps - 1) / n_steps) * w + b_div_n
    r_hat = np.sqrt(np.maximum(1.0, var_est / w))

    return r_hat


def compute_effective_sample_size(chain_flat: np.ndarray, tau: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute effective sample size N_eff = N_total / (2 * tau).

    Parameters
    ----------
    chain_flat : np.ndarray
        Array of shape (n_samples, n_params) or (n_samples,).
    tau : Optional[np.ndarray]
        Integrated autocorrelation time for each parameter.

    Returns
    -------
    n_eff : np.ndarray
        Effective number of independent samples. Returns np.nan for parameters
        if tau is not available, <= 0, or contains NaN (unconverged chain).
    """
    if chain_flat.ndim == 1:
        chain_flat = chain_flat[:, np.newaxis]

    n_samples, n_params = chain_flat.shape
    if tau is None or np.any(np.isnan(tau)) or np.any(np.asarray(tau) <= 0):
        warnings.warn(
            "MCMC chain has not converged according to Foreman-Mackey autocorrelation criterion; "
            "effective sample size (ESS) cannot be reliably computed.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return np.full(n_params, np.nan)

    tau_arr = np.asarray(tau, dtype=np.float64)
    n_eff = n_samples / (2.0 * np.maximum(1.0, tau_arr))
    return np.round(n_eff).astype(float)
