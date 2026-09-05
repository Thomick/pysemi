"""Plug-in log LR primitive used by all SeMI^SGD variants."""

import numpy as np


def plugin_log_lr_at_tau(
    theta_prev: np.ndarray,
    theta_curr: np.ndarray,
    eta: float,
    n_batch: int,
    mu_g: np.ndarray,
    sigma_inv: np.ndarray,
    delta_g: np.ndarray,
    m_star: float,
) -> float:
    """Plug-in log LR for a single SGD update under the Gaussian-gradient approximation."""
    N = theta_curr - theta_prev + eta * mu_g
    d = len(theta_prev)
    c0 = -0.5 * d * np.log((n_batch - 1) / n_batch) - m_star / (2 * (n_batch - 1))
    c1 = -n_batch / ((n_batch - 1) * eta) * (sigma_inv @ delta_g)
    c2_factor = -n_batch / (2 * (n_batch - 1) * (eta**2))
    return float(c0 + np.dot(c1, N) + c2_factor * (N @ (sigma_inv @ N)))


def phase_effective_lr(lr: float, K: int, batch_size: int, pool_size: int) -> float:
    """Phase-adapted ``η_eff = K · (batch_size / pool_size) · lr``."""
    return float(K * (batch_size / pool_size) * lr)
