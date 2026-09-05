"""FinalObservation baseline: log LR from the cumulative mean only."""

from typing import Optional

import numpy as np
from scipy.stats import norm


def compute_final_observation_log_lr(
    mu_T: float,
    mu: float,
    sigma: float,
    z_star: float,
    n_total: int,
) -> float:
    """Log LR of the FinalObservation test using the cumulative mean over n·T samples."""
    W_T = mu_T - mu
    c0 = -0.5 * np.log((n_total - 1) / n_total) - (z_star - mu) ** 2 / (
        2 * (n_total - 1) * sigma**2
    )
    c1 = n_total * (z_star - mu) / ((n_total - 1) * sigma**2)
    c2 = -n_total / (2 * (n_total - 1) * sigma**2)
    return c0 + c1 * W_T + c2 * W_T**2


def simulate_final_observation(
    mu: float,
    sigma: float,
    z_star: float,
    n_total: int,
    include_target: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """One sample of the FinalObservation log LR under H0 (default) or H1."""
    rng = np.random.default_rng() if rng is None else rng
    if include_target:
        data = rng.normal(mu, sigma, n_total - 1)
        data = np.append(data, z_star)
    else:
        data = rng.normal(mu, sigma, n_total)
    return compute_final_observation_log_lr(float(np.mean(data)), mu, sigma, z_star, n_total)


def final_observation_power_theoretical(
    alpha: float,
    mu: float,
    sigma: float,
    z_star: float,
    n_total: int,
) -> float:
    """Asymptotic FinalObservation power at level α for total sample size n·T."""
    m_star = ((z_star - mu) / sigma) ** 2
    sigma_mean_h0 = sigma / np.sqrt(n_total)
    sigma_mean_h1 = sigma * np.sqrt(n_total - 1) / n_total
    c1 = n_total * (z_star - mu) / ((n_total - 1) * sigma**2)
    c0 = -0.5 * np.log((n_total - 1) / n_total) - m_star / (2 * (n_total - 1))
    mu_lr_h0 = c0
    mu_lr_h1 = c0 + c1 * (z_star - mu) / n_total
    var_lr_h0 = c1**2 * sigma_mean_h0**2
    var_lr_h1 = c1**2 * sigma_mean_h1**2
    threshold = norm.ppf(1 - alpha) * np.sqrt(var_lr_h0) + mu_lr_h0
    z_score = (threshold - mu_lr_h1) / np.sqrt(var_lr_h1)
    return float(1 - norm.cdf(z_score))
