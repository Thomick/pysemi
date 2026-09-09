"""LR, max-LR and unif-LR statistics for the empirical-mean SeMI tests."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def compute_log_lr_t(
    mu_t: float,
    mu_t_minus_1: float,
    t: int,
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
) -> float:
    """Single-time log LR_t given cumulative means at t and t-1."""
    N_t = mu_t - ((t - 1) / t) * mu_t_minus_1 - mu / t
    c0 = -0.5 * np.log((n - 1) / n) - (z_star - mu) ** 2 / (2 * (n - 1) * sigma**2)
    c1 = (z_star - mu) * t * n / ((n - 1) * sigma**2)
    c2 = -(t**2) * n / (2 * sigma**2 * (n - 1))
    return c0 + c1 * N_t + c2 * N_t**2


def compute_log_lr_tau(
    empirical_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    tau: int,
) -> float:
    """SeMI* statistic: log LR_τ at known τ (1-indexed)."""
    T = len(empirical_means)
    if not 1 <= tau <= T:
        raise ValueError(f"tau must be in [1, {T}], got {tau}")
    mu_t = empirical_means[tau - 1]
    mu_t_minus_1 = mu if tau == 1 else empirical_means[tau - 2]
    return compute_log_lr_t(mu_t, mu_t_minus_1, tau, mu, sigma, z_star, n)


def compute_log_lr_max(
    empirical_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
) -> Tuple[float, int]:
    """SeMI^max statistic: ``(max_t log LR_t, argmax t)``."""
    T = len(empirical_means)
    log_lrs = np.array(
        [compute_log_lr_tau(empirical_means, mu, sigma, z_star, n, t) for t in range(1, T + 1)]
    )
    best = int(np.argmax(log_lrs))
    return float(log_lrs[best]), best + 1


def compute_log_lr_unif(
    empirical_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
) -> float:
    """SeMI^unif statistic: ``log (1/T) Σ_t LR_t``."""
    T = len(empirical_means)
    log_lrs = np.array(
        [compute_log_lr_tau(empirical_means, mu, sigma, z_star, n, t) for t in range(1, T + 1)]
    )
    return float(logsumexp(log_lrs) - np.log(T))


def simulate_empirical_means(
    mu: float,
    sigma: float,
    n: int,
    T: int,
    z_star: Optional[float] = None,
    tau_star: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Simulate cumulative empirical means under H0 or H1^τ⋆."""
    rng = np.random.default_rng() if rng is None else rng
    means: List[float] = []
    cum_sum = 0.0
    cum_count = 0
    for t in range(1, T + 1):
        if z_star is not None and tau_star is not None and t == tau_star:
            batch = rng.normal(mu, sigma, n - 1)
            batch = np.append(batch, z_star)
        else:
            batch = rng.normal(mu, sigma, n)
        cum_sum += float(np.sum(batch))
        cum_count += n
        means.append(cum_sum / cum_count)
    return means


def simulate_empirical_means_multiple_targets(
    mu: float,
    sigma: float,
    n: int,
    T: int,
    z_included: np.ndarray,
    tau_star: int,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Simulate cumulative means with k₁ targets inserted in batch τ⋆."""
    rng = np.random.default_rng() if rng is None else rng
    k1 = len(z_included)
    if k1 > n:
        raise ValueError(f"Cannot include {k1} targets in batch of size {n}")
    means: List[float] = []
    cum_sum = 0.0
    cum_count = 0
    for t in range(1, T + 1):
        if t == tau_star and k1 > 0:
            batch = np.concatenate([rng.normal(mu, sigma, n - k1), z_included])
        else:
            batch = rng.normal(mu, sigma, n)
        cum_sum += float(np.sum(batch))
        cum_count += n
        means.append(cum_sum / cum_count)
    return means


def semi_star_power_theoretical(
    alpha: float,
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    T: int,
    tau: int,
) -> float:
    """Asymptotic SeMI* power at level α for known τ."""
    m_star = ((z_star - mu) / sigma) ** 2
    sigma_N_h0 = sigma / (tau * np.sqrt(n))
    sigma_N_h1 = sigma * np.sqrt(n - 1) / (tau * n)
    c1 = (z_star - mu) * tau * n / ((n - 1) * sigma**2)
    c2 = -(tau**2) * n / (2 * sigma**2 * (n - 1))
    mu_lr_h0 = -0.5 * np.log((n - 1) / n) - m_star / (2 * (n - 1))
    mu_N_h1 = (z_star - mu) / (tau * n)
    mu_lr_h1 = mu_lr_h0 + c1 * mu_N_h1 + c2 * mu_N_h1**2
    var_lr_h0 = c1**2 * sigma_N_h0**2
    var_lr_h1 = c1**2 * sigma_N_h1**2
    threshold = norm.ppf(1 - alpha) * np.sqrt(var_lr_h0) + mu_lr_h0
    z_score = (threshold - mu_lr_h1) / np.sqrt(var_lr_h1)
    return float(1 - norm.cdf(z_score))


def semi_max_type_i_error(alpha_0: float, T: int) -> float:
    """Bonferroni overall Type-I error for SeMI^max with per-t level α₀ over T candidates."""
    return float(1 - (1 - alpha_0) ** T)


def semi_max_per_t_alpha(overall_alpha: float, T: int) -> float:
    """Per-t α giving overall Type-I error ``overall_alpha`` over T candidates."""
    return float(1 - (1 - overall_alpha) ** (1 / T))
