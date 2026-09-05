"""SeMI* under Laplace noise: Gaussian-Laplace convolution density and DP LR."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm


def laplace_sensitivity_clipping(sigma: float, n: int) -> float:
    """σ/√n: clipped-mean Laplace sensitivity used by the paper."""
    return float(sigma / np.sqrt(n))


def compute_log_prob_convolution(
    z: float,
    mu_G: float,
    sigma_G: float,
    b_L: float,
) -> float:
    """log p(z) for Z = G + L where G ~ N(μ_G, σ_G²) and L ~ Laplace(0, b_L)."""
    if b_L <= 1e-12:
        return float(norm.logpdf(z, loc=mu_G, scale=sigma_G))
    s2_b = sigma_G**2 / b_L
    s2_2b2 = sigma_G**2 / (2 * b_L**2)
    z_mu_b = (z - mu_G) / b_L
    log_term1 = s2_2b2 - z_mu_b
    arg1 = (z - mu_G - s2_b) / sigma_G
    log_phi1 = log_ndtr(arg1)
    log_term2 = s2_2b2 + z_mu_b
    arg2 = (-(z - mu_G) - s2_b) / sigma_G
    log_phi2 = log_ndtr(arg2)
    log_sum = np.logaddexp(log_term1 + log_phi1, log_term2 + log_phi2)
    return float(-np.log(2 * b_L) + log_sum)


def compute_noisy_log_lr_t(
    mu_t: float,
    mu_t_minus_1: float,
    t: int,
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    epsilon: float,
    sensitivity: Optional[float] = None,
) -> float:
    """Single-time log LR_t under Laplace(σ/(ε√n)) noise on the released cumulative means."""
    if t == 1:
        N_t = mu_t - mu / t
    else:
        N_t = mu_t - ((t - 1) / t) * mu_t_minus_1 - mu / t

    if sensitivity is None:
        sensitivity = laplace_sensitivity_clipping(sigma, n)
    b = sensitivity / epsilon
    b_L = b / t

    mu_G0 = 0.0
    sigma_G0 = sigma / (np.sqrt(n) * t)
    mu_G1 = (z_star - mu) / (n * t)
    sigma_G1 = (sigma * np.sqrt(n - 1)) / (n * t)

    log_lh_h1 = compute_log_prob_convolution(N_t, mu_G1, sigma_G1, b_L)
    log_lh_h0 = compute_log_prob_convolution(N_t, mu_G0, sigma_G0, b_L)
    return log_lh_h1 - log_lh_h0


def compute_noisy_log_lr_tau(
    noisy_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    tau: int,
    epsilon: float,
    sensitivity: Optional[float] = None,
) -> float:
    """DP SeMI* statistic: log LR_τ at known τ from observed noisy cumulative means."""
    T = len(noisy_means)
    if not 1 <= tau <= T:
        raise ValueError(f"tau must be in [1, {T}], got {tau}")
    mu_t = noisy_means[tau - 1]
    mu_t_minus_1 = mu if tau == 1 else noisy_means[tau - 2]
    return compute_noisy_log_lr_t(
        mu_t, mu_t_minus_1, tau, mu, sigma, z_star, n, epsilon, sensitivity
    )


def compute_noisy_log_lr_max(
    noisy_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    epsilon: float,
    sensitivity: Optional[float] = None,
) -> Tuple[float, int]:
    """DP SeMI^max statistic: ``(max_t log LR_t, argmax t)``."""
    T = len(noisy_means)
    log_lrs = np.array([
        compute_noisy_log_lr_tau(
            noisy_means, mu, sigma, z_star, n, t, epsilon, sensitivity
        )
        for t in range(1, T + 1)
    ])
    best = int(np.argmax(log_lrs))
    return float(log_lrs[best]), best + 1


def compute_noisy_log_lr_unif(
    noisy_means: Sequence[float],
    mu: float,
    sigma: float,
    z_star: float,
    n: int,
    epsilon: float,
    sensitivity: Optional[float] = None,
) -> float:
    """DP SeMI^unif statistic: ``log (1/T) Σ_t LR_t``."""
    T = len(noisy_means)
    log_lrs = np.array([
        compute_noisy_log_lr_tau(
            noisy_means, mu, sigma, z_star, n, t, epsilon, sensitivity
        )
        for t in range(1, T + 1)
    ])
    return float(logsumexp(log_lrs) - np.log(T))


def simulate_noisy_empirical_means(
    mu: float,
    sigma: float,
    n: int,
    T: int,
    epsilon: float,
    z_star: Optional[float] = None,
    tau_star: Optional[int] = None,
    sensitivity: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Simulate the cumulative means released with Laplace(σ/(ε√n)) noise at each phase."""
    rng = np.random.default_rng() if rng is None else rng
    if sensitivity is None:
        sensitivity = laplace_sensitivity_clipping(sigma, n)
    scale = sensitivity / epsilon
    out: List[float] = []
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
        noisy = cum_sum / cum_count + float(rng.laplace(0.0, scale))
        out.append(noisy)
    return out
