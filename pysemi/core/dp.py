"""SeMI* under a Laplace-noised empirical-mean mechanism.

The mechanism releases, at each phase t, the cumulative mean of the data seen so
far with independent Laplace noise added once per batch and carried forward,

    μ̂_t = (Σ_{i≤t} Σ_{j≤n} X_ij + Σ_{i≤t} ξ_i) / (nt),    ξ_i ~ Lap(n·Δ/ε),

where Δ is the sensitivity of a single batch *mean*. Each record enters exactly
one batch, so the released sequence is a post-processing of T independently
noised, disjoint batch sums and the whole sequence is ε-DP by parallel
composition. The statistic depends on the observations only through
Ȳ_τ = μ̂_τ − ((τ−1)/τ)·μ̂_{τ−1} = X̄_τ/τ + ξ_τ/(nτ), a Gaussian plus a Laplace,
whence the convolution density below.

This analysis is not in the SeMI paper, which covers the noiseless mechanism and
the DP-SGD audits. It is a library extension.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm


def laplace_sensitivity_clipping(sigma: float, n: int) -> float:
    """σ/√n, the batch-mean noise scale used by the examples.

    This is a calibration convention rather than a derived sensitivity: it sets the
    noise to the standard deviation of a clean batch mean, so ε reads as a
    signal-to-noise ratio. For an actual ε-DP guarantee pass the sensitivity implied
    by your clipping range, `laplace_sensitivity_range`.
    """
    return float(sigma / np.sqrt(n))


def laplace_sensitivity_range(clip_range: float, n: int) -> float:
    """Sensitivity of a batch mean of n values confined to an interval of width ``clip_range``."""
    if not np.isfinite(clip_range) or clip_range <= 0.0:
        raise ValueError(f"clip_range must be positive and finite, got {clip_range!r}")
    if n < 2:
        raise ValueError(f"n (batch size) must be ≥ 2, got {n!r}")
    return float(clip_range / n)


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
    log_lrs = np.array(
        [
            compute_noisy_log_lr_tau(noisy_means, mu, sigma, z_star, n, t, epsilon, sensitivity)
            for t in range(1, T + 1)
        ]
    )
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
    log_lrs = np.array(
        [
            compute_noisy_log_lr_tau(noisy_means, mu, sigma, z_star, n, t, epsilon, sensitivity)
            for t in range(1, T + 1)
        ]
    )
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
    """Simulate the released sequence μ̂_t = (Σ_{i≤t} Σ_j X_ij + Σ_{i≤t} ξ_i) / (nt), ξ_i ~ Lap(n·Δ/ε)."""
    rng = np.random.default_rng() if rng is None else rng
    if sensitivity is None:
        sensitivity = laplace_sensitivity_clipping(sigma, n)
    scale = sensitivity / epsilon
    out: List[float] = []
    cum_sum = 0.0
    cum_count = 0
    cum_noise = 0.0
    for t in range(1, T + 1):
        if z_star is not None and tau_star is not None and t == tau_star:
            batch = rng.normal(mu, sigma, n - 1)
            batch = np.append(batch, z_star)
        else:
            batch = rng.normal(mu, sigma, n)
        cum_sum += float(np.sum(batch))
        cum_count += n
        # Drawing fresh noise per release instead of carrying it forward would leave
        # N_t a difference of two Laplace variables, which is not the density
        # compute_noisy_log_lr_t evaluates, and would cost parallel composition.
        cum_noise += float(rng.laplace(0.0, scale))
        out.append(cum_sum / cum_count + cum_noise / t)
    return out
