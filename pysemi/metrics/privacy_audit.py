"""Empirical lower bounds on ε for (ε, δ)-DP audits."""

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .roc import compute_empirical_roc


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.98,
) -> Tuple[float, float]:
    """Two-sided Clopper-Pearson exact binomial CI at level ``confidence``."""
    alpha = 1 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = float(stats.beta.ppf(alpha / 2, successes, trials - successes + 1))
    if successes == trials:
        upper = 1.0
    else:
        upper = float(stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def compute_epsilon_lower_bound_cp(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float = 1e-4,
    n_h0: Optional[int] = None,
    n_h1: Optional[int] = None,
    confidence: float = 0.98,
    min_rate_floor: float = 0.0,
) -> float:
    """Pointwise Clopper-Pearson lower bound on ε. Falls back to a point estimate when sizes are None."""
    if n_h0 is None or n_h1 is None:
        return _point_epsilon(fpr, tpr, delta, min_rate_floor)

    if min_rate_floor > 0:
        keep = (fpr >= min_rate_floor) & (tpr >= min_rate_floor)
        fpr, tpr = fpr[keep], tpr[keep]
    if len(fpr) == 0:
        return 0.0

    min_rate = 0.1 / n_h0
    fp = np.round(fpr * n_h0).astype(int)
    tn = n_h0 - fp
    tp = np.round(tpr * n_h1).astype(int)
    fn = n_h1 - tp
    alpha = 1 - confidence

    with np.errstate(invalid="ignore", divide="ignore"):
        fpr_upper = np.where(fp == n_h0, 1.0, stats.beta.ppf(1 - alpha / 2, fp + 1, n_h0 - fp))
        tpr_lower = np.where(tp == 0, 0.0, stats.beta.ppf(alpha / 2, tp, n_h1 - tp + 1))
        fnr_lower = np.where(fn == 0, 0.0, stats.beta.ppf(alpha / 2, fn, n_h1 - fn + 1))
        tnr_upper = np.where(tn == n_h0, 1.0, stats.beta.ppf(1 - alpha / 2, tn + 1, n_h0 - tn))

    bounds = []
    mask1 = (fpr_upper > min_rate) & (tpr_lower > delta)
    if np.any(mask1):
        bounds.append(np.max(np.log((tpr_lower[mask1] - delta) / fpr_upper[mask1])))
    mask2 = (tnr_upper > min_rate) & (fnr_lower > delta)
    if np.any(mask2):
        bounds.append(np.max(np.log((fnr_lower[mask2] - delta) / tnr_upper[mask2])))
    if not bounds:
        return 0.0
    return max(0.0, float(np.max(bounds)))


def _point_epsilon(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float = 1e-4,
    min_rate_floor: float = 0.0,
) -> float:
    """Naive point estimate without CIs; for reference only."""
    if min_rate_floor > 0:
        keep = (fpr >= min_rate_floor) & (tpr >= min_rate_floor)
        fpr, tpr = fpr[keep], tpr[keep]
        if len(fpr) == 0:
            return 0.0
    mask = (fpr > 0) & (tpr > delta)
    eps1 = np.log((tpr[mask] - delta) / fpr[mask]) if np.any(mask) else np.array([])
    mask_c = (1 - fpr > 0) & (1 - tpr > delta)
    eps2 = np.log(((1 - tpr)[mask_c] - delta) / (1 - fpr)[mask_c]) if np.any(mask_c) else np.array([])
    all_eps = np.concatenate([eps1, eps2])
    if len(all_eps) == 0:
        return 0.0
    return float(np.max(all_eps))


def compute_epsilon_lower_bound_dkw(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float = 1e-4,
    n_h0: Optional[int] = None,
    n_h1: Optional[int] = None,
    xi: float = 0.02,
    min_rate_floor: float = 0.0,
) -> float:
    """Uniform DKW lower bound on ε that holds simultaneously at all thresholds."""
    if n_h0 is None or n_h1 is None:
        raise ValueError("n_h0 and n_h1 are required for the DKW bound")
    if min_rate_floor > 0:
        keep = (fpr >= min_rate_floor) & (tpr >= min_rate_floor)
        fpr, tpr = fpr[keep], tpr[keep]
        if len(fpr) == 0:
            return 0.0

    dkw_alpha = np.sqrt(np.log(2 / xi) / (2 * n_h0))
    dkw_beta = np.sqrt(np.log(2 / xi) / (2 * n_h1))
    fnr = 1 - tpr
    alpha_bar = fpr + dkw_alpha
    beta_bar = fnr + dkw_beta

    bounds = []
    mask1 = (beta_bar > 0) & ((1 - delta - alpha_bar) > 0)
    if np.any(mask1):
        bounds.append(np.max(np.log((1 - delta - alpha_bar[mask1]) / beta_bar[mask1])))
    mask2 = (alpha_bar > 0) & ((1 - delta - beta_bar) > 0)
    if np.any(mask2):
        bounds.append(np.max(np.log((1 - delta - beta_bar[mask2]) / alpha_bar[mask2])))
    if not bounds:
        return 0.0
    return max(0.0, float(np.max(bounds)))


def compute_epsilon_lower_bound_cp_stratified(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float = 1e-4,
    n_per_stratum: Optional[int] = None,
    T: int = 1,
    confidence: float = 0.98,
    min_rate_floor: float = 0.0,
) -> float:
    """Stratified Clopper-Pearson bound for a τ-informed adversary."""
    if n_per_stratum is None:
        raise ValueError("n_per_stratum is required for the stratified CP bound")
    return compute_epsilon_lower_bound_cp(
        fpr, tpr, delta=delta,
        n_h0=n_per_stratum, n_h1=n_per_stratum,
        confidence=confidence,
        min_rate_floor=min_rate_floor,
    )


def compute_epsilon_lower_bound_dkw_stratified(
    fpr: np.ndarray,
    tpr: np.ndarray,
    delta: float = 1e-4,
    n_per_stratum: Optional[int] = None,
    T: int = 1,
    xi: float = 0.02,
    min_rate_floor: float = 0.0,
) -> float:
    """Union-bounded DKW for τ-stratified ROCs (per-stratum n_τ, level ξ/2T)."""
    if n_per_stratum is None:
        raise ValueError("n_per_stratum is required for the stratified DKW bound")
    return compute_epsilon_lower_bound_dkw(
        fpr, tpr, delta=delta,
        n_h0=n_per_stratum, n_h1=n_per_stratum,
        xi=xi / (2 * max(1, T)),
        min_rate_floor=min_rate_floor,
    )


def bootstrap_epsilon_samples(
    h0: np.ndarray,
    h1: np.ndarray,
    delta: float = 1e-4,
    confidence: float = 0.98,
    n_bootstrap: int = 200,
    seed: int = 42,
    min_rate_floor: float = 0.0,
    bound_type: str = "cp",
    dkw_xi: float = 0.02,
    stratified_T: int = 1,
) -> np.ndarray:
    """Raw bootstrap distribution of ε lower-bound estimates."""
    rng = np.random.RandomState(seed)
    n = len(h0)
    n_eff = max(1, n // max(1, stratified_T))
    out = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx0 = rng.choice(n, n, replace=True)
        idx1 = rng.choice(n, n, replace=True)
        fpr_b, tpr_b = compute_empirical_roc(h0[idx0], h1[idx1])
        if bound_type == "dkw":
            out[b] = compute_epsilon_lower_bound_dkw(
                fpr_b, tpr_b, delta, n_eff, n_eff,
                xi=dkw_xi / (2 * max(1, stratified_T)),
                min_rate_floor=min_rate_floor,
            )
        else:
            out[b] = compute_epsilon_lower_bound_cp(
                fpr_b, tpr_b, delta, n_eff, n_eff, confidence,
                min_rate_floor=min_rate_floor,
            )
    return out


def bootstrap_epsilon_ci(
    h0: np.ndarray,
    h1: np.ndarray,
    delta: float = 1e-4,
    confidence: float = 0.98,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
    min_rate_floor: float = 0.0,
    bound_type: str = "cp",
    dkw_xi: float = 0.02,
    stratified_T: int = 1,
) -> Tuple[float, float, float]:
    """(median, lo, hi) of the bootstrap ε lower-bound distribution."""
    samples = bootstrap_epsilon_samples(
        h0, h1, delta, confidence, n_bootstrap, seed,
        min_rate_floor, bound_type, dkw_xi, stratified_T,
    )
    a = (1 - ci_level) / 2
    return (
        float(np.median(samples)),
        float(np.percentile(samples, 100 * a)),
        float(np.percentile(samples, 100 * (1 - a))),
    )


def subsample_epsilon_samples(
    h0: np.ndarray,
    h1: np.ndarray,
    delta: float = 1e-4,
    confidence: float = 0.98,
    n_iters: int = 200,
    seed: int = 42,
    min_rate_floor: float = 0.0,
    bound_type: str = "cp",
    dkw_xi: float = 0.02,
    stratified_T: int = 1,
    subsample_frac: float = 0.9,
) -> np.ndarray:
    """Subsample-without-replacement ε lower-bound distribution.

    Used when the with-replacement bootstrap would inflate the max-over-thresholds
    estimator (winner's curse); this draw is centered on the point estimate.
    """
    rng = np.random.RandomState(seed)
    n = len(h0)
    n_eff = max(1, n // max(1, stratified_T))
    m = max(10, int(subsample_frac * n))
    out = np.empty(n_iters)
    for b in range(n_iters):
        idx0 = rng.choice(n, m, replace=False)
        idx1 = rng.choice(n, m, replace=False)
        fpr_b, tpr_b = compute_empirical_roc(h0[idx0], h1[idx1])
        if bound_type == "dkw":
            out[b] = compute_epsilon_lower_bound_dkw(
                fpr_b, tpr_b, delta, n_eff, n_eff,
                xi=dkw_xi / (2 * max(1, stratified_T)),
                min_rate_floor=min_rate_floor,
            )
        else:
            out[b] = compute_epsilon_lower_bound_cp(
                fpr_b, tpr_b, delta, n_eff, n_eff, confidence,
                min_rate_floor=min_rate_floor,
            )
    return out


def subsample_epsilon_ci(
    h0: np.ndarray,
    h1: np.ndarray,
    delta: float = 1e-4,
    confidence: float = 0.98,
    n_iters: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
    min_rate_floor: float = 0.0,
    bound_type: str = "cp",
    dkw_xi: float = 0.02,
    stratified_T: int = 1,
    subsample_frac: float = 0.9,
) -> Tuple[float, float, float]:
    """(median, lo, hi) of the subsample ε lower-bound distribution."""
    samples = subsample_epsilon_samples(
        h0, h1, delta, confidence, n_iters, seed,
        min_rate_floor, bound_type, dkw_xi, stratified_T, subsample_frac,
    )
    a = (1 - ci_level) / 2
    return (
        float(np.median(samples)),
        float(np.percentile(samples, 100 * a)),
        float(np.percentile(samples, 100 * (1 - a))),
    )
