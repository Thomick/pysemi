"""Calibration properties measured during the 0.4.0 release check.

Each test here fails if a regression reintroduces a defect that shipped in 0.3.0
or if the closed-form power drifts from the exact error expressions.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

import pysemi
from pysemi.core.dp import laplace_sensitivity_clipping
from pysemi.metrics import (
    calibrate_threshold_grid,
    compute_empirical_roc,
    compute_epsilon_lower_bound_cp,
    compute_epsilon_lower_bound_cp_grid,
    compute_epsilon_lower_bound_dkw,
)


def _exact_errors(gamma, mu, sigma, z_star, n):
    """Exact (alpha, beta) of SeMI* at threshold gamma, from the paper's error lemma."""
    c = n / (n - 1)
    m_star = ((z_star - mu) / sigma) ** 2
    if gamma > 0.5 * (m_star + np.log(c)):
        return 0.0, 1.0
    a = np.sqrt(m_star * n)
    b = np.sqrt((n - 1) * (m_star + np.log(c) - 2 * gamma))
    alpha = norm.cdf(a + b) - norm.cdf(a - b)
    beta = norm.cdf(a / np.sqrt(c) - b * np.sqrt(c)) + norm.cdf(-a / np.sqrt(c) - b * np.sqrt(c))
    return float(alpha), float(beta)


@pytest.mark.parametrize("z_star", [1.0, 2.0, 4.0, 6.0])
def test_semi_star_power_matches_exact_error_expressions(z_star):
    """The closed-form power tracks the exact Type-II error at the exact level-alpha threshold."""
    mu, sigma, n, T, tau, alpha = 0.0, 1.0, 50, 10, 5, 0.05
    g_max = 0.5 * (((z_star - mu) / sigma) ** 2 + np.log(n / (n - 1)))
    gamma = brentq(lambda g: _exact_errors(g, mu, sigma, z_star, n)[0] - alpha, -50.0, g_max)
    exact_power = 1 - _exact_errors(gamma, mu, sigma, z_star, n)[1]
    approx = pysemi.SeMIStar(mu, sigma, z_star, n).power(alpha, T, tau)
    assert abs(approx - exact_power) < 0.005


def test_dp_increment_noise_shrinks_with_t():
    """The released sequence must carry one noise draw per batch, not one per release.

    Drawing fresh noise per release leaves the increment N_t contaminated by two
    Laplace variables whose scale does not fall with t, which is what shipped in
    0.3.0 and which breaks the density the DP statistic evaluates.
    """
    mu, sigma, n, T, eps = 0.0, 1.0, 50, 10, 1.0
    b = laplace_sensitivity_clipping(sigma, n) / eps
    dp = pysemi.SeMIStarDP(mu, sigma, 4.0, n, epsilon=eps)
    rng = np.random.default_rng(0)

    increments = {t: [] for t in range(1, T + 1)}
    for _ in range(4000):
        y = dp.simulate(T, rng=rng)
        for t in range(1, T + 1):
            prev = mu if t == 1 else y[t - 2]
            increments[t].append(y[t - 1] - ((t - 1) / t) * prev - mu / t)

    for t in (1, 2, 5, 10):
        expected = np.sqrt((sigma / (np.sqrt(n) * t)) ** 2 + 2 * (b / t) ** 2)
        assert np.std(increments[t]) == pytest.approx(expected, rel=0.08)


def test_dp_semi_star_holds_its_level_and_power_is_tau_invariant():
    """Thresholding at the H0 quantile holds the nominal level, and power does not move with tau."""
    mu, sigma, n, T, eps, z_star = 0.0, 1.0, 50, 10, 1.0, 4.0
    dp = pysemi.SeMIStarDP(mu, sigma, z_star, n, epsilon=eps)
    rng = np.random.default_rng(1)
    powers = []
    for tau in (2, 5, 9):
        cal = np.array([dp.log_lr(dp.simulate(T, rng=rng), tau) for _ in range(4000)])
        thr = np.quantile(cal, 0.95)
        h0 = np.array([dp.log_lr(dp.simulate(T, rng=rng), tau) for _ in range(4000)])
        h1 = np.array([dp.log_lr(dp.simulate(T, tau_star=tau, rng=rng), tau) for _ in range(4000)])
        assert np.mean(h0 >= thr) == pytest.approx(0.05, abs=0.015)
        powers.append(np.mean(h1 >= thr))
    assert max(powers) - min(powers) < 0.03


@pytest.mark.parametrize("bound", ["cp", "cp_grid", "dkw"])
def test_epsilon_lower_bounds_do_not_exceed_a_known_epsilon(bound):
    """Auditing Lap(0, 1) against Lap(1, 1), which is exactly (1, 0)-indistinguishable."""
    eps_true, n, reps = 1.0, 1000, 60
    rng = np.random.default_rng(2)
    breaches = 0
    for _ in range(reps):
        s0 = rng.laplace(0.0, 1.0 / eps_true, n)
        s1 = rng.laplace(1.0, 1.0 / eps_true, n)
        if bound == "cp_grid":
            cal = np.concatenate([rng.laplace(0.0, 1.0, n // 2), rng.laplace(1.0, 1.0, n // 2)])
            grid = calibrate_threshold_grid(cal, 99)
            est = compute_epsilon_lower_bound_cp_grid(s0, s1, grid, delta=0.0, xi=0.02)
        else:
            fpr, tpr = compute_empirical_roc(s0, s1)
            if bound == "cp":
                est = compute_epsilon_lower_bound_cp(fpr, tpr, 0.0, n, n, confidence=0.98)
            else:
                est = compute_epsilon_lower_bound_dkw(fpr, tpr, 0.0, n, n, xi=0.02)
        breaches += est > eps_true
    assert breaches <= 3


def test_cp_bound_tightens_towards_the_truth_with_more_data():
    """The bound is conservative but not vacuous, and improves with the sample size."""
    rng = np.random.default_rng(3)
    estimates = []
    for n in (500, 5000):
        s0 = rng.laplace(0.0, 1.0, n)
        s1 = rng.laplace(1.0, 1.0, n)
        fpr, tpr = compute_empirical_roc(s0, s1)
        estimates.append(compute_epsilon_lower_bound_cp(fpr, tpr, 0.0, n, n, confidence=0.98))
    assert 0.0 < estimates[0] < estimates[1] < 1.0
