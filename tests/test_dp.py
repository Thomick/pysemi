"""Tests for the DP (Laplace-noisy) SeMI* / SeMI^max / SeMI^unif."""

import numpy as np

import pysemi
from pysemi.core.dp import (
    compute_log_prob_convolution,
    compute_noisy_log_lr_max,
    compute_noisy_log_lr_t,
    compute_noisy_log_lr_tau,
    compute_noisy_log_lr_unif,
    laplace_sensitivity_clipping,
    simulate_noisy_empirical_means,
)
from pysemi.core.noisy_mean import compute_log_lr_tau


def test_convolution_log_pdf_matches_gaussian_in_no_noise_limit():
    z, mu, sigma = 0.3, 0.1, 0.5
    from scipy.stats import norm

    truth = float(norm.logpdf(z, loc=mu, scale=sigma))
    assert np.isclose(compute_log_prob_convolution(z, mu, sigma, b_L=1e-13), truth)


def test_noisy_log_lr_t_finite():
    for eps in [0.1, 1.0, 10.0]:
        v = compute_noisy_log_lr_t(
            mu_t=0.02,
            mu_t_minus_1=0.0,
            t=3,
            mu=0.0,
            sigma=1.0,
            z_star=3.0,
            n=50,
            epsilon=eps,
        )
        assert np.isfinite(v)


def test_sensitivity_default():
    assert np.isclose(laplace_sensitivity_clipping(1.0, 100), 0.1)


def test_dp_semi_star_h1_greater_than_h0_on_average():
    rng = np.random.default_rng(0)
    mu, sigma, z_star, n, T, tau, eps = 0.0, 1.0, 5.0, 30, 6, 3, 5.0
    h0 = [
        compute_noisy_log_lr_tau(
            simulate_noisy_empirical_means(mu, sigma, n, T, eps, rng=rng),
            mu,
            sigma,
            z_star,
            n,
            tau,
            eps,
        )
        for _ in range(300)
    ]
    h1 = [
        compute_noisy_log_lr_tau(
            simulate_noisy_empirical_means(
                mu, sigma, n, T, eps, z_star=z_star, tau_star=tau, rng=rng
            ),
            mu,
            sigma,
            z_star,
            n,
            tau,
            eps,
        )
        for _ in range(300)
    ]
    assert np.mean(h1) > np.mean(h0)


def test_dp_semi_max_finite():
    rng = np.random.default_rng(1)
    means = simulate_noisy_empirical_means(0.0, 1.0, 30, 5, epsilon=2.0, rng=rng)
    stat, best = compute_noisy_log_lr_max(means, 0.0, 1.0, 3.0, 30, 2.0)
    assert np.isfinite(stat)
    assert 1 <= best <= 5


def test_dp_semi_unif_finite():
    rng = np.random.default_rng(2)
    means = simulate_noisy_empirical_means(0.0, 1.0, 30, 5, epsilon=2.0, rng=rng)
    assert np.isfinite(compute_noisy_log_lr_unif(means, 0.0, 1.0, 3.0, 30, 2.0))


def test_dp_class_simulate_and_score():
    rng = np.random.default_rng(3)
    test = pysemi.SeMIStarDP(mu=0.0, sigma=1.0, z_star=4.0, n=40, epsilon=1.0)
    means = test.simulate(T=5, tau_star=2, rng=rng)
    assert np.isfinite(test.log_lr(means, tau=2))
    assert np.isfinite(test.log_lr_max(means))
    assert np.isfinite(test.log_lr_unif(means))


def test_dp_collapses_to_clean_when_epsilon_large():
    """As ε → ∞, the DP LR should approach the clean LR."""
    rng = np.random.default_rng(4)
    mu, sigma, z_star, n, T, tau = 0.0, 1.0, 3.0, 30, 5, 3
    clean_means = pysemi.SeMIStar(mu, sigma, z_star, n).simulate(T, tau_star=tau, rng=rng)
    dp_lr = compute_noisy_log_lr_t(
        clean_means[tau - 1],
        clean_means[tau - 2],
        tau,
        mu,
        sigma,
        z_star,
        n,
        epsilon=1e6,
    )
    assert np.isclose(dp_lr, compute_log_lr_tau(clean_means, mu, sigma, z_star, n, tau), atol=1e-3)
