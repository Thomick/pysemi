"""Tests for the empirical-mean SeMI core (SeMI*, SeMI^max, SeMI^unif)."""

import numpy as np
import pytest

import pysemi
from pysemi.core.noisy_mean import (
    compute_log_lr_max,
    compute_log_lr_t,
    compute_log_lr_tau,
    compute_log_lr_unif,
    semi_max_per_t_alpha,
    semi_max_type_i_error,
    semi_star_power_theoretical,
    simulate_empirical_means,
)


def test_log_lr_t_runs():
    lr = compute_log_lr_t(mu_t=0.05, mu_t_minus_1=0.0, t=2, mu=0.0, sigma=1.0, z_star=2.0, n=100)
    assert np.isfinite(lr)


def test_semi_star_h1_greater_than_h0_on_average():
    rng = np.random.default_rng(0)
    mu, sigma, z_star, n, T, tau = 0.0, 1.0, 3.0, 50, 10, 5
    h0 = [
        compute_log_lr_tau(simulate_empirical_means(mu, sigma, n, T, rng=rng), mu, sigma, z_star, n, tau)
        for _ in range(500)
    ]
    h1 = [
        compute_log_lr_tau(
            simulate_empirical_means(mu, sigma, n, T, z_star=z_star, tau_star=tau, rng=rng),
            mu, sigma, z_star, n, tau,
        )
        for _ in range(500)
    ]
    assert np.mean(h1) > np.mean(h0)


def test_semi_max_argmax_better_than_uniform():
    rng = np.random.default_rng(1)
    mu, sigma, z_star, n, T, tau_star = 0.0, 1.0, 6.0, 30, 8, 4
    correct = 0
    for _ in range(200):
        means = simulate_empirical_means(mu, sigma, n, T, z_star=z_star, tau_star=tau_star, rng=rng)
        _, best = compute_log_lr_max(means, mu, sigma, z_star, n)
        if best == tau_star:
            correct += 1
    assert correct / 200 > 1.0 / T + 0.1


def test_semi_unif_finite():
    rng = np.random.default_rng(2)
    means = simulate_empirical_means(0.0, 1.0, 20, 5, rng=rng)
    assert np.isfinite(compute_log_lr_unif(means, 0.0, 1.0, 2.0, 20))


def test_semi_unif_within_log_T_of_semi_max():
    """Paper relation: max log LR_t − log T ≤ log mean exp(log LR_t) ≤ max log LR_t."""
    rng = np.random.default_rng(3)
    T = 6
    means = simulate_empirical_means(0.0, 1.0, 30, T, z_star=4.0, tau_star=3, rng=rng)
    max_lr, _ = compute_log_lr_max(means, 0.0, 1.0, 4.0, 30)
    unif_lr = compute_log_lr_unif(means, 0.0, 1.0, 4.0, 30)
    assert max_lr - np.log(T) - 1e-9 <= unif_lr <= max_lr + 1e-9


def test_power_in_unit_interval():
    p = semi_star_power_theoretical(0.05, 0.0, 1.0, 3.0, 50, 10, 5)
    assert 0.0 <= p <= 1.0


def test_semi_max_alpha_roundtrip():
    T = 10
    overall = 0.05
    per_t = semi_max_per_t_alpha(overall, T)
    assert np.isclose(semi_max_type_i_error(per_t, T), overall)


def test_semistar_class_matches_functional():
    rng = np.random.default_rng(4)
    test = pysemi.SeMIStar(mu=0.0, sigma=1.0, z_star=2.0, n=20)
    means = test.simulate(T=6, tau_star=3, rng=rng)
    assert np.isclose(
        test.log_lr(means, tau=3),
        compute_log_lr_tau(means, 0.0, 1.0, 2.0, 20, 3),
    )


def test_semi_max_class():
    rng = np.random.default_rng(5)
    test = pysemi.SeMIMax(mu=0.0, sigma=1.0, z_star=4.0, n=30)
    means = test.simulate(T=7, tau_star=4, rng=rng)
    assert np.isfinite(test.log_lr(means))
    assert 1 <= test.best_tau(means) <= 7


def test_semi_unif_class():
    rng = np.random.default_rng(6)
    test = pysemi.SeMIUnif(mu=0.0, sigma=1.0, z_star=4.0, n=30)
    means = test.simulate(T=7, tau_star=4, rng=rng)
    assert np.isfinite(test.log_lr(means))


def test_semi_star_decide():
    test = pysemi.SeMIStar(mu=0.0, sigma=1.0, z_star=5.0, n=20)
    rng = np.random.default_rng(7)
    means = test.simulate(T=5, tau_star=3, rng=rng)
    assert test.decide(means, tau=3, threshold=-1e9) == 1
    assert test.decide(means, tau=3, threshold=1e9) == -1


def test_invalid_tau_raises():
    with pytest.raises(ValueError):
        compute_log_lr_tau([1.0, 2.0, 3.0], 0.0, 1.0, 2.0, 10, tau=0)
