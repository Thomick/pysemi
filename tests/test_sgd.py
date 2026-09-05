"""Tests for SeMI^SGD / SeMI^SGD-max / SeMI^SGD-unif."""

import numpy as np
import pytest

import pysemi
from pysemi.sgd import (
    LinearRegressionOracle,
    SeMISGD,
    SeMISGDMax,
    SeMISGDUnif,
    SoftmaxOracle,
    estimate_gradient_stats,
    mahalanobis_scores,
    phase_effective_lr,
    plugin_log_lr_at_tau,
)


def test_plugin_lr_finite():
    rng = np.random.default_rng(0)
    d, n = 5, 20
    theta_prev = rng.standard_normal(d)
    theta_curr = theta_prev + 0.01 * rng.standard_normal(d)
    mu_g = rng.standard_normal(d) * 0.1
    sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d)) @ rng.standard_normal((d, d)).T
    sigma_inv = np.linalg.inv(sigma)
    delta_g = rng.standard_normal(d)
    m_star = float(delta_g @ sigma_inv @ delta_g)
    out = plugin_log_lr_at_tau(theta_prev, theta_curr, eta=0.01, n_batch=n,
                                mu_g=mu_g, sigma_inv=sigma_inv, delta_g=delta_g, m_star=m_star)
    assert np.isfinite(out)


def test_phase_effective_lr():
    assert np.isclose(phase_effective_lr(lr=0.1, K=3, batch_size=10, pool_size=100), 0.03)


def test_linear_regression_oracle_grad():
    oracle = LinearRegressionOracle()
    theta = np.array([1.0, -1.0])
    g = oracle.grad_at(theta, np.array([2.0, 3.0]), 1.0)
    # residual = -2, grad = -2 * [2,3]
    assert np.allclose(g, [-4.0, -6.0])


def test_softmax_oracle_shapes():
    oracle = SoftmaxOracle(input_dim=4, num_classes=3)
    theta = np.zeros(oracle.n_params)
    g = oracle.grad_at(theta, np.array([0.1, 0.2, 0.3, 0.4]), y_star=1)
    assert g.shape == (oracle.n_params,)


def test_gradient_stats_and_mahalanobis():
    rng = np.random.default_rng(1)
    grads = rng.standard_normal((100, 5))
    mu, sigma = estimate_gradient_stats(grads)
    assert mu.shape == (5,)
    assert sigma.shape == (5, 5)
    scores = mahalanobis_scores(grads, mu, np.linalg.inv(sigma))
    assert scores.shape == (100,)
    assert np.all(scores >= 0)


def _sgd_run_linear(rng, theta0, x_star, y_star, T, n, d, eta, include_target):
    """Helper: run T SGD steps on synthetic linear-regression batches."""
    theta = theta0.copy()
    thetas = [theta.copy()]
    for t in range(1, T + 1):
        X = rng.standard_normal((n, d))
        y_true = X @ np.zeros(d) + rng.standard_normal(n) * 0.1
        if include_target and t == T // 2 + 1:
            X[-1] = x_star
            y_true[-1] = y_star
        residuals = (X @ theta) - y_true
        grad = (residuals[:, None] * X).mean(axis=0)
        theta = theta - eta * grad
        thetas.append(theta.copy())
    return thetas


def test_semi_sgd_separates_h1_from_h0_on_linear_regression():
    rng = np.random.default_rng(0)
    d, n, T, eta = 5, 50, 10, 0.05
    x_star = rng.standard_normal(d) * 3
    y_star = 5.0
    oracle = LinearRegressionOracle()

    bg_grads = np.array([
        oracle.grad_at(rng.standard_normal(d) * 0.5, rng.standard_normal(d), float(rng.standard_normal()))
        for _ in range(200)
    ])
    mu_g, sigma = estimate_gradient_stats(bg_grads, jitter=1e-2)
    sigma_inv = np.linalg.inv(sigma)

    h1_max, h0_max = [], []
    for _ in range(40):
        theta0 = rng.standard_normal(d) * 0.1
        thetas_h1 = _sgd_run_linear(rng, theta0, x_star, y_star, T, n, d, eta, include_target=True)
        thetas_h0 = _sgd_run_linear(rng, theta0, x_star, y_star, T, n, d, eta, include_target=False)
        h1_max.append(SeMISGDMax(thetas_h1, oracle, mu_g, sigma_inv, eta=eta, n_batch=n).log_lr(x_star, y_star))
        h0_max.append(SeMISGDMax(thetas_h0, oracle, mu_g, sigma_inv, eta=eta, n_batch=n).log_lr(x_star, y_star))
    assert np.mean(h1_max) > np.mean(h0_max)


def test_semi_sgd_unif_within_log_T_of_max():
    """log mean LR_t lies in [max log LR_t − log T, max log LR_t]."""
    rng = np.random.default_rng(1)
    oracle = LinearRegressionOracle()
    d, n, T = 3, 20, 5
    thetas = [rng.standard_normal(d) * 0.01 for _ in range(T + 1)]
    test_max = SeMISGDMax(thetas, oracle, np.zeros(d), np.eye(d), eta=0.1, n_batch=n)
    test_unif = SeMISGDUnif(thetas, oracle, np.zeros(d), np.eye(d), eta=0.1, n_batch=n)
    x_star = rng.standard_normal(d)
    y_star = 1.0
    m = test_max.log_lr(x_star, y_star)
    u = test_unif.log_lr(x_star, y_star)
    assert m - np.log(T) - 1e-9 <= u <= m + 1e-9


def test_semi_sgd_invalid_tau_raises():
    rng = np.random.default_rng(0)
    oracle = LinearRegressionOracle()
    thetas = [rng.standard_normal(3) for _ in range(4)]
    test = SeMISGD(thetas, oracle, mu_g=np.zeros(3), sigma_inv=np.eye(3), eta=0.1, n_batch=10)
    with pytest.raises(ValueError):
        test.log_lr(tau=0, x_star=np.zeros(3), y_star=0.0)
    with pytest.raises(ValueError):
        test.log_lr(tau=10, x_star=np.zeros(3), y_star=0.0)


def test_semi_sgd_too_short_trajectory():
    with pytest.raises(ValueError):
        SeMISGD([np.zeros(3)], LinearRegressionOracle(), np.zeros(3), np.eye(3), eta=0.1, n_batch=10)
