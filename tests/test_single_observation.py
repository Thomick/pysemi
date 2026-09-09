"""Tests for the FinalObservation baseline."""

import numpy as np

import pysemi
from pysemi.core.single_observation import (
    compute_final_observation_log_lr,
    final_observation_power_theoretical,
    simulate_final_observation,
)


def test_final_obs_lr_finite():
    assert np.isfinite(compute_final_observation_log_lr(0.01, 0.0, 1.0, 2.0, 1000))


def test_final_obs_h1_greater_than_h0_on_average():
    rng = np.random.default_rng(0)
    h0 = [
        simulate_final_observation(0.0, 1.0, 4.0, 200, include_target=False, rng=rng)
        for _ in range(500)
    ]
    h1 = [
        simulate_final_observation(0.0, 1.0, 4.0, 200, include_target=True, rng=rng)
        for _ in range(500)
    ]
    assert np.mean(h1) > np.mean(h0)


def test_final_obs_power_in_unit_interval():
    p = final_observation_power_theoretical(0.05, 0.0, 1.0, 3.0, 500)
    assert 0.0 <= p <= 1.0


def test_final_obs_class():
    baseline = pysemi.FinalObservation(0.0, 1.0, 2.0, n=20, T=10)
    assert baseline.n_total == 200
    assert 0.0 <= baseline.power(0.05) <= 1.0


def test_semi_star_dominates_final_obs():
    """At fixed n·T, SeMI* has more power than FinalObservation."""
    alpha, mu, sigma, z_star, n, T, tau = 0.05, 0.0, 1.0, 3.0, 50, 10, 5
    p_star = pysemi.SeMIStar(mu, sigma, z_star, n).power(alpha, T, tau)
    p_obs = pysemi.FinalObservation(mu, sigma, z_star, n, T).power(alpha)
    assert p_star >= p_obs - 1e-12
