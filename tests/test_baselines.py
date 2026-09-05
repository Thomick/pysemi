"""Tests for tau-based and Jagielski baseline attacks."""

import numpy as np
import pytest

from pysemi.sgd import (
    loss_diff_at_tau,
    loss_ratio_at_tau,
    max_loss_diff,
    max_loss_ratio,
    score_diff,
    score_ratio,
    back_front_attack,
    delta_attack_statistic,
    compute_loss_linear,
    compute_loss_trajectory,
)


def test_loss_diff_basic():
    losses = np.array([1.0, 0.3, 0.4, 0.5])
    assert np.isclose(loss_diff_at_tau(losses, tau=1), 0.7)
    val, idx = max_loss_diff(losses)
    assert np.isclose(val, 0.7)
    assert idx == 1


def test_loss_ratio_basic():
    losses = np.array([1.0, 0.5, 0.4, 0.5])
    ratio = loss_ratio_at_tau(losses, tau=1)
    assert ratio > 1.0
    val, idx = max_loss_ratio(losses)
    assert val == loss_ratio_at_tau(losses, tau=idx)


def test_score_diff_ratio():
    assert score_diff(1.0, 0.5) == 0.5
    assert score_ratio(1.0, 0.5) > 1.0


def test_back_front_invalid_method():
    losses = np.array([1.0, 0.5])
    with pytest.raises(ValueError):
        back_front_attack(losses, method="bogus")


def test_back_front_ratio_member_below_one():
    losses = np.array([1.0, 0.1])
    assert back_front_attack(losses, method="ratio") < 1.0
    assert back_front_attack(losses, method="diff") < 0


def test_delta_attack_finds_best():
    losses = np.array([1.0, 0.9, 0.1, 0.5])
    val, idx = delta_attack_statistic(losses, method="diff")
    assert idx == 2
    assert np.isclose(val, 0.8)


def test_compute_loss_linear_and_trajectory():
    theta = np.array([1.0, 1.0])
    assert np.isclose(compute_loss_linear(theta, np.array([1.0, 1.0]), 2.0), 0.0)
    thetas = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    losses = compute_loss_trajectory(thetas, np.array([1.0, 1.0]), 2.0)
    assert losses.shape == (2,)
    assert np.isclose(losses[1], 0.0)


def test_loss_diff_out_of_range_returns_zero():
    losses = np.array([1.0, 0.5])
    assert loss_diff_at_tau(losses, tau=0) == 0.0
    assert loss_diff_at_tau(losses, tau=5) == 0.0
    assert loss_ratio_at_tau(losses, tau=0) == 1.0
