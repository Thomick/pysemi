"""Tests for ROC + privacy-audit metrics."""

import numpy as np

from pysemi.metrics import (
    clopper_pearson_interval,
    compute_auc,
    compute_balanced_accuracy,
    compute_empirical_roc,
    compute_epsilon_lower_bound_cp,
    compute_epsilon_lower_bound_cp_stratified,
    compute_epsilon_lower_bound_dkw,
    compute_epsilon_lower_bound_dkw_stratified,
    compute_power_at_fpr,
    compute_tpr_at_low_fpr,
    bootstrap_epsilon_ci,
    subsample_epsilon_ci,
)


def test_empirical_roc_perfect_separation():
    h0 = np.linspace(0, 1, 200)
    h1 = np.linspace(2, 3, 200)
    fpr, tpr = compute_empirical_roc(h0, h1)
    assert compute_auc(fpr, tpr) > 0.99


def test_empirical_roc_random_chance():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal(500)
    h1 = rng.standard_normal(500)
    fpr, tpr = compute_empirical_roc(h0, h1)
    assert abs(compute_auc(fpr, tpr) - 0.5) < 0.1


def test_clopper_pearson_extremes():
    lo, hi = clopper_pearson_interval(0, 100)
    assert lo == 0.0 and hi < 0.1
    lo, hi = clopper_pearson_interval(100, 100)
    assert hi == 1.0 and lo > 0.9


def test_compute_power_at_fpr_and_tpr_at_low_fpr():
    h0 = np.linspace(0, 1, 200)
    h1 = np.linspace(2, 3, 200)
    fpr, tpr = compute_empirical_roc(h0, h1)
    assert 0.0 <= compute_power_at_fpr(fpr, tpr, 0.1) <= 1.0
    assert compute_tpr_at_low_fpr(h0, h1, target_fpr=0.05) > 0.5


def test_balanced_accuracy_well_separated():
    h0 = np.linspace(0, 1, 200)
    h1 = np.linspace(2, 3, 200)
    assert compute_balanced_accuracy(h0, h1) > 0.95


def test_epsilon_cp_returns_finite():
    rng = np.random.default_rng(0)
    h0 = rng.standard_normal(500)
    h1 = rng.standard_normal(500) + 1.0
    fpr, tpr = compute_empirical_roc(h0, h1)
    eps = compute_epsilon_lower_bound_cp(fpr, tpr, delta=1e-4, n_h0=500, n_h1=500)
    assert np.isfinite(eps) and eps >= 0


def test_epsilon_dkw_returns_finite():
    rng = np.random.default_rng(1)
    h0 = rng.standard_normal(2000)
    h1 = rng.standard_normal(2000) + 1.5
    fpr, tpr = compute_empirical_roc(h0, h1)
    eps = compute_epsilon_lower_bound_dkw(fpr, tpr, delta=1e-4, n_h0=2000, n_h1=2000)
    assert np.isfinite(eps) and eps >= 0


def test_epsilon_stratified_wrappers():
    rng = np.random.default_rng(2)
    h0 = rng.standard_normal(2000)
    h1 = rng.standard_normal(2000) + 1.5
    fpr, tpr = compute_empirical_roc(h0, h1)
    e_cp = compute_epsilon_lower_bound_cp_stratified(fpr, tpr, delta=1e-4, n_per_stratum=200, T=10)
    e_dkw = compute_epsilon_lower_bound_dkw_stratified(fpr, tpr, delta=1e-4, n_per_stratum=200, T=10)
    assert np.isfinite(e_cp) and np.isfinite(e_dkw)


def test_bootstrap_and_subsample_ci_shapes():
    rng = np.random.default_rng(3)
    h0 = rng.standard_normal(300)
    h1 = rng.standard_normal(300) + 1.0
    m_b, lo_b, hi_b = bootstrap_epsilon_ci(h0, h1, n_bootstrap=20)
    m_s, lo_s, hi_s = subsample_epsilon_ci(h0, h1, n_iters=20)
    assert lo_b <= m_b <= hi_b
    assert lo_s <= m_s <= hi_s
