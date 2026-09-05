"""Empirical ROC, AUC, TPR-at-FPR, balanced accuracy."""

from typing import Tuple

import numpy as np
from scipy.integrate import trapezoid


def compute_empirical_roc(
    stats_h0: np.ndarray,
    stats_h1: np.ndarray,
    n_points: int = 500,
    higher_is_h1: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical ROC from paired H0/H1 score arrays, log-uniformly spaced in FPR."""
    desired_fprs = np.logspace(-4, 0, n_points)
    if higher_is_h1:
        thresholds = np.quantile(stats_h0, 1 - desired_fprs)
        fpr = np.array([np.mean(stats_h0 >= t) for t in thresholds])
        tpr = np.array([np.mean(stats_h1 >= t) for t in thresholds])
    else:
        thresholds = np.quantile(stats_h0, desired_fprs)
        fpr = np.array([np.mean(stats_h0 <= t) for t in thresholds])
        tpr = np.array([np.mean(stats_h1 <= t) for t in thresholds])
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    unique = np.unique(fpr)
    # Keep max TPR per unique FPR to remove staircase artifacts before integration.
    tpr_smoothed = np.array([np.max(tpr[fpr == f]) for f in unique])
    return unique, tpr_smoothed


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC of a sorted ROC curve."""
    return float(trapezoid(tpr, fpr))


def compute_power_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_fpr: float,
) -> float:
    """TPR at the ROC point closest to ``target_fpr``."""
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx])


def compute_tpr_at_low_fpr(
    stats_h0: np.ndarray,
    stats_h1: np.ndarray,
    target_fpr: float = 0.01,
    higher_is_h1: bool = True,
) -> float:
    """TPR at the exact empirical threshold yielding ``target_fpr`` under H0."""
    if higher_is_h1:
        threshold = np.quantile(stats_h0, 1 - target_fpr)
        return float(np.mean(stats_h1 >= threshold))
    threshold = np.quantile(stats_h0, target_fpr)
    return float(np.mean(stats_h1 <= threshold))


def compute_balanced_accuracy(
    stats_h0: np.ndarray,
    stats_h1: np.ndarray,
    higher_is_h1: bool = True,
    n_thresholds: int = 1000,
) -> float:
    """``max_threshold (TPR + TNR) / 2``."""
    all_scores = np.concatenate([stats_h0, stats_h1])
    thresholds = np.unique(all_scores)
    if len(thresholds) > n_thresholds:
        thresholds = np.quantile(all_scores, np.linspace(0, 1, n_thresholds))
    best = 0.5
    for t in thresholds:
        if higher_is_h1:
            tpr = float(np.mean(stats_h1 >= t))
            tnr = float(np.mean(stats_h0 < t))
        else:
            tpr = float(np.mean(stats_h1 <= t))
            tnr = float(np.mean(stats_h0 > t))
        best = max(best, (tpr + tnr) / 2)
    return best
