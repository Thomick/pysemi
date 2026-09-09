"""Jagielski et al. baselines: back-front and delta attacks on the target loss."""

from typing import List, Sequence, Tuple

import numpy as np


def compute_loss_linear(theta: np.ndarray, x: np.ndarray, y: float) -> float:
    """½(θᵀx − y)² squared loss on (x, y)."""
    residual = float(np.dot(theta, x)) - float(y)
    return 0.5 * residual**2


def compute_loss_trajectory(
    thetas: Sequence[np.ndarray],
    x_star: np.ndarray,
    y_star: float,
) -> np.ndarray:
    """Loss on the target at each θ_t in the trajectory."""
    return np.array([compute_loss_linear(theta, x_star, y_star) for theta in thetas])


def score_diff(loss_before: float, loss_after: float) -> float:
    """``L_before − L_after``; positive ⇒ loss dropped."""
    return float(loss_before - loss_after)


def score_ratio(
    loss_before: float,
    loss_after: float,
    stability_constant: float = 0.0005,
) -> float:
    """``(c+L_before)/(c+L_after)``; >1 ⇒ loss dropped."""
    return float((stability_constant + loss_before) / (stability_constant + loss_after))


def back_front_attack(losses: np.ndarray, method: str = "ratio") -> float:
    """First-vs-last loss comparison; lower ⇒ more likely member (Jagielski sign convention)."""
    stab = 0.0005
    if method == "diff":
        return float(losses[-1] - losses[0])
    if method == "ratio":
        return float((stab + losses[-1]) / (stab + losses[0]))
    raise ValueError(f"method must be 'ratio' or 'diff', got {method!r}")


def delta_attack_statistic(
    losses: np.ndarray,
    method: str = "ratio",
) -> Tuple[float, int]:
    """``(max_t score(L_{t-1}, L_t), argmax t)``."""
    T = len(losses) - 1
    scores: List[float] = []
    for t in range(1, T + 1):
        before, after = float(losses[t - 1]), float(losses[t])
        s = score_ratio(before, after) if method == "ratio" else score_diff(before, after)
        scores.append(s)
    best = int(np.argmax(scores))
    return scores[best], best + 1
