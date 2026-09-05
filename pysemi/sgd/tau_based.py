"""Loss-only baseline scores: loss-diff and loss-ratio at τ and over τ."""

from typing import Tuple

import numpy as np


def loss_diff_at_tau(losses: np.ndarray, tau: int) -> float:
    """``L(θ_{τ-1}, z⋆) − L(θ_τ, z⋆)``; larger ⇒ more likely member."""
    if tau < 1 or tau >= len(losses):
        return 0.0
    return float(losses[tau - 1] - losses[tau])


def loss_ratio_at_tau(
    losses: np.ndarray,
    tau: int,
    stability_constant: float = 0.0005,
) -> float:
    """``(c + L_{τ-1}) / (c + L_τ)``; >1 ⇒ more likely member."""
    if tau < 1 or tau >= len(losses):
        return 1.0
    return float(
        (stability_constant + losses[tau - 1]) / (stability_constant + losses[tau])
    )


def max_loss_diff(losses: np.ndarray) -> Tuple[float, int]:
    """``(max_τ L_{τ-1} − L_τ, argmax τ)``."""
    T = len(losses) - 1
    if T < 1:
        return 0.0, 0
    diffs = np.array([float(losses[t - 1] - losses[t]) for t in range(1, T + 1)])
    best = int(np.argmax(diffs))
    return float(diffs[best]), best + 1


def max_loss_ratio(
    losses: np.ndarray,
    stability_constant: float = 0.0005,
) -> Tuple[float, int]:
    """``(max_τ (c+L_{τ-1})/(c+L_τ), argmax τ)``."""
    T = len(losses) - 1
    if T < 1:
        return 1.0, 0
    ratios = np.array([
        (stability_constant + losses[t - 1]) / (stability_constant + losses[t])
        for t in range(1, T + 1)
    ])
    best = int(np.argmax(ratios))
    return float(ratios[best]), best + 1
