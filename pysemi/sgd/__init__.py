"""SeMI^SGD attacks: plug-in log LR variants and loss-based baselines."""

from .jagielski import (
    back_front_attack,
    compute_loss_linear,
    compute_loss_trajectory,
    delta_attack_statistic,
    score_diff,
    score_ratio,
)
from .oracles import (
    GradientOracle,
    LinearRegressionOracle,
    SoftmaxOracle,
    estimate_gradient_stats,
    mahalanobis_scores,
)
from .plugin_lr import phase_effective_lr, plugin_log_lr_at_tau
from .semi_sgd import SeMISGD, SeMISGDMax, SeMISGDUnif
from .tau_based import (
    loss_diff_at_tau,
    loss_ratio_at_tau,
    max_loss_diff,
    max_loss_ratio,
)

__all__ = [
    "plugin_log_lr_at_tau",
    "phase_effective_lr",
    "GradientOracle",
    "SoftmaxOracle",
    "LinearRegressionOracle",
    "estimate_gradient_stats",
    "mahalanobis_scores",
    "SeMISGD",
    "SeMISGDMax",
    "SeMISGDUnif",
    "loss_diff_at_tau",
    "loss_ratio_at_tau",
    "max_loss_diff",
    "max_loss_ratio",
    "score_diff",
    "score_ratio",
    "back_front_attack",
    "delta_attack_statistic",
    "compute_loss_linear",
    "compute_loss_trajectory",
]
