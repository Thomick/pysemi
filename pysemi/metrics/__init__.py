"""ROC curves and privacy-audit epsilon estimators."""

from .privacy_audit import (
    bootstrap_epsilon_ci,
    bootstrap_epsilon_samples,
    calibrate_threshold_grid,
    clopper_pearson_interval,
    compute_epsilon_lower_bound_cp,
    compute_epsilon_lower_bound_cp_grid,
    compute_epsilon_lower_bound_cp_stratified,
    compute_epsilon_lower_bound_dkw,
    compute_epsilon_lower_bound_dkw_stratified,
    subsample_epsilon_ci,
    subsample_epsilon_samples,
)
from .roc import (
    compute_auc,
    compute_balanced_accuracy,
    compute_empirical_roc,
    compute_power_at_fpr,
    compute_tpr_at_low_fpr,
)

__all__ = [
    "compute_empirical_roc",
    "compute_auc",
    "compute_power_at_fpr",
    "compute_tpr_at_low_fpr",
    "compute_balanced_accuracy",
    "clopper_pearson_interval",
    "calibrate_threshold_grid",
    "compute_epsilon_lower_bound_cp",
    "compute_epsilon_lower_bound_cp_grid",
    "compute_epsilon_lower_bound_dkw",
    "compute_epsilon_lower_bound_cp_stratified",
    "compute_epsilon_lower_bound_dkw_stratified",
    "bootstrap_epsilon_ci",
    "bootstrap_epsilon_samples",
    "subsample_epsilon_ci",
    "subsample_epsilon_samples",
]
