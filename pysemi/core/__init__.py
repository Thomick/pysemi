"""Empirical-mean SeMI primitives (clean and Laplace-DP variants)."""

from .dp import (
    compute_log_prob_convolution,
    compute_noisy_log_lr_max,
    compute_noisy_log_lr_t,
    compute_noisy_log_lr_tau,
    compute_noisy_log_lr_unif,
    laplace_sensitivity_clipping,
    laplace_sensitivity_range,
    simulate_noisy_empirical_means,
)
from .noisy_mean import (
    compute_log_lr_max,
    compute_log_lr_t,
    compute_log_lr_tau,
    compute_log_lr_unif,
    semi_max_per_t_alpha,
    semi_max_type_i_error,
    semi_star_power_theoretical,
    simulate_empirical_means,
    simulate_empirical_means_multiple_targets,
)
from .single_observation import (
    compute_final_observation_log_lr,
    final_observation_power_theoretical,
    simulate_final_observation,
)

__all__ = [
    "compute_log_lr_t",
    "compute_log_lr_tau",
    "compute_log_lr_max",
    "compute_log_lr_unif",
    "simulate_empirical_means",
    "simulate_empirical_means_multiple_targets",
    "semi_star_power_theoretical",
    "semi_max_per_t_alpha",
    "semi_max_type_i_error",
    "compute_final_observation_log_lr",
    "final_observation_power_theoretical",
    "simulate_final_observation",
    "compute_log_prob_convolution",
    "compute_noisy_log_lr_t",
    "compute_noisy_log_lr_tau",
    "compute_noisy_log_lr_max",
    "compute_noisy_log_lr_unif",
    "simulate_noisy_empirical_means",
    "laplace_sensitivity_clipping",
    "laplace_sensitivity_range",
]
