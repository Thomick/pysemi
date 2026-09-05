"""pysemi — Sequential Membership Inference attacks for models with multiple updates.

Companion library to Michel, Basu, Kaufmann (2026), "Sequential Membership
Inference attacks for auditing models with updates".
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from . import core, metrics, sgd
from .core.dp import (
    compute_noisy_log_lr_max,
    compute_noisy_log_lr_tau,
    compute_noisy_log_lr_unif,
    laplace_sensitivity_clipping,
    simulate_noisy_empirical_means,
)
from .core.noisy_mean import (
    compute_log_lr_max,
    compute_log_lr_tau,
    compute_log_lr_unif,
    semi_max_per_t_alpha,
    semi_max_type_i_error,
    semi_star_power_theoretical,
    simulate_empirical_means,
)
from .core.single_observation import (
    compute_final_observation_log_lr,
    final_observation_power_theoretical,
)
from .sgd.oracles import GradientOracle, LinearRegressionOracle, SoftmaxOracle
from .sgd.semi_sgd import SeMISGD, SeMISGDMax, SeMISGDUnif

__version__ = "0.3.0"


def _validate_empirical_mean_args(sigma: float, n: int) -> None:
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    if n < 2:
        raise ValueError(f"n (batch size) must be ≥ 2, got {n!r}")


class _SeMIBase:
    """Shared state for the empirical-mean SeMI variants."""

    def __init__(self, mu: float, sigma: float, z_star: float, n: int):
        _validate_empirical_mean_args(float(sigma), int(n))
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.z_star = float(z_star)
        self.n = int(n)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mu={self.mu!r}, sigma={self.sigma!r}, "
            f"z_star={self.z_star!r}, n={self.n!r})"
        )

    def simulate(
        self,
        T: int,
        tau_star: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list:
        """Simulate cumulative empirical means under H0 (default) or H1^τ⋆."""
        z = self.z_star if tau_star is not None else None
        return simulate_empirical_means(self.mu, self.sigma, self.n, T, z, tau_star, rng)


class SeMIStar(_SeMIBase):
    """SeMI* statistic: log LR_τ at known τ (NP-optimal for H_0 vs H_1^τ)."""

    def log_lr(self, empirical_means: Sequence[float], tau: int) -> float:
        return compute_log_lr_tau(empirical_means, self.mu, self.sigma, self.z_star, self.n, tau)

    def decide(self, empirical_means: Sequence[float], tau: int, threshold: float) -> int:
        """+1 if log LR_τ ≥ threshold (accept H_1^τ), −1 otherwise."""
        return 1 if self.log_lr(empirical_means, tau) >= threshold else -1

    def power(self, alpha: float, T: int, tau: int) -> float:
        """Asymptotic power at level α."""
        return semi_star_power_theoretical(
            alpha, self.mu, self.sigma, self.z_star, self.n, T, tau
        )


class SeMIMax(_SeMIBase):
    """SeMI^max statistic: ``max_t log LR_t`` (GLR for H_0 vs ⋃_t H_1^t)."""

    def log_lr(self, empirical_means: Sequence[float]) -> float:
        stat, _ = compute_log_lr_max(
            empirical_means, self.mu, self.sigma, self.z_star, self.n
        )
        return stat

    def best_tau(self, empirical_means: Sequence[float]) -> int:
        """Argmax t (1-indexed) of log LR_t."""
        return self.log_lr_and_tau(empirical_means)[1]

    def log_lr_and_tau(self, empirical_means: Sequence[float]) -> Tuple[float, int]:
        return compute_log_lr_max(
            empirical_means, self.mu, self.sigma, self.z_star, self.n
        )

    def decide(self, empirical_means: Sequence[float], threshold: float) -> int:
        return 1 if self.log_lr(empirical_means) >= threshold else -1

    @staticmethod
    def per_t_alpha(overall_alpha: float, T: int) -> float:
        """Per-t α achieving overall Type-I error ``overall_alpha`` over T candidates."""
        return semi_max_per_t_alpha(overall_alpha, T)

    @staticmethod
    def overall_alpha(per_t_alpha: float, T: int) -> float:
        """Overall Type-I error for per-t level ``per_t_alpha`` over T candidates."""
        return semi_max_type_i_error(per_t_alpha, T)


class SeMIUnif(_SeMIBase):
    """SeMI^unif statistic: ``log (1/T) Σ_t LR_t`` (NP-optimal under τ ~ Unif([T]))."""

    def log_lr(self, empirical_means: Sequence[float]) -> float:
        return compute_log_lr_unif(
            empirical_means, self.mu, self.sigma, self.z_star, self.n
        )

    def decide(self, empirical_means: Sequence[float], threshold: float) -> int:
        return 1 if self.log_lr(empirical_means) >= threshold else -1


class SeMIStarDP(SeMIStar):
    """SeMI* applied to the Laplace-DP cumulative-mean release."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        z_star: float,
        n: int,
        epsilon: float,
        sensitivity: Optional[float] = None,
    ):
        super().__init__(mu, sigma, z_star, n)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive and finite, got {epsilon!r}")
        if sensitivity is not None and (not np.isfinite(sensitivity) or sensitivity <= 0.0):
            raise ValueError(f"sensitivity must be positive and finite, got {sensitivity!r}")
        self.epsilon = float(epsilon)
        self.sensitivity = (
            float(sensitivity)
            if sensitivity is not None
            else laplace_sensitivity_clipping(self.sigma, self.n)
        )

    def __repr__(self) -> str:
        return (
            f"SeMIStarDP(mu={self.mu!r}, sigma={self.sigma!r}, z_star={self.z_star!r}, "
            f"n={self.n!r}, epsilon={self.epsilon!r}, sensitivity={self.sensitivity!r})"
        )

    def log_lr(self, noisy_means: Sequence[float], tau: int) -> float:
        return compute_noisy_log_lr_tau(
            noisy_means, self.mu, self.sigma, self.z_star, self.n, tau,
            self.epsilon, self.sensitivity,
        )

    def log_lr_max(self, noisy_means: Sequence[float]) -> float:
        stat, _ = compute_noisy_log_lr_max(
            noisy_means, self.mu, self.sigma, self.z_star, self.n,
            self.epsilon, self.sensitivity,
        )
        return stat

    def log_lr_unif(self, noisy_means: Sequence[float]) -> float:
        return compute_noisy_log_lr_unif(
            noisy_means, self.mu, self.sigma, self.z_star, self.n,
            self.epsilon, self.sensitivity,
        )

    def simulate(
        self,
        T: int,
        tau_star: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list:
        z = self.z_star if tau_star is not None else None
        return simulate_noisy_empirical_means(
            self.mu, self.sigma, self.n, T, self.epsilon,
            z, tau_star, self.sensitivity, rng,
        )

    def power(self, alpha: float, T: int, tau: int) -> float:
        # Override the inherited (non-DP) closed-form; estimate empirically instead.
        raise NotImplementedError(
            "Closed-form power for SeMIStarDP is not available. "
            "Estimate empirically: simulate H0/H1 trajectories and threshold log_lr "
            "at the α-quantile of H0."
        )


class FinalObservation:
    """FinalObservation baseline: log LR from the cumulative mean over n·T samples."""

    def __init__(self, mu: float, sigma: float, z_star: float, n: int, T: int):
        _validate_empirical_mean_args(float(sigma), int(n))
        if T < 1:
            raise ValueError(f"T (number of phases) must be ≥ 1, got {T!r}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.z_star = float(z_star)
        self.n = int(n)
        self.T = int(T)

    def __repr__(self) -> str:
        return (
            f"FinalObservation(mu={self.mu!r}, sigma={self.sigma!r}, "
            f"z_star={self.z_star!r}, n={self.n!r}, T={self.T!r})"
        )

    @property
    def n_total(self) -> int:
        return self.n * self.T

    def log_lr(self, final_mean: float) -> float:
        return compute_final_observation_log_lr(
            final_mean, self.mu, self.sigma, self.z_star, self.n_total
        )

    def power(self, alpha: float) -> float:
        """Asymptotic power at level α."""
        return final_observation_power_theoretical(
            alpha, self.mu, self.sigma, self.z_star, self.n_total
        )


__all__ = [
    "__version__",
    "core",
    "sgd",
    "metrics",
    "SeMIStar",
    "SeMIMax",
    "SeMIUnif",
    "SeMIStarDP",
    "FinalObservation",
    "SeMISGD",
    "SeMISGDMax",
    "SeMISGDUnif",
    "GradientOracle",
    "SoftmaxOracle",
    "LinearRegressionOracle",
]
