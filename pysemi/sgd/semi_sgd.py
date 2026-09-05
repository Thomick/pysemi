"""SeMI^SGD, SeMI^SGD-max and SeMI^SGD-unif attacks on parameter trajectories."""

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from .oracles import GradientOracle
from .plugin_lr import plugin_log_lr_at_tau


class SeMISGD:
    """Plug-in SeMI^SGD attack from a trajectory of parameter snapshots."""

    def __init__(
        self,
        theta_trajectory: Sequence[np.ndarray],
        oracle: GradientOracle,
        mu_g: np.ndarray,
        sigma_inv: np.ndarray,
        eta: float,
        n_batch: int,
    ):
        if len(theta_trajectory) < 2:
            raise ValueError("theta_trajectory must contain at least θ_0 and θ_1")
        self.thetas: List[np.ndarray] = [np.asarray(t, dtype=float) for t in theta_trajectory]
        d = self.thetas[0].shape[0]
        for k, t in enumerate(self.thetas):
            if t.shape != (d,):
                raise ValueError(
                    f"theta_trajectory[{k}] has shape {t.shape}, expected ({d},). "
                    f"All snapshots must be flat vectors of the same dimension."
                )
        mu_g = np.asarray(mu_g, dtype=float)
        sigma_inv = np.asarray(sigma_inv, dtype=float)
        if mu_g.shape != (d,):
            raise ValueError(f"mu_g shape {mu_g.shape} does not match theta dim ({d},)")
        if sigma_inv.shape != (d, d):
            raise ValueError(f"sigma_inv shape {sigma_inv.shape} does not match ({d}, {d})")
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError(f"eta must be positive and finite, got {eta!r}")
        if n_batch < 2:
            raise ValueError(f"n_batch must be ≥ 2, got {n_batch!r}")
        self.oracle = oracle
        self.mu_g = mu_g
        self.sigma_inv = sigma_inv
        self.eta = float(eta)
        self.n_batch = int(n_batch)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(T={self.T}, n_params={self.thetas[0].shape[0]}, "
            f"eta={self.eta!r}, n_batch={self.n_batch!r})"
        )

    @property
    def T(self) -> int:
        """Number of phase transitions (τ ranges over 1..T)."""
        return len(self.thetas) - 1

    def log_lr_t(self, t: int, x_star: np.ndarray, y_star) -> float:
        """Plug-in log LR_t at a single t (1-indexed)."""
        if not 1 <= t <= self.T:
            raise ValueError(f"t must be in [1, {self.T}], got {t}")
        theta_prev = self.thetas[t - 1]
        theta_curr = self.thetas[t]
        grad_star = self.oracle.grad_at(theta_prev, x_star, y_star)
        delta_g = grad_star - self.mu_g
        m_star = float(delta_g @ self.sigma_inv @ delta_g)
        return plugin_log_lr_at_tau(
            theta_prev, theta_curr, self.eta, self.n_batch,
            self.mu_g, self.sigma_inv, delta_g, m_star,
        )

    def log_lr(self, tau: int, x_star: np.ndarray, y_star) -> float:
        """SeMI^SGD statistic: log LR_τ at known τ."""
        return self.log_lr_t(tau, x_star, y_star)

    def log_lr_all_t(self, x_star: np.ndarray, y_star) -> np.ndarray:
        """All log LR_t over t = 1..T."""
        return np.array([self.log_lr_t(t, x_star, y_star) for t in range(1, self.T + 1)])


class SeMISGDMax(SeMISGD):
    """SeMI^SGD-max: ``max_t log LR_t`` (GLR over τ)."""

    def log_lr(self, x_star: np.ndarray, y_star) -> float:  # type: ignore[override]
        return float(np.max(self.log_lr_all_t(x_star, y_star)))

    def best_tau(self, x_star: np.ndarray, y_star) -> int:
        """Argmax t (1-indexed) of log LR_t."""
        return int(np.argmax(self.log_lr_all_t(x_star, y_star))) + 1

    def log_lr_and_tau(self, x_star: np.ndarray, y_star) -> Tuple[float, int]:
        """``(max_t log LR_t, argmax t)``."""
        lrs = self.log_lr_all_t(x_star, y_star)
        idx = int(np.argmax(lrs))
        return float(lrs[idx]), idx + 1


class SeMISGDUnif(SeMISGD):
    """SeMI^SGD-unif: ``log (1/T) Σ_t LR_t``."""

    def log_lr(self, x_star: np.ndarray, y_star) -> float:  # type: ignore[override]
        lrs = self.log_lr_all_t(x_star, y_star)
        return float(logsumexp(lrs) - np.log(len(lrs)))
