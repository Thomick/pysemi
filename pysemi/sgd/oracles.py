"""Gradient oracles used by the SeMI^SGD attacks.

An oracle is anything that exposes ``grad_at(theta, x_star, y_star)``: given a
flat parameter vector, return the per-sample gradient at the target. Three
concrete oracles ship with the library; any user object satisfying the
protocol works.
"""

from typing import Protocol, Tuple

import numpy as np


class GradientOracle(Protocol):
    """Maps (θ, x⋆, y⋆) → ∇_θ L(θ; x⋆, y⋆) as a flat vector."""

    def grad_at(self, theta: np.ndarray, x_star: np.ndarray, y_star) -> np.ndarray: ...


class SoftmaxOracle:
    """Numpy last-layer multinomial-logistic oracle, layout ``[vec(W); b]``."""

    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    @property
    def n_params(self) -> int:
        return self.input_dim * self.num_classes + self.num_classes

    def _unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w_end = self.input_dim * self.num_classes
        W = theta[:w_end].reshape(self.input_dim, self.num_classes)
        b = theta[w_end:]
        return W, b

    def _forward(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        W, b = self._unpack(theta)
        x = np.atleast_2d(x)
        logits = x @ W + b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def grad_at(self, theta: np.ndarray, x_star: np.ndarray, y_star) -> np.ndarray:
        """Per-sample CE gradient at (x⋆, y⋆), flattened to match θ's layout."""
        x = np.atleast_2d(x_star).astype(float)
        probs = self._forward(theta, x)
        y_oh = np.zeros_like(probs)
        y_oh[0, int(y_star)] = 1.0
        delta = probs - y_oh
        gw = (x.T @ delta).flatten()
        gb = delta[0]
        return np.concatenate([gw, gb])

    def loss(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy on (x, y)."""
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)
        probs = self._forward(theta, x)
        return float(-np.mean(np.log(probs[np.arange(len(y)), y.astype(int)] + 1e-12)))


class LinearRegressionOracle:
    """Analytic SGD-linear oracle: ``∇ ½(θᵀx − y)² = (θᵀx − y) x``."""

    def grad_at(self, theta: np.ndarray, x_star: np.ndarray, y_star) -> np.ndarray:
        residual = float(np.dot(theta, x_star)) - float(y_star)
        return residual * np.asarray(x_star, dtype=float)

    def loss(self, theta: np.ndarray, x_star: np.ndarray, y_star) -> float:
        residual = float(np.dot(theta, x_star)) - float(y_star)
        return 0.5 * residual**2


def estimate_gradient_stats(
    gradients: np.ndarray,
    jitter: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample mean and jittered covariance of a (n_samples, d) gradient matrix."""
    mu = np.mean(gradients, axis=0)
    sigma = np.cov(gradients, rowvar=False)
    sigma = sigma + jitter * np.eye(gradients.shape[1])
    return mu, sigma


def mahalanobis_scores(
    gradients: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance (g − μ)ᵀ Σ⁻¹ (g − μ) for each row of ``gradients``."""
    diff = gradients - mu
    return np.sum((diff @ sigma_inv) * diff, axis=1)
