"""High-level PyTorch helpers for arbitrary-NN SeMI^SGD audits."""

from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch  # noqa: F401 (availability check; user-side loaders may yield tensors)
except ImportError as e:
    raise ImportError(
        "pysemi.torch requires `torch`. Install with `pip install pysemi[torch]`."
    ) from e

from .oracle import TorchOracle


class TrajectoryRecorder:
    """Snapshot the auditable parameter vector at phase boundaries during training."""

    def __init__(self, oracle: TorchOracle):
        self.oracle = oracle
        self.trajectory: List[np.ndarray] = []

    def record(self) -> None:
        """Append the current θ to the trajectory."""
        self.trajectory.append(self.oracle.get_params())

    def reset(self) -> None:
        self.trajectory = []

    def as_list(self) -> List[np.ndarray]:
        return list(self.trajectory)


def estimate_gradient_stats_from_loader(
    oracle: TorchOracle,
    loader: Iterable,
    theta: np.ndarray,
    n_samples: Optional[int] = None,
    jitter: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and (jittered) covariance of per-sample gradients evaluated at θ over a DataLoader."""
    grads: List[np.ndarray] = []
    collected = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            X, y = batch[0], batch[1]
        else:
            raise ValueError("loader must yield (X, y) pairs")
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        psg = oracle.per_sample_grads(theta, X, y)
        grads.append(psg)
        collected += psg.shape[0]
        if n_samples is not None and collected >= n_samples:
            break
    G = np.concatenate(grads, axis=0)
    if n_samples is not None:
        G = G[:n_samples]
    mu = G.mean(axis=0)
    sigma = np.cov(G, rowvar=False)
    sigma = sigma + jitter * np.eye(G.shape[1])
    return mu, sigma
