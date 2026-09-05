"""PyTorch integration for SeMI^SGD audits of arbitrary neural networks.

Importing this submodule requires ``torch`` (``pip install pysemi[torch]``).
"""

from .audit import TrajectoryRecorder, estimate_gradient_stats_from_loader
from .oracle import TorchLastLayerOracle, TorchOracle

__all__ = [
    "TorchOracle",
    "TorchLastLayerOracle",
    "TrajectoryRecorder",
    "estimate_gradient_stats_from_loader",
]
