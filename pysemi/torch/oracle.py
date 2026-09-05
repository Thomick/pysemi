"""PyTorch oracle for arbitrary-NN SeMI^SGD audits."""

from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:
    raise ImportError(
        "pysemi.torch requires `torch`. Install with `pip install pysemi[torch]`."
    ) from e


def _resolve_params(
    module: "nn.Module",
    params: Optional[Sequence[str]],
) -> List["nn.Parameter"]:
    """Return ``nn.Parameter`` objects, by name if given else all trainable params."""
    if params is None:
        return [p for p in module.parameters() if p.requires_grad]
    named = dict(module.named_parameters())
    out = []
    for name in params:
        if name not in named:
            raise KeyError(f"Parameter {name!r} not in module. Available: {list(named)}")
        out.append(named[name])
    return out


class TorchOracle:
    """Per-sample gradient oracle for an arbitrary subset of an ``nn.Module``'s parameters."""

    def __init__(
        self,
        module: "nn.Module",
        loss_fn: Optional[Callable] = None,
        param_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ):
        self.module = module
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(module.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(self.device)
        self._params = _resolve_params(module, param_names)
        self._param_names = list(param_names) if param_names is not None else [
            name for name, p in module.named_parameters() if p.requires_grad
        ]
        self._shapes = [tuple(p.shape) for p in self._params]
        self._sizes = [int(np.prod(s)) for s in self._shapes]

    @property
    def n_params(self) -> int:
        return int(sum(self._sizes))

    @property
    def param_names(self) -> List[str]:
        return list(self._param_names)

    def get_params(self) -> np.ndarray:
        """Current parameter vector flattened in ``param_names`` order."""
        return np.concatenate([
            p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in self._params
        ])

    def set_params(self, theta: np.ndarray) -> None:
        """Write a flat parameter vector back into the module."""
        offset = 0
        with torch.no_grad():
            for p, size, shape in zip(self._params, self._sizes, self._shapes):
                chunk = theta[offset:offset + size].reshape(shape)
                p.copy_(torch.as_tensor(chunk, dtype=p.dtype, device=p.device))
                offset += size

    def _flatten_grad(self) -> np.ndarray:
        chunks = []
        for p in self._params:
            if p.grad is None:
                raise RuntimeError(
                    "Parameter has no gradient. Did backward() get called "
                    "with the auditable params in the graph?"
                )
            chunks.append(p.grad.detach().cpu().numpy().reshape(-1).astype(np.float64))
        return np.concatenate(chunks)

    def grad_at(self, theta: np.ndarray, x_star, y_star) -> np.ndarray:
        """Per-sample gradient of ``loss_fn(module(x⋆), y⋆)`` wrt the audited params, at θ."""
        self.set_params(theta)
        was_training = self.module.training
        self.module.eval()
        try:
            x = self._to_tensor(x_star, batch_axis=True)
            y = self._to_target(y_star)
            for p in self._params:
                p.grad = None
            out = self.module(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            return self._flatten_grad()
        finally:
            if was_training:
                self.module.train()

    def per_sample_grads(self, theta: np.ndarray, X, y) -> np.ndarray:
        """Per-sample gradients for a batch (X, y); shape ``(batch, n_params)``."""
        self.set_params(theta)
        was_training = self.module.training
        self.module.eval()
        try:
            X_t = self._to_tensor(X, batch_axis=False)
            y_t = self._to_target(y)
            return self._per_sample_loop(X_t, y_t)
        finally:
            if was_training:
                self.module.train()

    def _per_sample_loop(self, X: "torch.Tensor", y: "torch.Tensor") -> np.ndarray:
        out = np.empty((X.shape[0], self.n_params), dtype=np.float64)
        for i in range(X.shape[0]):
            for p in self._params:
                p.grad = None
            xi = X[i:i + 1]
            yi = y[i:i + 1]
            loss = self.loss_fn(self.module(xi), yi)
            loss.backward()
            out[i] = self._flatten_grad()
        return out

    def loss(self, theta: np.ndarray, x, y) -> float:
        """Mean loss on (x, y) at parameters θ."""
        self.set_params(theta)
        was_training = self.module.training
        self.module.eval()
        try:
            with torch.no_grad():
                x_t = self._to_tensor(x, batch_axis=False)
                y_t = self._to_target(y)
                return float(self.loss_fn(self.module(x_t), y_t).item())
        finally:
            if was_training:
                self.module.train()

    def _to_tensor(self, x, batch_axis: bool) -> "torch.Tensor":
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        if batch_axis and t.ndim == self._params[0].ndim - 1:
            t = t.unsqueeze(0)
        if batch_axis and t.ndim < 2:
            t = t.unsqueeze(0)
        return t

    def _to_target(self, y) -> "torch.Tensor":
        y_arr = np.atleast_1d(np.asarray(y))
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            return torch.as_tensor(y_arr, dtype=torch.long, device=self.device)
        return torch.as_tensor(y_arr, dtype=torch.float32, device=self.device)


def _find_last_linear(module: "nn.Module") -> str:
    last_name = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError("No nn.Linear submodule found")
    return last_name


class TorchLastLayerOracle(TorchOracle):
    """``TorchOracle`` restricted to the last ``nn.Linear`` of ``module``."""

    def __init__(
        self,
        module: "nn.Module",
        loss_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        last_name = _find_last_linear(module)
        prefix = f"{last_name}." if last_name else ""
        super().__init__(
            module,
            loss_fn=loss_fn,
            param_names=[f"{prefix}weight", f"{prefix}bias"],
            device=device,
        )

    @property
    def in_features(self) -> int:
        return int(self._params[0].shape[1])

    @property
    def out_features(self) -> int:
        return int(self._params[0].shape[0])
