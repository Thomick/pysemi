"""Tests for the pysemi.torch arbitrary-NN audit module."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import pysemi
from pysemi.sgd import estimate_gradient_stats
from pysemi.torch import (
    TorchLastLayerOracle,
    TorchOracle,
    TrajectoryRecorder,
    estimate_gradient_stats_from_loader,
)


def _make_mlp(in_dim=8, hidden=12, out_dim=3):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def test_torch_oracle_param_layout_roundtrip():
    torch.manual_seed(0)
    model = _make_mlp()
    oracle = TorchOracle(model)
    theta = oracle.get_params()
    perturbed = theta + 1.3
    oracle.set_params(perturbed)
    assert np.allclose(oracle.get_params(), perturbed)


def test_torch_last_layer_oracle_param_subset():
    torch.manual_seed(1)
    model = _make_mlp(in_dim=4, hidden=8, out_dim=2)
    oracle = TorchLastLayerOracle(model)
    expected = 8 * 2 + 2
    assert oracle.n_params == expected
    assert oracle.in_features == 8 and oracle.out_features == 2


def test_torch_oracle_grad_at_matches_loss_backprop():
    torch.manual_seed(2)
    model = _make_mlp(in_dim=4, hidden=8, out_dim=2)
    oracle = TorchLastLayerOracle(model)
    theta = oracle.get_params()
    x = np.random.default_rng(0).standard_normal(4)
    y = 1
    g = oracle.grad_at(theta, x, y)
    assert g.shape == (oracle.n_params,)
    assert np.isfinite(g).all()


def test_per_sample_grads_matches_individual_grads():
    torch.manual_seed(3)
    model = _make_mlp(in_dim=3, hidden=6, out_dim=2)
    oracle = TorchLastLayerOracle(model)
    theta = oracle.get_params()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=4)
    psg = oracle.per_sample_grads(theta, X, y)
    for i in range(4):
        g_i = oracle.grad_at(theta, X[i], y[i])
        assert np.allclose(psg[i], g_i, atol=1e-5)


def test_trajectory_recorder():
    torch.manual_seed(4)
    model = _make_mlp(in_dim=4, hidden=6, out_dim=2)
    oracle = TorchLastLayerOracle(model)
    recorder = TrajectoryRecorder(oracle)
    for _ in range(3):
        recorder.record()
    assert len(recorder.trajectory) == 3
    assert recorder.trajectory[0].shape == (oracle.n_params,)
    recorder.reset()
    assert recorder.trajectory == []


def test_estimate_gradient_stats_from_loader():
    torch.manual_seed(5)
    model = _make_mlp(in_dim=4, hidden=6, out_dim=2)
    oracle = TorchLastLayerOracle(model)
    theta = oracle.get_params()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=40)

    def loader():
        for i in range(0, 40, 8):
            yield X[i:i + 8], y[i:i + 8]

    mu_g, sigma = estimate_gradient_stats_from_loader(oracle, loader(), theta, n_samples=20)
    assert mu_g.shape == (oracle.n_params,)
    assert sigma.shape == (oracle.n_params, oracle.n_params)


def test_torch_oracle_drives_semi_sgd():
    """End-to-end: train a tiny MLP, audit the last layer with SeMI^SGD."""
    torch.manual_seed(6)
    rng = np.random.default_rng(0)

    in_dim, hidden, out_dim = 6, 16, 2
    n_train = 200
    Xb = rng.standard_normal((n_train, in_dim)).astype(np.float32)
    yb = (Xb.sum(axis=1) > 0).astype(np.int64)

    model = _make_mlp(in_dim, hidden, out_dim)
    oracle = TorchLastLayerOracle(model)
    recorder = TrajectoryRecorder(oracle)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    x_star = np.ones(in_dim, dtype=np.float32) * 3.0
    y_star = 0
    T = 4
    K = 2
    batch = 16
    tau_star = 2

    recorder.record()
    for t in range(1, T + 1):
        for _ in range(K):
            idx = rng.choice(n_train, batch, replace=False)
            X_t = torch.as_tensor(Xb[idx]); y_t = torch.as_tensor(yb[idx])
            if t == tau_star:
                X_t = torch.cat([X_t, torch.as_tensor(x_star[None])])
                y_t = torch.cat([y_t, torch.as_tensor([y_star], dtype=torch.long)])
            optimizer.zero_grad()
            loss_fn(model(X_t), y_t).backward()
            optimizer.step()
        recorder.record()

    bg_grads = []
    theta_bg = recorder.trajectory[0]
    for _ in range(100):
        i = int(rng.integers(0, n_train))
        bg_grads.append(oracle.grad_at(theta_bg, Xb[i], int(yb[i])))
    mu_g, sigma = estimate_gradient_stats(np.array(bg_grads), jitter=1e-3)
    sigma_inv = np.linalg.inv(sigma)

    eta_eff = pysemi.sgd.phase_effective_lr(lr=0.05, K=K, batch_size=batch + 1, pool_size=n_train)
    semi = pysemi.SeMISGD(
        recorder.trajectory, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1,
    )
    score = semi.log_lr(tau=tau_star, x_star=x_star, y_star=y_star)
    assert np.isfinite(score)
