"""SeMI^SGD audit of an arbitrary PyTorch neural network.

End-to-end example: trains a small MLP on synthetic 2-class data over T phases.
Under H1 the canary z⋆ = (x⋆, y⋆) is inserted at a uniformly random phase τ;
under H0 never. The last linear layer is the auditable parameter subset
(matching exp17). Reports AUCs for SeMI^SGD known-τ, SeMI^SGD-max,
SeMI^SGD-unif, and the loss-only and Jagielski baselines.

Requires PyTorch: `pip install pysemi[torch]`.

Run: python examples/example_torch_nn_audit.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import pysemi
from pysemi.metrics import compute_auc, compute_empirical_roc
from pysemi.sgd import (
    SeMISGD,
    SeMISGDMax,
    SeMISGDUnif,
    delta_attack_statistic,
    loss_diff_at_tau,
    max_loss_diff,
)
from pysemi.torch import (
    TorchLastLayerOracle,
    TrajectoryRecorder,
    estimate_gradient_stats_from_loader,
)


def build_model(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """Small backbone + linear head; the head is the auditable layer."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def synthetic_data(n: int, in_dim: int, rng: np.random.Generator):
    """Two Gaussian blobs with small mean separation."""
    means = np.array([[-0.3], [0.3]]) * np.ones((1, in_dim))
    X = np.vstack([
        rng.standard_normal((n // 2, in_dim)) + means[0],
        rng.standard_normal((n - n // 2, in_dim)) + means[1],
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(n // 2, dtype=np.int64),
                         np.ones(n - n // 2, dtype=np.int64)])
    return X, y


def pretrain_backbone(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 3, lr: float = 1e-2):
    """Pretrain the full model briefly to get a non-trivial backbone."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()


def freeze_backbone(model: nn.Module, oracle: TorchLastLayerOracle) -> None:
    """Freeze everything except the auditable last-layer parameters."""
    auditable = {id(p) for p in oracle._params}  # noqa: SLF001 (internal access OK for example)
    for p in model.parameters():
        p.requires_grad = id(p) in auditable


def run_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    X: np.ndarray, y: np.ndarray,
    K: int, batch: int,
    canary: tuple | None,
    rng: np.random.Generator,
):
    """K SGD steps; optionally inject (x⋆, y⋆) into every step."""
    for _ in range(K):
        idx = rng.choice(len(X), batch, replace=False)
        X_t = torch.as_tensor(X[idx]); y_t = torch.as_tensor(y[idx])
        if canary is not None:
            x_star, y_star = canary
            X_t = torch.cat([X_t, torch.as_tensor(x_star[None])])
            y_t = torch.cat([y_t, torch.as_tensor([y_star], dtype=torch.long)])
        optimizer.zero_grad()
        loss_fn(model(X_t), y_t).backward()
        optimizer.step()


def simulate_run(
    rng_seed: int,
    X: np.ndarray, y: np.ndarray,
    x_star: np.ndarray, y_star: int,
    T: int, K: int, batch: int, lr: float,
    in_dim: int, hidden: int, out_dim: int,
    insert_at: int | None,
):
    """One paired training run; returns (trajectory, losses_on_target, oracle, eta_eff)."""
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)
    model = build_model(in_dim, hidden, out_dim)
    pretrain_backbone(model, X, y, epochs=3, lr=1e-2)
    oracle = TorchLastLayerOracle(model)
    freeze_backbone(model, oracle)
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=lr,
    )
    loss_fn = nn.CrossEntropyLoss()

    recorder = TrajectoryRecorder(oracle)
    losses = []
    recorder.record()
    losses.append(oracle.loss(recorder.trajectory[-1], x_star[None], [y_star]))
    canary = (x_star, y_star)
    for t in range(1, T + 1):
        run_phase(model, optimizer, loss_fn, X, y, K, batch,
                  canary if (insert_at == t) else None, rng)
        recorder.record()
        losses.append(oracle.loss(recorder.trajectory[-1], x_star[None], [y_star]))
    return recorder.trajectory, np.array(losses), oracle


def main():
    in_dim, hidden, out_dim = 16, 24, 2
    n_data = 1500
    T = 6
    K = 3
    batch = 32
    lr = 0.05
    n_sims = 40
    rng = np.random.default_rng(0)

    X, y = synthetic_data(n_data, in_dim, rng)

    # Strong outlier canary
    x_star = np.ones(in_dim, dtype=np.float32) * 4.0
    y_star = 0

    # Estimate background gradient stats from a held-out loader using a
    # frozen backbone trained on the same distribution
    print("Setting up reference model and estimating background gradient stats...")
    torch.manual_seed(999)
    ref_model = build_model(in_dim, hidden, out_dim)
    pretrain_backbone(ref_model, X, y, epochs=3, lr=1e-2)
    ref_oracle = TorchLastLayerOracle(ref_model)
    theta_bg = ref_oracle.get_params()

    bg_loader = ((X[i:i + 32], y[i:i + 32]) for i in range(0, n_data, 32))
    mu_g, sigma = estimate_gradient_stats_from_loader(
        ref_oracle, bg_loader, theta_bg, n_samples=300, jitter=1e-3,
    )
    sigma_inv = np.linalg.inv(sigma)
    eta_eff = pysemi.sgd.phase_effective_lr(lr=lr, K=K, batch_size=batch + 1, pool_size=n_data)
    print(f"η_eff = K·(batch/pool)·lr = {eta_eff:.4f}\n")

    print(f"Running {n_sims} paired H0/H1 simulations on an MLP with T = {T} phases...")
    rec = {k: ([], []) for k in [
        "SeMI^SGD (known τ)", "SeMI^SGD-max", "SeMI^SGD-unif",
        "loss_diff at τ", "max_loss_diff", "delta_diff",
    ]}
    for s in range(n_sims):
        tau = int(rng.integers(1, T + 1))
        thetas_h1, losses_h1, oracle_h1 = simulate_run(
            s, X, y, x_star, y_star, T, K, batch, lr, in_dim, hidden, out_dim, insert_at=tau,
        )
        thetas_h0, losses_h0, oracle_h0 = simulate_run(
            10000 + s, X, y, x_star, y_star, T, K, batch, lr, in_dim, hidden, out_dim, insert_at=None,
        )

        h1 = SeMISGD(thetas_h1, oracle_h1, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)
        h0 = SeMISGD(thetas_h0, oracle_h0, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)
        h1_max = SeMISGDMax(thetas_h1, oracle_h1, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)
        h0_max = SeMISGDMax(thetas_h0, oracle_h0, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)
        h1_unif = SeMISGDUnif(thetas_h1, oracle_h1, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)
        h0_unif = SeMISGDUnif(thetas_h0, oracle_h0, mu_g, sigma_inv, eta=eta_eff, n_batch=batch + 1)

        rec["SeMI^SGD (known τ)"][0].append(h0.log_lr(tau, x_star, y_star))
        rec["SeMI^SGD (known τ)"][1].append(h1.log_lr(tau, x_star, y_star))
        rec["SeMI^SGD-max"][0].append(h0_max.log_lr(x_star, y_star))
        rec["SeMI^SGD-max"][1].append(h1_max.log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][0].append(h0_unif.log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][1].append(h1_unif.log_lr(x_star, y_star))
        rec["loss_diff at τ"][0].append(loss_diff_at_tau(losses_h0, tau))
        rec["loss_diff at τ"][1].append(loss_diff_at_tau(losses_h1, tau))
        rec["max_loss_diff"][0].append(max_loss_diff(losses_h0)[0])
        rec["max_loss_diff"][1].append(max_loss_diff(losses_h1)[0])
        rec["delta_diff"][0].append(delta_attack_statistic(losses_h0, "diff")[0])
        rec["delta_diff"][1].append(delta_attack_statistic(losses_h1, "diff")[0])

    print(f"\n{'method':<24} {'AUC':>6}")
    for name, (h0, h1) in rec.items():
        fpr, tpr = compute_empirical_roc(np.array(h0), np.array(h1))
        print(f"{name:<24} {compute_auc(fpr, tpr):>6.3f}")


if __name__ == "__main__":
    main()
