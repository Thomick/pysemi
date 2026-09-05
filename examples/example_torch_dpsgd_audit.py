"""DP-SGD audit of a PyTorch MLP at a **fixed** insertion phase τ.

Mirrors exp17's disjoint-aggregation protocol:

- The dataset is shuffled once and split into T disjoint slices. Phase t uses
  only its slice (the "phase pool").
- Each phase runs K DP-SGD steps on the auditable last-layer parameters
  (per-sample gradient clipping at L2 norm C + Gaussian noise N(0, (σ·C)²) on
  the summed clipped gradient, then averaged).
- Under H1 the canary z⋆ = (x⋆, y⋆) is appended to the phase-τ slice only;
  under H0 it is never present. The canary appears in each step of phase τ
  with probability ≈ batch/(slice_size + 1) — matching the q factor in
  η_eff = K·q·lr that SeMI^SGD's plug-in formula assumes.
- τ is **fixed** across all simulations (the known-τ setting in which SeMI* is
  NP-optimal). Disjoint aggregation also enjoys parallel composition for the
  DP accountant: each phase may spend the full ε budget.

For each method we report the empirical AUC and the Clopper-Pearson and DKW
lower bounds on the audit ε at δ = 1e-4.

Notes on the small-scale demo:

- 200 simulations is far too few for tight ε bounds. CP bounds become
  informative around 1k–10k sims; DKW needs even more. At this scale most
  methods report ε ≈ 0; the loss_diff baseline is often the first to lift off
  because its score has higher SNR than the plug-in formula for a clipped
  canary.
- DP-SGD's per-sample clipping breaks SeMI^SGD's Gaussian-gradient assumption.
  The example clips both the canary gradient and the background gradient
  statistics to L2 norm C to keep the plug-in formula consistent with the
  training dynamics, but the approximation only sharpens at larger scale
  (more sims, longer trajectories, more phases) — exactly the regime of
  exp17 in the paper.

Requires PyTorch: `pip install pysemi[torch]`.

Run: python examples/example_torch_dpsgd_audit.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import pysemi
from pysemi.metrics import (
    compute_auc,
    compute_empirical_roc,
    compute_epsilon_lower_bound_cp,
    compute_epsilon_lower_bound_dkw,
)
from pysemi.sgd import (
    SeMISGD,
    SeMISGDMax,
    SeMISGDUnif,
    delta_attack_statistic,
    loss_diff_at_tau,
    max_loss_diff,
)
from pysemi.torch import TorchLastLayerOracle, TrajectoryRecorder


def build_model(in_dim: int, out_dim: int) -> nn.Module:
    """Single linear classifier — keeps the audit on raw inputs, no feature compression."""
    return nn.Linear(in_dim, out_dim)


def synthetic_data(n: int, in_dim: int, rng: np.random.Generator):
    X = np.vstack([
        rng.standard_normal((n // 2, in_dim)) - 0.3,
        rng.standard_normal((n - n // 2, in_dim)) + 0.3,
    ]).astype(np.float32)
    y = np.concatenate([
        np.zeros(n // 2, dtype=np.int64),
        np.ones(n - n // 2, dtype=np.int64),
    ])
    return X, y


def warm_start(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, lr: float):
    """Run a few full-data Adam steps so θ_0 is not random; all params are audited."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X); y_t = torch.as_tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()


class _ClippedOracle:
    """Wraps a GradientOracle, clipping its output to L2 norm `clip_C`."""

    def __init__(self, oracle, clip_C: float):
        self._oracle = oracle
        self._clip_C = float(clip_C)

    def grad_at(self, theta, x, y):
        g = self._oracle.grad_at(theta, x, y)
        norm = float(np.linalg.norm(g))
        return g if norm <= self._clip_C else g * (self._clip_C / norm)


def build_phase_pools(
    X: np.ndarray, y: np.ndarray,
    x_star: np.ndarray, y_star: int,
    T: int, tau: int,
    include_canary: bool,
    split_seed: int,
):
    """Disjoint aggregation: shuffle X then split into T slices; phase τ gets the canary under H1.

    Returns a list of (X_phase, y_phase) tuples of length T.
    """
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(X))
    slices = np.array_split(perm, T)
    x_can = x_star[None]
    y_can = np.array([y_star], dtype=np.int64)
    pools = []
    for t in range(1, T + 1):
        idx = slices[t - 1]
        Xp, yp = X[idx], y[idx]
        if include_canary and t == tau:
            Xp = np.concatenate([Xp, x_can])
            yp = np.concatenate([yp, y_can])
        pools.append((Xp, yp))
    return pools


def dpsgd_step(
    model: nn.Module,
    oracle: TorchLastLayerOracle,
    loss_fn,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    lr: float,
    clip_C: float,
    noise_sigma: float,
):
    """One DP-SGD step on the auditable params: per-sample clip + Gaussian noise."""
    n = X_batch.shape[0]
    per_sample = []
    for i in range(n):
        for p in oracle._params:  # noqa: SLF001
            p.grad = None
        loss_fn(model(X_batch[i:i + 1]), y_batch[i:i + 1]).backward()
        flat = torch.cat([p.grad.detach().reshape(-1) for p in oracle._params])
        per_sample.append(flat)
    G = torch.stack(per_sample)
    norms = G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    G_clipped = G * torch.clamp(clip_C / norms, max=1.0)
    summed = G_clipped.sum(dim=0)
    noise = torch.randn_like(summed) * (noise_sigma * clip_C)
    avg = (summed + noise) / n
    offset = 0
    with torch.no_grad():
        for p in oracle._params:  # noqa: SLF001
            size = p.numel()
            g = avg[offset:offset + size].reshape(p.shape)
            p.add_(g, alpha=-lr)
            offset += size


def run_phase_dpsgd(
    model, oracle, loss_fn,
    X_phase, y_phase, K, batch, lr, clip_C, noise_sigma, rng,
):
    """K DP-SGD steps on `(X_phase, y_phase)`; batches sampled uniformly from the phase pool."""
    n_pool = len(X_phase)
    for _ in range(K):
        idx = rng.choice(n_pool, batch, replace=False)
        X_t = torch.as_tensor(X_phase[idx]); y_t = torch.as_tensor(y_phase[idx])
        dpsgd_step(model, oracle, loss_fn, X_t, y_t, lr, clip_C, noise_sigma)


def simulate_run(
    seed: int,
    X, y, x_star, y_star,
    T, tau, K, batch, lr, clip_C, noise_sigma,
    in_dim, out_dim,
    include_canary: bool,
):
    """One paired DP-SGD training run; returns (trajectory, losses, oracle)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(in_dim, out_dim)
    warm_start(model, X, y, epochs=3, lr=1e-2)
    oracle = TorchLastLayerOracle(model)
    loss_fn = nn.CrossEntropyLoss()

    # Same split across H0/H1 so the only difference is canary presence in phase τ
    phase_pools = build_phase_pools(X, y, x_star, y_star, T, tau, include_canary, split_seed=12345)

    recorder = TrajectoryRecorder(oracle)
    recorder.record()
    losses = [oracle.loss(recorder.trajectory[-1], x_star[None], [y_star])]
    for t in range(1, T + 1):
        X_phase, y_phase = phase_pools[t - 1]
        run_phase_dpsgd(
            model, oracle, loss_fn,
            X_phase, y_phase, K, batch, lr, clip_C, noise_sigma, rng,
        )
        recorder.record()
        losses.append(oracle.loss(recorder.trajectory[-1], x_star[None], [y_star]))
    return recorder.trajectory, np.array(losses), oracle


def main():
    in_dim, out_dim = 16, 2
    n_data = 600              # split into T disjoint slices of ~100 each
    T = 6
    tau = T // 2 + 1          # fixed insertion phase (known-τ setting)
    K = 6
    batch = 16
    lr = 0.1
    clip_C = 1.0
    noise_sigma = 0.5         # DP-SGD noise multiplier σ
    n_sims = 200
    delta = 1e-4
    rng = np.random.default_rng(0)

    X, y = synthetic_data(n_data, in_dim, rng)
    slice_size = n_data // T
    pool_tau = slice_size + 1

    x_star = np.ones(in_dim, dtype=np.float32) * 6.0   # strong outlier
    y_star = 0                                          # mislabelled outlier canary

    print(f"DP-SGD audit (disjoint aggregation, fixed τ = {tau})")
    print(f"  σ = {noise_sigma}, C = {clip_C}, lr = {lr}, T = {T}, K = {K}, "
          f"batch = {batch}, slice = {slice_size}, n_sims = {n_sims}")
    print(f"  Canary appears with probability ≈ batch / (slice + 1) = "
          f"{batch / pool_tau:.3f} per step of phase τ "
          f"(expected K·q = {K * batch / pool_tau:.2f} appearances)\n")

    print(f"Reference H0 run to obtain θ_{{τ-1}} for background gradient stats...")
    ref_thetas, _, ref_oracle = simulate_run(
        99_999, X, y, x_star, y_star, T, tau, K, batch, lr, clip_C, noise_sigma,
        in_dim, out_dim, include_canary=False,
    )
    theta_bg = ref_thetas[tau - 1]  # θ_{τ-1} as in the paper

    # Estimate (μ_g, Σ_g) on CLIPPED per-sample gradients, since DP-SGD's update
    # uses per-sample clipping at L2 norm C. The plug-in LR's Gaussian-gradient
    # model then matches the actual training dynamics.
    clipped_ref = _ClippedOracle(ref_oracle, clip_C)
    rng_bg = np.random.default_rng(7777)
    bg_grads = np.array([
        clipped_ref.grad_at(theta_bg, X[i], int(y[i]))
        for i in (int(rng_bg.integers(0, n_data)) for _ in range(300))
    ])
    mu_g = bg_grads.mean(axis=0)
    sigma = np.cov(bg_grads, rowvar=False) + 1e-3 * np.eye(bg_grads.shape[1])
    sigma_inv = np.linalg.inv(sigma)

    eta_eff = pysemi.sgd.phase_effective_lr(lr=lr, K=K, batch_size=batch, pool_size=pool_tau)
    print(f"η_eff = K · (batch / pool_τ) · lr = {eta_eff:.4f}\n")

    print(f"Running {n_sims} paired H0/H1 DP-SGD simulations at fixed τ = {tau}...")
    rec = {k: ([], []) for k in [
        "SeMI^SGD (known τ)", "SeMI^SGD-max", "SeMI^SGD-unif",
        "loss_diff at τ", "max_loss_diff", "delta_diff",
    ]}
    for s in range(n_sims):
        thetas_h1, losses_h1, oracle_h1 = simulate_run(
            s, X, y, x_star, y_star, T, tau, K, batch, lr, clip_C, noise_sigma,
            in_dim, out_dim, include_canary=True,
        )
        thetas_h0, losses_h0, oracle_h0 = simulate_run(
            10_000 + s, X, y, x_star, y_star, T, tau, K, batch, lr, clip_C, noise_sigma,
            in_dim, out_dim, include_canary=False,
        )

        # Clip the canary gradient too: the plug-in formula assumes unclipped
        # Gaussian gradients but DP-SGD only uses each per-sample gradient up to
        # L2 norm C. Without the match δ_g and m⋆ are over-estimated.
        common_h1 = dict(theta_trajectory=thetas_h1, oracle=_ClippedOracle(oracle_h1, clip_C),
                         mu_g=mu_g, sigma_inv=sigma_inv, eta=eta_eff, n_batch=batch)
        common_h0 = dict(theta_trajectory=thetas_h0, oracle=_ClippedOracle(oracle_h0, clip_C),
                         mu_g=mu_g, sigma_inv=sigma_inv, eta=eta_eff, n_batch=batch)

        rec["SeMI^SGD (known τ)"][0].append(SeMISGD(**common_h0).log_lr(tau, x_star, y_star))
        rec["SeMI^SGD (known τ)"][1].append(SeMISGD(**common_h1).log_lr(tau, x_star, y_star))
        rec["SeMI^SGD-max"][0].append(SeMISGDMax(**common_h0).log_lr(x_star, y_star))
        rec["SeMI^SGD-max"][1].append(SeMISGDMax(**common_h1).log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][0].append(SeMISGDUnif(**common_h0).log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][1].append(SeMISGDUnif(**common_h1).log_lr(x_star, y_star))
        rec["loss_diff at τ"][0].append(loss_diff_at_tau(losses_h0, tau))
        rec["loss_diff at τ"][1].append(loss_diff_at_tau(losses_h1, tau))
        rec["max_loss_diff"][0].append(max_loss_diff(losses_h0)[0])
        rec["max_loss_diff"][1].append(max_loss_diff(losses_h1)[0])
        rec["delta_diff"][0].append(delta_attack_statistic(losses_h0, "diff")[0])
        rec["delta_diff"][1].append(delta_attack_statistic(losses_h1, "diff")[0])
        if (s + 1) % 10 == 0:
            print(f"  {s + 1}/{n_sims} sims done")

    print(f"\n{'method':<22} {'AUC':>6}  {'ε_CP':>6}  {'ε_DKW':>6}")
    print("-" * 50)
    for name, (h0, h1) in rec.items():
        h0_a, h1_a = np.array(h0), np.array(h1)
        fpr, tpr = compute_empirical_roc(h0_a, h1_a)
        auc = compute_auc(fpr, tpr)
        eps_cp = compute_epsilon_lower_bound_cp(fpr, tpr, delta=delta, n_h0=n_sims, n_h1=n_sims)
        eps_dkw = compute_epsilon_lower_bound_dkw(fpr, tpr, delta=delta, n_h0=n_sims, n_h1=n_sims)
        print(f"{name:<22} {auc:>6.3f}  {eps_cp:>6.3f}  {eps_dkw:>6.3f}")


if __name__ == "__main__":
    main()
