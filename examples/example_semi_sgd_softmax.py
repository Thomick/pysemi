"""SeMI^SGD on a numpy softmax last-layer audit, no PyTorch needed.

Trains a numpy softmax classifier on synthetic 2-class Gaussian data over T
phases. Under H1 the canary z⋆ = (x⋆, y⋆) is inserted at a uniformly random
phase τ; under H0 never. Compares SeMI^SGD (known τ), SeMI^SGD-max,
SeMI^SGD-unif, the loss-only baselines, and the Jagielski delta attack.

Run: python examples/example_semi_sgd_softmax.py
"""

from __future__ import annotations

import numpy as np

import pysemi
from pysemi.metrics import compute_auc, compute_empirical_roc
from pysemi.sgd import (
    SeMISGD,
    SeMISGDMax,
    SeMISGDUnif,
    SoftmaxOracle,
    delta_attack_statistic,
    estimate_gradient_stats,
    loss_diff_at_tau,
    max_loss_diff,
)


def _train_phase(theta, oracle, X, y, lr, K, batch, rng):
    for _ in range(K):
        idx = rng.choice(len(X), batch, replace=False)
        per_sample = np.array([oracle.grad_at(theta, X[i], y[i]) for i in idx])
        theta = theta - lr * per_sample.mean(axis=0)
    return theta


def simulate_run(rng, X, y, x_star, y_star, T, K, lr, batch, pool, oracle, insert_at):
    theta = rng.standard_normal(oracle.n_params) * 0.01
    thetas = [theta.copy()]
    losses = [oracle.loss(theta, x_star, np.atleast_1d(y_star))]
    for t in range(1, T + 1):
        idx = rng.choice(len(X), pool, replace=False)
        Xp, yp = X[idx], y[idx]
        if insert_at == t:
            Xp = np.concatenate([Xp, np.atleast_2d(x_star)], axis=0)
            yp = np.concatenate([yp, np.atleast_1d(y_star)], axis=0)
        theta = _train_phase(theta, oracle, Xp, yp, lr, K, batch, rng)
        thetas.append(theta.copy())
        losses.append(oracle.loss(theta, x_star, np.atleast_1d(y_star)))
    return thetas, np.array(losses)


def main():
    rng = np.random.default_rng(0)
    input_dim, num_classes = 8, 2
    n_total = 1500
    T, K, batch, pool = 8, 5, 16, 200
    lr = 0.05
    n_sims = 60

    mean0 = -np.ones(input_dim) * 0.3
    mean1 = np.ones(input_dim) * 0.3
    X = np.vstack(
        [
            rng.standard_normal((n_total // 2, input_dim)) + mean0,
            rng.standard_normal((n_total // 2, input_dim)) + mean1,
        ]
    )
    y = np.concatenate([np.zeros(n_total // 2, dtype=int), np.ones(n_total // 2, dtype=int)])

    x_star = np.ones(input_dim) * 5.0
    y_star = 0  # mislabelled outlier

    oracle = SoftmaxOracle(input_dim=input_dim, num_classes=num_classes)

    print("Estimating background gradient stats...")
    bg = np.array(
        [
            oracle.grad_at(
                rng.standard_normal(oracle.n_params) * 0.05,
                X[int(rng.integers(0, n_total))],
                y[int(rng.integers(0, n_total))],
            )
            for _ in range(800)
        ]
    )
    mu_g, sigma = estimate_gradient_stats(bg, jitter=1e-3)
    sigma_inv = np.linalg.inv(sigma)
    eta_eff = pysemi.sgd.phase_effective_lr(lr=lr, K=K, batch_size=batch, pool_size=pool)
    print(f"η_eff = K·(batch/pool)·lr = {eta_eff:.4f}\n")

    print("Running paired H0/H1 simulations...")
    rec = {
        k: ([], [])
        for k in [
            "SeMI^SGD (known τ)",
            "SeMI^SGD-max",
            "SeMI^SGD-unif",
            "loss_diff at τ",
            "max_loss_diff",
            "delta_diff",
        ]
    }
    for _ in range(n_sims):
        tau = int(rng.integers(1, T + 1))
        thetas_h1, losses_h1 = simulate_run(
            rng, X, y, x_star, y_star, T, K, lr, batch, pool, oracle, insert_at=tau
        )
        thetas_h0, losses_h0 = simulate_run(
            rng, X, y, x_star, y_star, T, K, lr, batch, pool, oracle, insert_at=None
        )

        sgd_h1 = SeMISGD(thetas_h1, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)
        sgd_h0 = SeMISGD(thetas_h0, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)
        sgd_max_h1 = SeMISGDMax(thetas_h1, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)
        sgd_max_h0 = SeMISGDMax(thetas_h0, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)
        sgd_unif_h1 = SeMISGDUnif(thetas_h1, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)
        sgd_unif_h0 = SeMISGDUnif(thetas_h0, oracle, mu_g, sigma_inv, eta=eta_eff, n_batch=batch)

        rec["SeMI^SGD (known τ)"][0].append(sgd_h0.log_lr(tau, x_star, y_star))
        rec["SeMI^SGD (known τ)"][1].append(sgd_h1.log_lr(tau, x_star, y_star))
        rec["SeMI^SGD-max"][0].append(sgd_max_h0.log_lr(x_star, y_star))
        rec["SeMI^SGD-max"][1].append(sgd_max_h1.log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][0].append(sgd_unif_h0.log_lr(x_star, y_star))
        rec["SeMI^SGD-unif"][1].append(sgd_unif_h1.log_lr(x_star, y_star))

        rec["loss_diff at τ"][0].append(loss_diff_at_tau(losses_h0, tau))
        rec["loss_diff at τ"][1].append(loss_diff_at_tau(losses_h1, tau))
        rec["max_loss_diff"][0].append(max_loss_diff(losses_h0)[0])
        rec["max_loss_diff"][1].append(max_loss_diff(losses_h1)[0])
        rec["delta_diff"][0].append(delta_attack_statistic(losses_h0, "diff")[0])
        rec["delta_diff"][1].append(delta_attack_statistic(losses_h1, "diff")[0])

    print(f"\n{'method':<22} {'AUC':>6}")
    for name, (h0, h1) in rec.items():
        fpr, tpr = compute_empirical_roc(np.array(h0), np.array(h1))
        print(f"{name:<22} {compute_auc(fpr, tpr):>6.3f}")


if __name__ == "__main__":
    main()
