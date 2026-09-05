"""SeMI^max vs SeMI^unif vs FinalObservation under random τ (mirror of exp2).

Compares the GLR variant SeMI^max, the uniform-prior variant SeMI^unif, and the
FinalObservation snapshot baseline by ROC and AUC when τ is drawn uniformly at
random in {1, …, T}. SeMI^max and SeMI^unif are expected to stay close (within
log T of each other) and to dominate FinalObservation.

Run: python examples/example_glr_unknown_tau.py
"""

from __future__ import annotations

import numpy as np

import pysemi
from pysemi.metrics import compute_auc, compute_empirical_roc


def main():
    mu, sigma, z_star = 0.0, 1.0, 8.0
    n, T = 40, 10
    n_sims = 2000
    rng = np.random.default_rng(0)

    semi_max = pysemi.SeMIMax(mu, sigma, z_star, n)
    semi_unif = pysemi.SeMIUnif(mu, sigma, z_star, n)
    final_obs = pysemi.FinalObservation(mu, sigma, z_star, n, T)

    max_h0, max_h1, unif_h0, unif_h1, fo_h0, fo_h1 = [], [], [], [], [], []
    for _ in range(n_sims):
        means_h0 = semi_max.simulate(T, tau_star=None, rng=rng)
        tau = int(rng.integers(1, T + 1))
        means_h1 = semi_max.simulate(T, tau_star=tau, rng=rng)

        max_h0.append(semi_max.log_lr(means_h0))
        max_h1.append(semi_max.log_lr(means_h1))
        unif_h0.append(semi_unif.log_lr(means_h0))
        unif_h1.append(semi_unif.log_lr(means_h1))
        fo_h0.append(final_obs.log_lr(means_h0[-1]))
        fo_h1.append(final_obs.log_lr(means_h1[-1]))

    rocs = {
        "SeMI^max": (np.array(max_h0), np.array(max_h1)),
        "SeMI^unif": (np.array(unif_h0), np.array(unif_h1)),
        "FinalObservation": (np.array(fo_h0), np.array(fo_h1)),
    }

    print(f"{'method':<18} {'AUC':>6}")
    for name, (h0, h1) in rocs.items():
        fpr, tpr = compute_empirical_roc(h0, h1)
        print(f"{name:<18} {compute_auc(fpr, tpr):>6.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (h0, h1) in rocs.items():
        fpr, tpr = compute_empirical_roc(h0, h1)
        ax.plot(fpr, tpr, label=f"{name} (AUC={compute_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Unknown-τ SeMI variants, T = {T}, n = {n}, z⋆ = {z_star}σ")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("examples/example_glr_unknown_tau.png", dpi=120)
    print("\nSaved figure to examples/example_glr_unknown_tau.png")


if __name__ == "__main__":
    main()
