"""SeMI* power curve vs. FinalObservation baseline (mirror of paper exp1).

At a fixed total sample size n·T and τ = T/2, plot theoretical and empirical
power of SeMI* and of the FinalObservation snapshot baseline as T grows.

Run from the project root:
    python examples/example_semi_star_power_curve.py
"""

from __future__ import annotations

import numpy as np

import pysemi
from pysemi.metrics import compute_tpr_at_low_fpr


def empirical_semi_star_power(test, T, tau_star, n_sims, alpha, rng):
    h0, h1 = [], []
    for _ in range(n_sims):
        h0.append(test.log_lr(test.simulate(T, tau_star=None, rng=rng), tau=tau_star))
        h1.append(test.log_lr(test.simulate(T, tau_star=tau_star, rng=rng), tau=tau_star))
    return compute_tpr_at_low_fpr(np.array(h0), np.array(h1), target_fpr=alpha)


def main():
    mu, sigma, z_star = 0.0, 1.0, 4.0
    n_total = 600
    Ts = [2, 5, 10, 15, 20, 30]
    alpha = 0.05
    n_sims = 800
    rng = np.random.default_rng(0)

    print(f"{'T':>4} {'n':>5} {'tau':>4}  {'SeMI* (thy)':>11}  {'SeMI* (emp)':>11}  "
          f"{'FinalObs (thy)':>14}  {'FinalObs (emp)':>14}")
    for T in Ts:
        n = n_total // T
        tau = T // 2 + 1
        seq = pysemi.SeMIStar(mu, sigma, z_star, n)
        fo = pysemi.FinalObservation(mu, sigma, z_star, n, T)

        p_seq_thy = seq.power(alpha, T, tau)
        p_obs_thy = fo.power(alpha)
        p_seq_emp = empirical_semi_star_power(seq, T, tau, n_sims, alpha, rng)

        obs_h0, obs_h1 = [], []
        for _ in range(n_sims):
            obs_h0.append(fo.log_lr(seq.simulate(T, tau_star=None, rng=rng)[-1]))
            obs_h1.append(fo.log_lr(seq.simulate(T, tau_star=tau, rng=rng)[-1]))
        p_obs_emp = compute_tpr_at_low_fpr(np.array(obs_h0), np.array(obs_h1), target_fpr=alpha)

        print(f"{T:>4d} {n:>5d} {tau:>4d}  {p_seq_thy:>11.3f}  {p_seq_emp:>11.3f}  "
              f"{p_obs_thy:>14.3f}  {p_obs_emp:>14.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to save a plot)")
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    p_seq_curve, p_obs_curve = [], []
    for T in Ts:
        n = n_total // T
        tau = T // 2 + 1
        p_seq_curve.append(pysemi.SeMIStar(mu, sigma, z_star, n).power(alpha, T, tau))
        p_obs_curve.append(pysemi.FinalObservation(mu, sigma, z_star, n, T).power(alpha))
    ax.plot(Ts, p_seq_curve, "o-", label="SeMI* (theoretical)")
    ax.plot(Ts, p_obs_curve, "s--", label="FinalObservation")
    ax.set_xlabel("T (number of phases)")
    ax.set_ylabel(f"power at α = {alpha}")
    ax.set_title(f"Power vs T at fixed n·T = {n_total}, z⋆ = {z_star}σ")
    ax.legend()
    fig.tight_layout()
    fig.savefig("examples/example_semi_star_power_curve.png", dpi=120)
    print("\nSaved figure to examples/example_semi_star_power_curve.png")


if __name__ == "__main__":
    main()
