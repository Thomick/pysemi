"""DP SeMI* privacy audit, empirical ε lower bound vs. true budget.

For a Laplace mechanism releasing the running mean at each batch, we run the DP
SeMI^max attack and report the Clopper-Pearson and DKW lower bounds on ε.

Run: python examples/example_dp_audit.py
"""

from __future__ import annotations

import numpy as np

import pysemi
from pysemi.metrics import (
    compute_empirical_roc,
    compute_epsilon_lower_bound_cp,
    compute_epsilon_lower_bound_dkw,
)


def main():
    mu, sigma, z_star = 0.0, 1.0, 10.0
    n, T = 30, 10
    n_sims = 4000
    rng = np.random.default_rng(0)

    print(f"{'ε_true':>9}  {'CP':>6}  {'DKW':>6}  (lower bounds, n_sims = {n_sims})")
    for eps_true in [0.5, 1.0, 2.0, 5.0]:
        test = pysemi.SeMIStarDP(mu, sigma, z_star, n, epsilon=eps_true)
        h0, h1 = [], []
        for _ in range(n_sims):
            means_h0 = test.simulate(T, tau_star=None, rng=rng)
            tau = int(rng.integers(1, T + 1))
            means_h1 = test.simulate(T, tau_star=tau, rng=rng)
            h0.append(test.log_lr_max(means_h0))
            h1.append(test.log_lr_max(means_h1))
        h0, h1 = np.array(h0), np.array(h1)
        fpr, tpr = compute_empirical_roc(h0, h1)
        eps_cp = compute_epsilon_lower_bound_cp(fpr, tpr, delta=1e-4, n_h0=n_sims, n_h1=n_sims)
        eps_dkw = compute_epsilon_lower_bound_dkw(fpr, tpr, delta=1e-4, n_h0=n_sims, n_h1=n_sims)
        print(f"{eps_true:>9.2f}  {eps_cp:>6.3f}  {eps_dkw:>6.3f}")


if __name__ == "__main__":
    main()
