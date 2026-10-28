# pysemi

Sequential Membership Inference (SeMI) attacks for auditing models that go
through multiple updates. Library companion to *Sequential Membership
Inference Attacks* (Michel, Basu, Kaufmann, NeurIPS 2026).

Naming follows the paper:

| Class                       | Statistic                                            | NP-optimal against    |
| --------------------------- | ---------------------------------------------------- | --------------------- |
| `SeMIStar`                  | log LR_τ at known τ                                  | H_0 vs H_1^τ          |
| `SeMIMax`                   | max_t log LR_t (GLR)                                 | H_0 vs ⋃_t H_1^t      |
| `SeMIUnif`                  | log (1/T) Σ_t LR_t                                   | H_0 vs τ ~ Unif([T])  |
| `SeMIStarDP`                | SeMI* under Laplace-DP cumulative-mean release       | DP audits             |
| `FinalObservation`          | snapshot LR from the final cumulative mean           | snapshot baseline     |
| `SeMISGD` / `Max` / `Unif`  | plug-in log LR variants for (DP-)SGD-trained models  | Gaussian-gradient NP  |

## Installation

```bash
pip install pysemi              # core (numpy + scipy)
pip install pysemi[torch]       # + arbitrary-NN audits (PyTorch)
pip install pysemi[analysis]    # + matplotlib for example plots
```

## Quick start — empirical-mean SeMI variants

```python
import pysemi
import numpy as np

rng = np.random.default_rng(0)

# Known τ (NP-optimal): SeMI*
test = pysemi.SeMIStar(mu=0.0, sigma=1.0, z_star=4.0, n=50)
means = test.simulate(T=10, tau_star=3, rng=rng)
test.log_lr(means, tau=3)
test.power(alpha=0.05, T=10, tau=3)

# Unknown τ
pysemi.SeMIMax(mu=0.0, sigma=1.0, z_star=4.0, n=50).log_lr(means)
pysemi.SeMIUnif(mu=0.0, sigma=1.0, z_star=4.0, n=50).log_lr(means)

# Laplace-DP release of the running mean
dp = pysemi.SeMIStarDP(mu=0.0, sigma=1.0, z_star=4.0, n=50, epsilon=1.0)
noisy_means = dp.simulate(T=10, tau_star=3, rng=rng)
dp.log_lr(noisy_means, tau=3)        # SeMI*
dp.log_lr_max(noisy_means)           # SeMI^max
dp.log_lr_unif(noisy_means)          # SeMI^unif

# Snapshot baseline
pysemi.FinalObservation(0.0, 1.0, 4.0, n=50, T=10).log_lr(noisy_means[-1])
```

## SeMI^SGD — auditing arbitrary parametric models

`SeMISGD`, `SeMISGDMax` and `SeMISGDUnif` all consume a trajectory of flat
parameter snapshots (θ_0, …, θ_T) saved at phase boundaries, plus a
`GradientOracle` that exposes the per-sample gradient at the target.

Three oracles ship with the library:

- `SoftmaxOracle` — numpy last-layer multinomial-logistic oracle. Used for
  the Fashion-MNIST and Purchase-100 audits of the paper.
- `LinearRegressionOracle` — analytic ½(θᵀx − y)² gradient. The
  SGD-linear baseline of the paper is just `SeMISGD` with this oracle (no
  separate code path).
- `pysemi.torch.TorchOracle` / `TorchLastLayerOracle` — wraps an
  `nn.Module` for arbitrary-NN audits (see below).

```python
from pysemi import SeMISGD, SeMISGDMax, SeMISGDUnif
from pysemi.sgd import SoftmaxOracle, estimate_gradient_stats, phase_effective_lr

oracle = SoftmaxOracle(input_dim=784, num_classes=10)

# 1) Pre-estimate background gradient statistics
mu_g, sigma = estimate_gradient_stats(background_gradient_matrix)
sigma_inv = np.linalg.inv(sigma)

# 2) Each variant takes the same arguments
common = dict(theta_trajectory=thetas, oracle=oracle, mu_g=mu_g, sigma_inv=sigma_inv,
              eta=phase_effective_lr(lr, K, batch_size, pool_size), n_batch=batch_size)
SeMISGD(**common).log_lr(tau, x_star, y_star)        # known τ
SeMISGDMax(**common).log_lr(x_star, y_star)          # max_t log LR_t
SeMISGDUnif(**common).log_lr(x_star, y_star)         # log mean exp(log LR_t)
```

## Auditing an arbitrary PyTorch model

`pysemi.torch` provides the glue for end-to-end audits of any `nn.Module`:

```python
from pysemi.torch import TorchOracle, TorchLastLayerOracle, TrajectoryRecorder, estimate_gradient_stats_from_loader
import pysemi

# Audit only the last nn.Linear (as the paper does), or audit any parameter subset
oracle = TorchLastLayerOracle(model)
# or:  TorchOracle(model, param_names=["classifier.weight", "classifier.bias"])

# 1) Record θ at each phase boundary during training
recorder = TrajectoryRecorder(oracle)
recorder.record()                                 # θ_0
for phase in range(1, T + 1):
    for _ in range(K):
        # one SGD step on a batch
        ...
    recorder.record()                             # θ_phase

# 2) Pre-compute background gradient stats from a DataLoader at θ_{τ-1}
mu_g, sigma = estimate_gradient_stats_from_loader(
    oracle, background_loader, theta=recorder.trajectory[tau - 1], n_samples=2000,
)
sigma_inv = np.linalg.inv(sigma)

# 3) Run the attack
score = pysemi.SeMISGD(
    recorder.trajectory, oracle, mu_g, sigma_inv,
    eta=pysemi.sgd.phase_effective_lr(lr, K, batch_size, pool_size),
    n_batch=batch_size,
).log_lr(tau, x_star, y_star)
```

See `examples/example_torch_nn_audit.py` for a complete pipeline: a small
MLP audited via `TorchLastLayerOracle` over 6 phases of training.

## Metrics

- `pysemi.metrics.compute_empirical_roc`, `compute_auc`,
  `compute_tpr_at_low_fpr`, `compute_balanced_accuracy`.
- `compute_epsilon_lower_bound_cp` / `_dkw` (and `_stratified` variants for
  τ-informed adversaries) — Clopper-Pearson and DKW lower bounds on ε for
  (ε, δ)-DP audits, matching the estimators used in the paper.
- `bootstrap_epsilon_ci` and `subsample_epsilon_ci` for CIs.

## Examples (miniature paper reproductions)

| Script                             | What it shows                                                  |
| ---------------------------------- | -------------------------------------------------------------- |
| `example_semi_star_power_curve.py` | Power vs T at fixed n·T: SeMI* stays flat, FinalObservation decays |
| `example_glr_unknown_tau.py`       | SeMI^max vs SeMI^unif vs FinalObservation under random τ        |
| `example_dp_audit.py`              | DP SeMI* empirical ε lower bounds vs. the true budget           |
| `example_semi_sgd_softmax.py`      | SeMI^SGD on a numpy softmax last-layer audit                    |
| `example_torch_nn_audit.py`        | SeMI^SGD on a PyTorch MLP, full pipeline                        |
| `example_torch_dpsgd_audit.py`     | DP-SGD training with per-method ε lower bounds                  |

## Development

```bash
pip install -e ".[dev,torch,analysis]"
pytest             # 89 tests, ~6 s with torch
ruff check pysemi  # lint
mypy pysemi        # type-check
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Stability

pysemi is pre-1.0. The public API is everything exported from `pysemi`,
`pysemi.core`, `pysemi.sgd`, `pysemi.metrics`, and `pysemi.torch`.
Underscored symbols are internal. `pysemi.torch` is not imported by
`import pysemi`, since torch is an optional dependency, so import it
explicitly. We follow semantic versioning, so breaking changes happen on
minor bumps before 1.0, with migration notes in the changelog.

Not everything here is in the paper. The paper covers the noiseless
empirical-mean mechanism, the SeMI^SGD extension and the DP-SGD audits.
The Laplace-noised empirical-mean mechanism behind `SeMIStarDP` and
`pysemi.core.dp` is a library extension, documented in the module
docstring of `pysemi/core/dp.py`.

## Citing

```bibtex
@inproceedings{michel2026semi,
  author    = {Michel, Thomas and Basu, Debabrota and Kaufmann, Emilie},
  title     = {Sequential Membership Inference Attacks},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

MIT.
