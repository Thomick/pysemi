# Changelog

All notable changes to pysemi are documented here. The project follows
[Semantic Versioning](https://semver.org/). The public API is everything
exported from `pysemi`, `pysemi.core`, `pysemi.sgd`, `pysemi.metrics`,
and `pysemi.torch`; underscored symbols are internal.

## [0.3.0] — 2026-06-06

### Added

- Input validation on `SeMIStar`, `SeMIMax`, `SeMIUnif`, `SeMIStarDP`,
  `FinalObservation`, and `SeMISGD`: rejects σ ≤ 0, n < 2, ε ≤ 0,
  shape-mismatched θ/μ_g/Σ⁻¹, and η ≤ 0 at construction time with clear
  messages.
- `__repr__` on all user-facing classes for usable print output.
- `CHANGELOG.md`.
- mypy config (strict `Optional` handling, scipy/torch ignored as missing
  stubs); coverage reporting via pytest-cov.

### Changed

- `SeMIStarDP.power()` now raises `NotImplementedError` instead of silently
  returning the non-DP asymptotic power inherited from `SeMIStar`.
- `compute_epsilon_lower_bound_cp` / `_dkw` and the stratified variants
  now annotate `n_h0`, `n_h1`, `n_per_stratum` as `Optional[int]` rather
  than the incorrect `int = None`.
- `compute_log_lr_unif`, `compute_noisy_log_lr_unif`, and
  `SeMISGDUnif.log_lr` now use `scipy.special.logsumexp` instead of a
  hand-rolled max-shift.
- `SeMIMax.best_tau` delegates to `log_lr_and_tau` instead of recomputing
  the LR sweep.

### Fixed

- `TorchOracle._flatten_grad` now raises an explanatory `RuntimeError` if
  a parameter has no gradient, instead of a bare `AttributeError`.
- `TorchOracle` no longer uses `_` as a loop variable that is actually read.

## [0.2.0] — 2026-06-06

### Added

- Renamed classes to paper conventions: `SeMIStar`, `SeMIMax`, `SeMIUnif`,
  `SeMIStarDP`, `FinalObservation`; SGD variants `SeMISGD`, `SeMISGDMax`,
  `SeMISGDUnif`.
- `pysemi.torch` module for arbitrary-NN audits: `TorchOracle`,
  `TorchLastLayerOracle`, `TrajectoryRecorder`,
  `estimate_gradient_stats_from_loader`.
- New example `examples/example_torch_nn_audit.py` (small PyTorch MLP).
- New example `examples/example_torch_dpsgd_audit.py` (DP-SGD audit at
  fixed τ with per-method ε lower bounds, disjoint aggregation protocol).

### Changed

- All core function names mirror the paper: `compute_log_lr_t`,
  `compute_log_lr_tau`, `compute_log_lr_max`, `compute_log_lr_unif`, and
  their DP counterparts.

### Fixed

- `TorchLastLayerOracle` now handles modules that are themselves a single
  `nn.Linear` (empty submodule name).

## [0.1.0] — 2026-06-06

Initial release. Extracted the SeMI paper's algorithms into a
pip-installable library:

- `SeMIStar` family for the empirical-mean setting (clean and DP).
- `SeMISGD` family for parametric models with `SoftmaxOracle` and
  `LinearRegressionOracle`.
- Loss-only baselines (loss-diff, loss-ratio) and Jagielski baselines.
- ROC and privacy-audit metrics (Clopper-Pearson and DKW ε bounds, with
  stratified variants and bootstrap CIs).
- 48 tests covering all modules, four runnable example scripts mirroring
  exp1, exp2, exp3/15, and exp17.
