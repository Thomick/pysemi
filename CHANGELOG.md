# Changelog

All notable changes to pysemi are documented here. The project follows
[Semantic Versioning](https://semver.org/). The public API is everything
exported from `pysemi`, `pysemi.core`, `pysemi.sgd`, `pysemi.metrics`,
and `pysemi.torch`; underscored symbols are internal.

## [0.4.0] — 2026-09-09

### Fixed

- `simulate_noisy_empirical_means` drew fresh Laplace noise for every released
  cumulative mean, while every `compute_noisy_log_lr_*` function assumes the
  noise is drawn once per batch and carried forward. The two disagreed by a
  factor growing linearly in t, reaching 11x in the standard deviation of the
  recovered increment at t = 10, and a threshold realising a 0.050 Type-I error
  under the intended mechanism realised 0.353 under the simulator. The
  simulator now accumulates one draw per batch, which also makes the released
  sequence ε-DP by parallel composition over disjoint batches rather than
  requiring composition over T overlapping releases.
- `compute_epsilon_lower_bound_cp` applied Clopper-Pearson at the full
  confidence level to each of the hundreds of ROC points it then maximised
  over, so the returned value was not a bound at the stated confidence. Limits
  are now taken at level (1 − confidence)/(2K) over the K points supplied.
  Measured against a mechanism of known exact ε, the corrected bound breaches
  in 0 of 400 runs at 50, 500 and 2000 grid points, where the uncorrected one
  used 0.003 to 0.018 of a 0.02 budget and grew with the grid.

### Added

- `compute_epsilon_lower_bound_cp_grid` and `calibrate_threshold_grid`, the
  audit protocol with a threshold grid calibrated on rounds held out from the
  ones being bounded, so the thresholds do not depend on the data they score.
- `cp_upper`, the Clopper-Pearson upper limit, exposed for building audits.
- `laplace_sensitivity_range`, the sensitivity of a batch mean over a stated
  clipping range, for callers who need an actual ε-DP guarantee.
- Tests locking in the agreement of `semi_star_power_theoretical` with the
  exact Type-I and Type-II error expressions, and the calibration of the DP
  statistic against its simulator.
- GitHub Actions running the suite on Python 3.9 to 3.13, the linters and a
  build check, plus a release workflow publishing through PyPI trusted
  publishing.

### Changed

- `laplace_sensitivity_clipping` keeps returning σ/√n, and its docstring now
  says that this sets the noise to the standard deviation of a clean batch
  mean, which is a calibration convention and not a sensitivity derived from a
  clipping range.
- `pysemi.core.dp` documents the mechanism it assumes, and states that this
  analysis is a library extension rather than a result of the paper.
- Build requires setuptools 77, which the PEP 639 `license` string needs.
  Building against setuptools 68 previously failed outright.
- Dropped the mypy `python_version` pin, which made mypy refuse NumPy's
  recent stubs on Python 3.13.
- Examples no longer ship their generated plots, which running them rewrote.
- The sdist now carries `examples/` and `CHANGELOG.md` alongside `tests/`.
- Removed the internal experiment numbering from the README, the changelog and
  the example docstrings.

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
- 48 tests covering all modules, four runnable example scripts.
