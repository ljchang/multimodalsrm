# Changelog

All notable changes to `multimodalsrm` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Lag scan and scan-seeded response lags.** `scan_lags(data)` estimates relative
  modality delays from the observations before fitting, by lagged canonical
  correlation between every pair of modalities, and returns a `LagScan` with
  reliability notes. `LagScan.seed_responses` starts free lags at the scanned delays
  with bounds of that lag ± a margin, and `MultimodalSRM(lag_init="scan")` does both
  at fit time. On synthetic recordings with true relative delays of 2, 5 and 10 s,
  default R-MSRM fits recovered them in 1 of 9 cases with no bound flagged and
  scan-initialized fits in 9 of 9. `multimodalsrm.bayesian.lag_priors` centres GP lag
  priors on the seeded lags, since the GP MAP search starts lags from their priors;
  with it a single GP start recovered delays of 5 and 10 s in 4 of 4 cases against
  0 of 4 from wide bounds. On the one empirical recording tested the scan found
  no reliable delay for any modality pair, and ordinary fits there reached the
  same lag from starts 12 s apart; an unreliable scan warns and leaves the
  configured lags and bounds unchanged.
- **`timing_report()` on both estimators.** Reports each modality's peak delay
  relative to a reference modality or to the across-modality mean, pairwise
  differences, the clock origin that was subtracted, and lags resting at a bound.
  Only relative timing is identified, so no modality needs a fixed lag. For
  posterior fits it summarizes peak delay per draw. `peak_delay(kernel)` gives the
  peak of any response family; for Gamma, DoubleGamma and BatemanSCR it is better
  determined than the `lag` onset parameter.
- `MultimodalSRM(kernel_tol=1e-4)` adds a response-step condition to convergence:
  no free response coordinate may have moved further than `kernel_tol` in the last
  iteration (seconds for lags, log units for positive parameters). Alternating
  updates can move a response by a small, nearly constant step for hundreds of
  iterations while every objective decrement stays below `tol`. In a synthetic
  eight-participant fit the objective-only rule reported convergence at iteration
  65 with a response width of 0.669 that later settled at 0.75. `kernel_tol=None`
  restores the objective-only rule, and earlier pickles load with it.
- `MultimodalSRM(kernel_extrapolation=True)` follows each iteration's response step
  further while refitted latents and maps keep lowering the complete objective. The
  objective and its stationary points are unchanged. The same synthetic fit reached
  the new criterion in 99 iterations instead of 373. Per-iteration `kernel_step` and
  `kernel_extrapolation` records are in `restart_diagnostics_`.
- `SearchConfig(relative_gradient_tolerance=1e-6)` adds an objective-relative MAP
  gradient criterion. `meets_gradient_tolerance` is now true when the largest
  physical projected gradient is at most `physical_gradient_tolerance` or at most
  `relative_gradient_tolerance * abs(objective)`. Diagnostics add
  `relative_projected_gradient` and `gradient_criterion`. The absolute threshold
  alone could not separate a converged large fit (gradient 0.037 at objective
  858,992) from a stalled start (gradient 60,800 at objective 1,195,617).
  `relative_gradient_tolerance=0` restores the absolute-only gate, and archives
  written before this option keep it.

### Changed

- **R-MSRM latent smoothness is now specified as a correlation time, with stronger
  defaults.** `MultimodalSRM(latent_timescale=2.0)` (timestamp units, normally
  seconds) replaces `temporal_strength`. The latent penalty is
  `latent_ridge * (mean z² + latent_timescale² * mean (dz/dt)²)`, and
  `latent_ridge` now defaults to 0.1. The earlier defaults (`latent_ridge=1e-4`,
  `temporal_strength=1e-2`) left latent nodes between observations nearly
  unconstrained. In synthetic sweeps over four sampling layouts and three noise
  levels the new defaults were much better when noise was moderate or high or one
  stream predicted another, and up to 0.03 R² worse at the lowest noise level.
  They are tuned on synthetic data only. Fits that relied on the old defaults will change; pass
  `temporal_strength=1e-2` to reproduce them.
- `max_observations` now bounds what each GP backend factorizes per run: eligible
  scalars for dense and spectral algebra, exact modality/time nodes times factors
  for grouped algebra, and nodes for grouped state-space filtering. Run baselines,
  correlated noise and scalar state-space updates keep the scalar count. The error
  reports the measured size and the alternatives.
- The GP error for a non-shared response names the modality and the fix, and the
  MAP convergence warning reports the gradient, objective and both limits.
- Grouped GP marginal prediction reuses each parameter draw's factorization
  across output features and evaluates response covariance at unique query
  times. Query batches and one-draw conditioning state bound temporary memory.
  White and OU noise, feature masks, and draw-specific latent rotations retain
  the same conditional moments; run-baseline predictions retain their existing
  calculation. Caches live only within the current prediction result.

### Deprecated

- `MultimodalSRM(temporal_strength=...)`. It still works, warns, keeps the earlier
  `latent_ridge=1e-4` default, and equals `latent_ridge * latent_timescale**2`.

### Performance

- R-MSRM response operators are shared across participants whose timestamps are
  identical within a run and modality, instead of being rebuilt per participant.
  The quadrature operator for Gamma, DoubleGamma, sampled and Identity responses
  integrates rows in chunks rather than one at a time and returns the same
  matrices. A synthetic eight-participant fit with a learned Gamma response fell
  from 482 s to 21 s, and its Gaussian counterpart from 34 s to 12 s, with
  identical objectives.

## [0.2.0] — 2026-09-22

### Added

- **Grouped state-space filtering and smoothing extended to learned responses.**
  Node grouping at exact `(modality, native timestamp)` pairs now also covers learned
  responses under strictly positive noise priors, including changing event order and
  coincident times. Noiseless support retains scalar updates.
- **Grouped fixed-response state-space filtering and smoothing.** Observations that
  share an exact `(modality, native timestamp)` are collapsed into single nodes for
  filtering and RTS smoothing, rather than repeating a Kalman update and storing a
  transition per observed scalar. In a synthetic fixed-Bateman case 8,002 scalar
  events become 257 nodes and complete fit time falls from 19.6 s to 8.0 s, with both
  fits converging to the same objective. Grouping engages automatically inside
  `linear_algebra="state_space"` when responses are fixed and noise support is
  strictly positive; learned responses and priors admitting zero variance keep scalar
  updates. Timestamps are compared exactly, with no binning or added jitter, and
  archive schemas and historical fit evidence are preserved.
- **`BatemanSCR` as the shared skin-conductance response for R and GP workflows.**
  Exposes positive rise/decay time constants and lag, with vectorized R cell
  integration, complex-step derivatives, and a compact two-state realization for
  differentiable state-space MAP. Retains the toolbox's 90-second finite support and
  continuous L2 normalization, and handles equal and nearly equal time constants
  without singular partial fractions. Dense and grouped GP require explicit response
  quadrature; state-space MAP checks a uniform bound for the truncated tail.
  It replaces `BachSCR` (see Removed); archives are never silently converted, so
  switching an existing analysis requires an explicit response change and a refit.
- **A `bayesian-cuda` extra, and one-time advice when a visible NVIDIA card goes
  unused.** The extra installs the pinned JAX CUDA plugin on Linux x86\_64 and
  resolves to the CPU runtime elsewhere. A fit process warns once when an NVIDIA
  driver is visible but JAX initialized its CPU backend, unless the platform was
  pinned through the environment.
- **A one-time warning when the CPU factorization callback runs oversubscribed.**
  The symmetric factorization runs LAPACK POTRF/POTRI inside a JAX host callback
  while XLA's own thread pool already occupies the cores. On a 64-core Linux
  workstation the BLAS default of one thread per core slowed a 2,727-dimension
  factorization by up to 10x relative to 16 threads. The callback now emits a
  `RuntimeWarning` once per process when the visible BLAS pool exceeds half the
  physical cores. No thread limits are changed and no numerical behavior changes.

- **Optional R-MSRM warm start for the first GP MAP restart.**
  `SearchConfig(r_init=True)` seeds the first full-training restart from a bounded
  CPU R-MSRM fit, which supplies loadings, offsets, residual noise variances and
  Gaussian response estimates. It is **off by default**: GP priors, native
  timestamps, remaining prior-quantile restarts and convergence checks are
  unchanged, and a numerical failure retains the historical start with a warning and
  a recorded reason. The measured benefit is scale-dependent — small synthetic cases
  show fewer iterations but no reduction in total time, and an evaluation on a real
  recording did not reproduce the larger synthetic gain — so enabling it is a
  per-analysis decision. Archives missing the setting restore it as false without
  rewriting saved fit evidence.

### Changed

- **Posterior mixing diagnostics are computed in batched NumPy over every quantity
  at once.** Rank-normalized split R-hat, bulk and 5%/95% tail ESS, Monte Carlo
  errors and BFMI reproduce ArviZ's definitions to rounding. The diagnostics phase
  of a 57,758-quantity posterior drops from 106 s to under 1 s at 15 draws, and
  from about 160 s to under 40 s at 1,200 draws. The summary is a small
  array-backed table rather than a pandas DataFrame, so ArviZ, xarray and pandas
  all leave the `bayesian` extra, and ArviZ joins the `test` extra as the
  reference implementation.

### Removed

- **Breaking: `BachSCR` is removed.** Use `BatemanSCR` for skin-conductance
  workflows. `BachSCR` is no longer importable or accepted by R or GP fitting, and
  its separate rational approximation, quadrature formula, parameter constraints and
  serialization registration are gone. GP archives containing Bach records now fail
  before model reconstruction with migration guidance: reproduce them in their
  original environment, or configure `BatemanSCR` and refit from the original
  observations. No fitted parameters or archives are converted automatically.
  Remaining response identifiers and archive schemas are unchanged.

## [0.1.0] — 2026-09-22

First package release of the extracted models.

### Added

- Extract R-MSRM and GP-MSRM production modules from the frozen source handoff,
  retaining model objectives, response conventions, priors and numerical behavior.
- Expose `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM`
  with optional Bayesian dependencies.
- Transfer focused synthetic core/Bayesian tests and add small R fitting and
  GP MAP/archive-replay examples.
- Document supported posterior workflows, numerical backends, legacy archive
  compatibility and the source-bound warmup checkpoint contract.
- Retain Linux/macOS installed-wheel and installed-sdist CI, integration/version
  gates and release-driven Trusted Publishing infrastructure.
- Add a searchable Zensical documentation site with GitHub Pages deployment.
- Separate fast PR checks from the full Linux/macOS release suite.

Test evidence belongs to the integration record and corresponding CI run; this
changelog makes no calibration or recovery claim. Learned FIR responses remain
deferred.

[Unreleased]: https://github.com/ljchang/multimodalsrm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ljchang/multimodalsrm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ljchang/multimodalsrm/releases/tag/v0.1.0
