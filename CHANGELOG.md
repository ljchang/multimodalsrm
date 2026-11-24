# Changelog

All notable changes to `multimodalsrm` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`BayesianMultimodalSRM.timing_report(uncertainty="laplace")`.** A MAP fit gave
  lags without any measure of uncertainty. The option adds model-based standard
  errors for each lag relative to the reference modality and for each pairwise lag
  difference, from the inverse Hessian of the negative log posterior at the MAP.
  In a simulation study (three modalities, two factors, 244 parameters) the
  Laplace standard deviation of each lag was 1.01 to 1.04 times that of a Gibbs
  posterior, and 90% intervals contained the true lag 87% and 96% of the time on
  data generated from the model. On deliberately misspecified data they contained
  it 79% and 54% of the time: the errors assume the model is right and are too
  narrow when it is not. Response widths are excluded, because their marginal
  posterior sits far from the MAP. A lag at a bound gets `NaN`, and the report
  states whether the Hessian is positive definite once the factor-rotation
  directions, which are known in closed form, are removed. Off by default; available for MAP fits with
  dense, grouped or state-space algebra and at most 2000 parameters.
- **`MultimodalSRM(response_penalty_scaling="likelihood")`.** `Response.lag_prior`,
  `Response.prior` and `pooling_strength` were added unscaled to a normalized
  reconstruction loss, which made them far stronger than their stated values: in a
  synthetic fit a `Normal(0, 1)` lag prior shrank a one-second relative delay to
  0.33 s, and partial pooling at a strength matching the true 0.3 s spread between
  participants removed more than 80% of that spread. The new option multiplies each
  penalty by the factor a Gaussian likelihood over the scored observations implies
  (unit residual variance in standardized units, observations counted as
  independent). The same prior then left the delay at 1.00 s and tightened smoothly
  as its standard deviation fell, and partial pooling kept more than 80% of the
  participant spread. Calibration uses the same scaling. The default `"objective"`
  keeps the original behaviour.
- **`edge_observations` on both estimators.** An observation used to be scored
  only if its response support envelope (every lag the bounds allow plus the
  response's full tail) lay inside the run: a fixed `DoubleGamma` scored 18% of a
  60 s run and a fixed `BatemanSCR` none of it. `"include"` scores every
  observation in the run, like a "same" convolution; `"exclude"` keeps the earlier
  "valid" behaviour. R-MSRM extends each latent grid by the envelope's reach in
  whole grid cells and drops the extension from reported latents; GP-MSRM drops
  the exclusion, which a stationary latent process does not need (dense, grouped
  and state-space algebra; not spectral). On an empirical recording split into a 320 s training run and a
  184 s test run, the brain rows scored in training rose from 662 to 784 of 784
  with a fixed `DoubleGamma`, and predictable held-out brain rows from 329 to 454.
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
  relative to a reference modality, pairwise differences, the clock origin that
  was subtracted, and lags resting at a bound. The reference only sets which
  modality is shown at zero; by default it is the first modality whose lag is not
  estimated, or the first modality when every lag is estimated.
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

- **GP-MSRM lag summaries share one reference.** With every lag estimated,
  `relative_lag_draws()` and the fit summary's peak delays now show the first
  modality at zero, as `timing_report()` does, instead of centring on the
  across-modality mean. With several fixed lags the reference is the first in the
  order the responses were given rather than the first alphabetically. Pairwise
  differences are unchanged, and fits saved under the earlier convention keep it.
- **`Response(pooling=...)` now defaults to `"shared"`.** The earlier default,
  `"partial"` with `pooling_strength=1`, already behaved almost like complete
  pooling, because the penalty is added unscaled to a normalized loss: in a
  synthetic three-participant fit the participants' lags differed by 0.002 s and
  the fitted relative delay was the same to 0.001 s under either setting. The
  default now says what it does, and matches GP-MSRM, which supports shared
  responses only. To estimate participant responses, pass `pooling="partial"` or
  `"none"`, preferably with `MultimodalSRM(response_penalty_scaling="likelihood")`
  so that `pooling_strength` is `1 / sd²` of the spread between participants.
  `Response.lag_only` changes with it.
- **Edge observations are scored by default.** `MultimodalSRM` now defaults to
  `edge_observations="include"` and `BayesianMultimodalSRM` to `"auto"`, which
  includes for dense, grouped and state-space algebra and excludes for spectral
  algebra. Fits with a non-`Identity` response therefore score more rows than
  before and give different numbers; pass `edge_observations="exclude"` to
  reproduce earlier fits. Saved R-MSRM pickles and GP-MSRM archives keep the
  setting they were fitted with. In held-out synthetic runs with a 25 s causal
  response, error on the rows both settings predict was 0.054 against 0.055 for
  120 s and 300 s runs, and in 60 s runs exclusion left too few rows for a stable
  fit in one of six recordings; the rows only `"include"` predicts had an error
  near 0.18, where predicting the mean scores 1. The GP showed the same pattern
  (0.114 against 0.143 on shared rows over five recordings). On the empirical
  recording, where cross-modal prediction is near chance under either setting,
  the two did not differ beyond noise.
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

### Fixed

- **Bus error with `response_quadrature_order=32` and a `Gaussian` response.**
  `BayesianMultimodalSRM` evaluated response covariances in fixed batches of 32
  time differences. With 256 quadrature nodes (order 32 on the eight Gaussian
  panels) each batch intermediate was exactly 64 KiB, the largest size that
  jaxlib 0.11's CPU fusion runtime places on the stack of an XLA worker thread,
  and several of them nested overflowed that 544 KiB stack. The process died with
  "Fatal Python error: Bus error", most visibly after a MAP search when the final
  log-likelihood was evaluated. The batch is now sized so these intermediates
  always exceed the stack-allocation limit. Each time difference is still
  evaluated by the same arithmetic.
- **GP response-quadrature gradients at quadrature endpoints.** With
  `response_quadrature_order` set, the weighted Matérn sum bounded its exponential
  coordinates with `clip`, whose derivative is one half where the query equals the
  first or last quadrature node. The node at that boundary is counted on one side
  only, so the covariance gradient gained a slope that is not there. Values were
  always right. A response node lands exactly on an endpoint when a learned
  Gaussian response has its default width of 1 and timestamps are 2, 8 or 10 s
  apart, or when two modalities start from identical response parameters on the
  same clock. That is typically the starting point of a fit rather than a later
  iterate. In a synthetic two-participant fit on a 2 s clock the objective gradient
  for a Gaussian width was −7.4654 where a central difference gives −7.4623, and
  for a lag −5.11320 against −5.11332; both now agree with the difference to 2e-8.
  In the primitive, with nodes `[-5, -0.2, 3]`, signed weights and a 3 s length
  scale, the slope at the last node was −0.381 against a true 0.023. Second
  derivatives, which MAP polishing uses, were affected at the same points.

### Performance

- GP response quadrature computes each pair of modalities once when a covariance
  is requested between a set of timestamps and itself, filling the transposed
  block, and evaluates one value per distinct absolute time difference within a
  modality. Timestamps are compared exactly, and queries against a different clock
  or against traced times take the full calculation as before. On synthetic
  two-participant problems with two learned Gaussian responses on CPU, one
  objective-and-gradient evaluation ran 1.3 to 1.8 times faster across dense and
  grouped algebra, orders 64 and 384, and 24 or 48 timestamps per modality, timed
  on a busy machine; objectives agreed to 3e-12 and gradients to 2e-11 away from the
  endpoint cases above.
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
