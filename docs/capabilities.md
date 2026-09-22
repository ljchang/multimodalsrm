# Capabilities and evidence boundaries

Both model families use named observations of the form `{participant: {run: {modality: TimeSeries(...)}}}` on native clocks. `TimeSeries.values` is observations by features. Missing observations use masks; missing streams may be omitted. The default model has one exact shared latent response per run and individual participant–modality mappings. Separate private trajectories, learned graphs and participant-specific response pooling are not part of the supported GP posterior target.

## Estimation and numerical backends

MAP is an estimation method. `linear_algebra` selects a GP computational backend; the two choices are separate.

| Model / operation | Supported scope | Boundary |
| --- | --- | --- |
| R-MSRM regularized fit | Shared latent grid (`latent_dt`), individual mappings, fixed or estimated parametric responses; fit, infer, target-excluded prediction and calibration | Regularization and response constraints influence identifiability; no posterior uncertainty claim |
| GP MAP, `linear_algebra="dense"` or `"grouped"` | Continuous Matérn GP; analytic Identity/Gaussian; explicit quadrature for Gamma/DoubleGamma/BatemanSCR | Grouped algebra preserves the modeled likelihood; it is not a different prior |
| GP MAP, `linear_algebra="state_space"` | Explicit opt-in Identity, Gaussian, fixed integer-shape Gamma/DoubleGamma, and learned BatemanSCR; independent noise | Response approximations and restored tails need the declared covariance error tolerance; no state-space posterior sampling |
| GP `linear_algebra="spectral"` | Explicit finite-basis approximation retained from the source implementation | Rank/padding and boundary errors need separate qualification; not the full supported posterior workflow below |
| Full GP posterior training and joint updates | Dense/grouped; independent observation noise; no run baselines; fixed unit GP variance; fixed or learned single shared GP timescale | No sampled-kernel posterior, participant response pooling, factor-specific timescales or state-space sampling |
| Joint posterior trajectories | `sample_latent` on supported training, donor-update, calibrated and independent-participant posterior models | Request all times for a joint calculation together; separate calls do not preserve cross-call correlation |

Full posterior responses are analytic Identity/Gaussian or Gamma/DoubleGamma/BatemanSCR enabled with an explicit `response_quadrature_order`. Quadrature approximates response integrals, while the latent process remains continuous. Compare integration orders and independent integrals over the declared parameter bounds; the analytic `covariance_tolerance` does not certify quadrature accuracy. Learned FIR responses remain deferred; `SampledKernel` does not imply a sampled-response posterior implementation.

State-space MAP uses native-time Matérn-3/2 Kalman inference and an RTS smoother. Gaussian responses use rational or Laguerre approximation; Gamma shapes remain fixed while supported scales and lags may be estimated. BatemanSCR learns rise/decay time constants and lag in two response states, with a bound for the restored 90-second tail. These backend choices do not change MAP into posterior inference. See the [SCR response guide](scr-responses.md) for configuration and migration.

With fixed or learned responses and a strictly positive noise prior, `state_space`
automatically groups identical modality/time measurements for both filtering
and smoothing. This retains each measurement's noise, residual and likelihood
normalization, all participant contributions, feature masks and separate run
priors. Timestamps are matched exactly; close times are never binned. Learned
lags reorder nodes without merging them, and learned widths/scales update the
response realization. Noise priors whose support includes zero retain scalar updates.
The [grouped filtering guide](grouped-state-space.md) describes the numerical
method, configuration and performance checks.

A learned `length_scale` uses a bounded physical `Prior` with finite, strictly positive bounds. One timescale is shared across factors and runs; loadings set amplitude while GP variance stays one. Learning the timescale changes the inference target and requires new convergence/identification checks. MAP options outside the full posterior scope, such as fixed run-baseline priors, must not be assumed to carry over to joint posterior updates.

## Responses and workflow semantics

`Response(..., estimate=True, fixed={...})` estimates only unfixed parameters, within explicit bounds. For example, `Response(Gaussian(width=0.3), estimate=True, fixed={"width": 0.3}, bounds={"lag": (-1.5, 1.5)}, pooling="shared")` estimates a shared lag while preserving width. Response support can remove edge observations; inspect output validity masks rather than treating every requested time as observed or usable.

R `predict(data, targets=..., source=...)` excludes all named target payloads before preparation. Its `source` selects within-participant, across-participant or combined observations. R fitting estimates preprocessing from training data and reuses it for new-run predictions.

For GP MAP, `condition(donors, targets=..., mode="frozen")` conditions new runs while holding training MAP parameters fixed. Its uncertainty is conditional on that MAP, not parameter-posterior uncertainty. The small [GP example](https://github.com/ljchang/multimodalsrm/blob/main/examples/gp_map_quickstart.py) demonstrates this route. GP uses supplied observation units; if using `TrainingStandardizer`, fit it on training observations only, exclude held-out targets before transforming donors, and save it with the model.

For a full supported GP posterior:

- `condition(..., mode="joint")` refits and samples the reference plus the current donor batch, including the prior once. Repeated calls replace the donor batch; provide the complete batch to retain several new runs.
- `calibrate_posterior(...)` jointly samples group and new-participant parameters using original training run IDs and in-domain calibration times. Each feature needs at least `K+1` support-eligible observations; this is a feasibility condition.
- `condition_participants(...)` returns one updated posterior per participant, using only that participant's current independent-run observations in addition to the fitted reference.
- `infer_latent(...)` reports marginal moments and mixture components. `sample_latent(...)` draws joint conditional paths across times and factors, preserving parameter-draw uncertainty. Additional trajectories per parameter draw do not increase parameter effective sample size.

Multifactor posterior fits require explicit training `factor_anchors`, one observed training feature per factor. Draw-specific QR rotations define reporting coordinates; raw parameter draws and their diagnostics are retained. An anchor or fixed-lag reference does not establish absolute physiological timing.

Existing `.calibrate` and `.transform` APIs retain their MAP meanings. Do not interpret them as the joint posterior methods above. New run IDs must be distinct from reference training IDs where a new independent run is required.

## Interpreting outputs

Inspect `converged_` for R fits and `map_diagnostics_` for GP MAP fits. Posterior fits require `sampling_diagnostics_`, including raw parameters, reporting orientation, divergences, tree depth and BFMI. A successful API call, archive replay or test run does not establish convergence. Marginal intervals do not automatically form simultaneous trajectory bands.

Frozen source studies qualify bounded synthetic configurations only. Their results do not establish general coverage, arbitrary-rank convergence, broad response/timing recovery, empirical scalability or physiological mechanisms, and do not automatically transfer to new runtimes. Report prediction, response recovery, convergence, numerical accuracy and uncertainty calibration separately.
