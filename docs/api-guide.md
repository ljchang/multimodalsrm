# API: choose a model and its settings

Start here when you know what you want to estimate but are unsure which arguments to use. The linked reference pages list every public export, constructor default, and public method defined on those classes. These pages follow the checked-out source; the installed release may have fewer options.

!!! info "Current settings and queued changes"
    This reference follows the checked-out source. The [upcoming changes guide](upcoming-changes.md) covers the queued BachSCR removal and `SearchConfig.r_init` option. Bateman is used for new SCR examples; Bach is documented as a legacy API while it remains exported.

## The choices at a glance

| Question | Setting | Meaning |
| --- | --- | --- |
| How is the common response represented? | `MultimodalSRM` or `BayesianMultimodalSRM` | A regularized latent grid, or a continuous latent GP |
| How many common factors? | `features=K` | Dimension of the shared response, not an inferred count of biological processes |
| What makes the latent response smooth? | R: `temporal_strength`; GP: `length_scale` | A derivative penalty, or the Matérn-3/2 prior timescale |
| How does a modality respond to that signal? | `responses={name: Response(...)}` | Identity, Gaussian, Gamma, DoubleGamma, BatemanSCR, or a fixed sampled curve where supported |
| Which response parameters are learned? | `estimate`, `fixed`, `bounds` | Learn unfixed parameters within their bounds, or fix the entire response |
| Who shares a response filter? | `Response.pooling` | Shared across people, partially pooled, or separate; GP requires `"shared"` |
| Who shares the latent trajectory? | R: `latent_pooling` | Default `"shared"` is exact equality; extensions change the model |
| Do I want GP MAP or posterior sampling? | `inference="map"` / `"posterior"` | Point parameter estimate or parameter draws; **default is `"posterior"`** |
| Which GP computation? | `linear_algebra` | Dense, grouped, state-space, or spectral; separate from inference |
| How do I evaluate an analysis choice? | Protected runs and target exclusion | Compare held-out prediction on common valid support; inspect recovery and convergence separately |

[Estimator signatures](api/estimators.md) · [Responses and priors](api/responses.md) · [Search and sampling](api/configuration.md) · [Data and results](api/data-results.md) · [Evaluation](api/evaluation.md)

## Data and units

```python
from multimodalsrm import TimeSeries

# Each values array has shape (time, feature); times are strictly increasing.
data = {"participant-01": {"run-01": {
    "fmri": TimeSeries(values=fmri_values, times=fmri_times, mask=fmri_mask),
    "rating": TimeSeries(values=rating_values, times=rating_times),
}}}
```

The names and arrays above are illustrative. Use one consistent time unit, normally seconds. Modalities keep their native sampling times. A mask is Boolean, has the same shape as `values`, and marks observed entries with `True`. Omit fully missing streams. Feature counts may differ between participant–modality pairs but must stay fixed across runs for a given pair. See [data preparation](data.md).

## R-MSRM settings

Import `MultimodalSRM` from `multimodalsrm`. `latent_dt` is required. The estimator fits preprocessing on the training data and reuses it for predictions.

| Arguments (defaults) | Choice and interpretation |
| --- | --- |
| `features=10`, `latent_dt` required | Positive factor count and positive grid spacing in timestamp units. A finer grid increases resolution and cost; it is not a response width. |
| `responses=None` | Defaults to fixed Identity for every observed modality. Otherwise supply a `Response` for every observed modality. |
| `latent_pooling="shared"` | One identical latent response per run for all participants. `"population"` softly couples participant trajectories to their population mean; `"neighborhood"` softly couples them using a supplied affinity; `"components"` imposes exact sharing within connected components. |
| `latent_strength=1.0` | Coupling penalty for soft pooling. Does not relax exact equality under `"shared"`. Pass a symmetric nonnegative named graph to `fit(..., affinity=...)` for neighborhood/components. |
| `modality_weights=None` | Equal weights across configured modalities. Explicit nonnegative weights must cover those modalities and sum to one. These control the training objective, not evaluation weights. |
| `loading_ridge=1e-3`, `loading_penalty_scaling="pair"` | Loading shrinkage, averaged by pair. `"feature"` additionally divides each pair's loading penalty by its fitted feature count. |
| `latent_ridge=1e-4`, `temporal_strength=1e-2` | Latent magnitude and temporal derivative penalties. The ridge must be positive; temporal strength may be zero. |
| `balance_scale=True` | Balances loading/latent scale under the regularized objective; does not identify physiological amplitude. |
| `init="random"` | Also `"spectral"` or `"hybrid"`. Hybrid uses a spectral first start and random remaining starts. Repeated spectral starts are redundant. |
| `n_init=1`, `n_jobs=1`, `random_state=None` | Restart count, process workers (`-1` uses available CPUs, capped by restarts), and seed. An integer seed makes starts reproducible across worker counts. |
| `max_iter=100`, `kernel_max_iter=50`, `tol=1e-5` | Outer optimization budget, response optimization budget, and stopping tolerance. Inspect `converged_` and `restart_diagnostics_`. |
| `gap_threshold=None` | Default splits integration support above twice the median usable interval. A positive explicit threshold uses timestamp units. |

## GP-MSRM settings

Import `BayesianMultimodalSRM`, `BayesianPriors`, and `Prior` from `multimodalsrm.bayesian`. The GP consumes supplied units without internal standardization. Set `JAX_ENABLE_X64=true` before starting Python/JAX for fitting.

| Arguments (defaults) | Choice and interpretation |
| --- | --- |
| `priors` required | `BayesianPriors(noise=..., loading_sd=..., offset_sd=..., filters=...)`. Noise is **variance**. Filter priors map modality names to free parameter names. |
| `features=1` | Number of factors; multifactor posterior fits require explicit training `factor_anchors`. |
| `responses=None` | Fixed Identity by default. Explicit responses must use `pooling="shared"`; GP priors go in `BayesianPriors`, not `Response.prior` or `lag_prior`. |
| `length_scale=3.0` | Positive fixed Matérn-3/2 timescale, or a `Prior` with finite positive bounds to learn one common timescale in supported dense/grouped workflows. GP variance stays one. No alternative latent covariance family selector exists. |
| `inference="posterior"` | Choose `"map"` explicitly for MAP only. Sampling and MAP optimize/use the same physical parameter density, with the latent GP marginalized. |
| `linear_algebra="dense"` | `"grouped"` reduces repeated modality/time functionals without changing the modeled likelihood. `"state_space"` is opt-in MAP with response restrictions. `"spectral"` is an explicit finite-basis approximation. See the [support matrix](capabilities.md#estimation-and-numerical-backends). |
| `anchor=None`, `factor_anchors=None` | Training feature tuples `(participant, modality, feature_index)` define reporting orientation. Multifactor anchors contain one tuple per factor. They do not establish physiological timing. |
| `reference_modality=None` | Optional modality with fixed lag to define a timing convention. Without it, an existing fixed lag is used if available; all-free lags need proper priors. |
| `search=None`, `sampler=None` | Use default `SearchConfig()` and `SamplerConfig()`; see below. |
| `sample_blocks=None` | Full target by default. A nonempty unique subset of `"filter"`, `"loading"`, `"offset"`, `"noise"`, `"gp"` samples conditional on remaining parameters; this is not the full posterior. |
| `random_state=0` | Reproducibility seed, or `None`. |
| `max_observations=800` | Per-run eligible scalar-observation guard, retained by grouped and state-space paths. Raising it permits a larger problem but does not guarantee feasible runtime. |
| `covariance_tolerance=1e-7` | Error tolerance for supported analytic/response approximation checks. Does **not** certify response quadrature or spectral accuracy. |
| `response_quadrature_order=None` | Explicit integer order 8–1024 per panel for dense/grouped structured response integrals. Needed for Gamma, DoubleGamma and BatemanSCR on those paths. Compare orders for numerical qualification. |
| `state_space_gaussian="auto"` | Also `"rational"` or `"laguerre"`; selects a Gaussian response approximation for state-space MAP. |
| `spectral=None` | `SpectralConfig(rank=..., padding=...)` is required only for `linear_algebra="spectral"`. Padding is per side in timestamp units; rank/padding need joint accuracy checks. |
| `run_baseline_sd=None` | Optional fixed per-modality Gaussian run-offset prior SDs. These are marginalized observation offsets, not private latent trajectories. Outside the full posterior workflow. |
| `noise_timescales=None` | Optional fixed per-modality OU residual timescales for dense/grouped MAP, without run baselines. Distinct from the shared latent timescale and response width. |

Fixed-response state-space filtering and smoothing automatically group exactly matching modality/time observations when noise support is strictly positive. This is separate from selecting `linear_algebra="grouped"`; no new switch is needed. See [grouped state-space computation](grouped-state-space.md) and the [queued learned-response extension](upcoming-changes.md).

### Search and sampling

[Exact defaults and signatures](api/configuration.md) are generated from source.

| Object | Main choices | Interpretation |
| --- | --- | --- |
| `SearchConfig` | `starts=16`, `maxiter=1200`, `n_jobs=1` | Independent MAP starts; positive thread count. |
| `SearchConfig` | `ftol=1e-12`, `gtol=1e-6`, `physical_gradient_tolerance=1e-3` | Objective/optimizer stopping and physical-gradient diagnostic thresholds. |
| `SearchConfig` | `refine_maxiter=0`, `polish_max_parameters=256` | Optional serial refinement budget and dimension limit for polishing. |
| `SearchConfig` | `conditioning="none"` or `"diagonal"` | Optional coordinate scaling for supported full multifactor dense/grouped MAP targets. |
| `SamplerConfig` | `chains=4`, `warmup=1000`, `draws=1200` | Chain count, warmup transitions, retained draws per chain. These are budgets, not convergence guarantees. |
| `SamplerConfig` | `target_accept=0.99`, `max_tree_depth=10` | NUTS adaptation target and trajectory limit. |
| `SamplerConfig` | `chain_method="sequential"` | Also `"parallel"` or `"vectorized"`; execution choice subject to available devices/runtime. |
| `SamplerConfig` | `mass_matrix="dense"`, `max_dense_parameters=1024` | Dense or diagonal metric and dense-metric capacity guard. |
| `SamplerConfig` | `start_objective_window=5.0`, `start_jitter=0.08` | Selection/jitter of sampling starting points. |
| `SamplerConfig` | `orientation_refresh="none"` or `"haar"` | Optional common rotation/reflection refresh for supported full multifactor targets with isotropic unbounded Gaussian loading priors. |

The queued R initialization change adds `SearchConfig(r_init=True)` and uses `r_init=False` for historical starts. This field is **pending** in this documentation baseline; see [upcoming changes](upcoming-changes.md#r-initialization-changes-a-search-default).

### A GP MAP configuration with a learned Gaussian lag

This creates a configuration; it does not fit data. Prior scales below are illustrative and must be chosen in the supplied measurement units.

```python
from multimodalsrm import Gaussian, Identity, Response
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior

responses = {
    "rating": Response(Identity(), estimate=False, pooling="shared"),
    "fmri": Response(
        Gaussian(width=1.0, lag=2.0),
        pooling="shared", estimate=True, fixed={"width": 1.0},
        bounds={"lag": (0.0, 5.0)},
    ),
}
model = BayesianMultimodalSRM(
    features=1, inference="map", linear_algebra="grouped",
    responses=responses, length_scale=3.0,
    priors=BayesianPriors(
        noise=Prior.lognormal(-2.0, 0.7),
        loading_sd=1.0, offset_sd=1.0,
        filters={"fmri": {"lag": Prior.normal(2.0, 1.0)}},
    ),
)
```

The `Response` bounds truncate and renormalize the corresponding GP filter prior. See [tutorials](tutorials.md) for runnable fits and [temporal kernels](temporal-kernels.md) for fixed, lag-only and shape-learning examples.

## Methods: which operation do I need?

| Task | R-MSRM | GP-MSRM |
| --- | --- | --- |
| Train parameters | `fit(data)` | `fit(data)` with explicit inference choice |
| Infer shared latent signal | `infer_latent(data, times=...)` | `infer_latent(times=...)` on fitted/conditioned model; returns marginal moments/mixtures |
| Predict excluded targets | `predict(data, targets=..., source="within"/"across"/"both")` | `condition(donors, targets=..., mode="frozen")` for MAP, then `predict(times=...)` |
| Jointly update a posterior with new runs | Not a posterior method | `condition(..., mode="joint")`; **joint is the default**, and repeated calls replace the donor batch |
| Calibrate a new participant | `calibrate(data, reference=...)` | MAP `calibrate(...)` or supported joint `calibrate_posterior(...)` |
| Decode each participant independently | Select R donor source for prediction | `transform(data, times=...)` for MAP; `condition_participants(...)` for supported posterior updates |
| Draw a joint latent trajectory | Not available | `sample_latent(...)`; request all correlated query times in one call |
| Inspect response curves | `kernel(...)`, `plot_kernels(...)` | Consult fitted parameters and posterior summaries; R inspection methods are not shared GP methods |

[Full method signatures](api/estimators.md) include progress, query and calibration options. Consult the [capability guide](capabilities.md) before combining advanced options. A finite objective, narrow interval or successful call does not establish convergence or recovery.
