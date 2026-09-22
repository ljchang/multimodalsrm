# Migration from personalized-srm

Use a separate environment for the new package and retain the frozen research environment for historical replay. This package is `multimodalsrm` version `0.1.0`, not a new release of the `personalized_srm` namespace.

## Imports

| Previous import | New import |
| --- | --- |
| `from personalized_srm import MultimodalSRM` | `from multimodalsrm import MultimodalSRM` |
| `from personalized_srm.multimodal import TimeSeries, Response, Gaussian` | `from multimodalsrm import TimeSeries, Response, Gaussian` |
| `from personalized_srm.multimodal.bayesian import BayesianMultimodalSRM` | `from multimodalsrm.bayesian import BayesianMultimodalSRM` |
| `from personalized_srm.multimodal.bayesian.workflow import save_model, load_model` | `from multimodalsrm.bayesian.workflow import save_model, load_model` |

Update imports and install the Bayesian extra when needed. Data nesting, response coordinates, estimator parameters and target-exclusion semantics are retained by the extraction. Legacy estimator catalogs and research experiment runners are not migrated. No alias package for the old namespace is provided.

## Fitted GP model archives

Use `save_model` and `load_model` from `multimodalsrm.bayesian.workflow`. The JSON/numeric archive format and historical serialized identifiers intentionally retain their `personalized-srm` names. Do not rewrite those strings to match the Python namespace.

Schemas 1 and 2 represent MAP group and MAP participant states; schema 3 represents training posteriors; schema 4 represents updated posterior states. The loader reconstructs a fitted model for prediction/reporting without a new parameter optimization or MCMC run. Saved diagnostics retain their original success, failure or undefined status. The extraction check replayed source-generated synthetic archives for all four schemas under the new namespace, including calibrated predictions and seeded posterior paths. See the [validation record](extraction-validation.json). Successful replay does not requalify the old scientific result under a new runtime.

Keep original archives unchanged. Archives can contain native observations and posterior draws, so store them with the research data rather than committing them to the package. Saving requires a new destination. Preserve any training standardizer with `save_model(..., standardizer=scaler)` and use the restored standardizer consistently. The [MAP quickstart](https://github.com/ljchang/multimodalsrm/blob/main/examples/gp_map_quickstart.py) performs a synthetic save/load comparison.

## R pickle/joblib models

Python pickle/joblib stores class module paths, including paths under `personalized_srm.multimodal`. Installing `multimodalsrm` alone does not resolve those paths. There is no automatic converter or compatibility unpickler in this package.

For an existing trusted R model, retain the original source checkout, dependency versions and preprocessing artifacts. Load it and make predictions in that original environment. Export only the numeric outputs and metadata needed for downstream analysis. For new-package use, refit from the authorized original training data with the same configuration, then compare objective, kernels, validity masks and predictions against the old environment under explicit tolerances. A refit is a new fitted artifact; preserve its provenance and do not call it a converted old model. If original data are unavailable, continue prediction/replay in the old environment.

## Warmup checkpoints

Warmup checkpoints are different from fitted GP archives. They restore the complete sampler state at the end of warmup and restart retained sampling from that boundary. They do not append partially retained draws. Repeated/resumed trajectories must not be combined as independent samples.

Compatibility checks hash every Python source file under the model package tree, as well as recording numerical runtime, devices, PRNG configuration and target identity. The namespace extraction changes that source tree even when mathematical behavior is preserved. **An existing research-repository warmup checkpoint must resume in its original frozen source and runtime environment.** Do not edit checkpoint identities, remove hashes or assume fitted-archive compatibility implies warmup compatibility.

New package checkpoints can be created with `fit(..., warmup_checkpoint=path)` and resumed with the identical configuration, data, explicit integer seed, runtime and source using `resume_warmup=True`. This API covers full dense/grouped posterior training with independent noise and no run baselines; it does not cover posterior updates, conditional parameter targets or state-space inference. Checkpoints require new destinations and never overwrite existing ones. Retain the source/environment for each long run.
