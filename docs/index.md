# MultimodalSRM

## Shared responses across people and modalities

MultimodalSRM connects observations recorded at different sampling times through a shared latent response. Each participant and modality has its own mapping from that response to measured features. Response filters describe how a modality smooths or delays the shared signal.

Use it to fit a common representation, infer latent responses on new recordings, or predict a held-out modality from the observations you make available.

[Get started](getting-started.md){ .md-button .md-button--primary }
[How it works](concepts.md){ .md-button }

!!! info "Version 0.1.0"
    Install the package from PyPI. These pages follow `main`; use the `v0.1.0` source tag when matching an analysis to this release.

## Choose a model

| | R-MSRM | GP-MSRM |
| --- | --- | --- |
| Python class | `MultimodalSRM` | `BayesianMultimodalSRM` |
| Shared response | Regularized latent values on a grid | Continuous latent Gaussian process |
| Estimation | Regularized optimization | MAP or supported posterior sampling |
| Uncertainty | No posterior uncertainty | Conditional prediction or parameter-posterior uncertainty, depending on the workflow |
| Start here | [R tutorial](tutorials.md#r-msrm-fit-and-held-out-prediction) | [GP MAP tutorial](tutorials.md#gp-msrm-map-and-fitted-archive-replay) |

Both model families accept native timestamps, masks, and missing streams. The default is one exact common latent response per run, with individual participant–modality mappings.

## Learn the workflow

1. [Install the package](getting-started.md) in a dedicated Python environment.
2. [Prepare your data](data.md) with explicit participant, run, and modality names.
3. [Run the synthetic tutorials](tutorials.md) for fitting and held-out prediction.
4. Read the [capability and diagnostic guide](capabilities.md) before using advanced Bayesian workflows.
5. Consult [migration and persistence](migration.md) when loading historical fitted models.

These first guides cover the data contract, model concepts, and runnable R/GP MAP examples. Detailed posterior tutorials and a complete API reference are future additions. Successful software checks do not establish fit convergence, uncertainty calibration, response recovery, or empirical usefulness.

The project is [MIT licensed](https://github.com/ljchang/multimodalsrm/blob/main/LICENSE). Code, issues, and contributions live on [GitHub](https://github.com/ljchang/multimodalsrm).
