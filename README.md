# MultimodalSRM

[![CI](https://github.com/ljchang/multimodalsrm/actions/workflows/ci.yml/badge.svg)](https://github.com/ljchang/multimodalsrm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Shared response models for observations measured across people, modalities and native sampling times.

**Development version: `0.1.0.dev0`.** This repository contains the extracted R-MSRM and GP-MSRM implementations. It has not been released on PyPI. Install from a checkout for development; a configured Trusted Publisher does not mean a release has occurred. Software checks, fit convergence, uncertainty calibration and empirical recovery are separate claims.

## Models

- **R-MSRM (`MultimodalSRM`)** estimates one exact common latent response per run by default, with participant–modality mappings and regularized response kernels.
- **GP-MSRM (`BayesianMultimodalSRM`)** uses a continuous shared latent Gaussian process, individual observation mappings, explicit priors, MAP estimation and supported posterior workflows.

Both use named native-time observations, masks and missing streams. They preserve explicit information boundaries for held-out prediction. Learned FIR responses are deferred. Legacy neighborhood, graph, mixture and reference estimators are not exported by this package. See the [capability and backend matrix](docs/capabilities.md) before choosing a workflow.

## Install from source

Python 3.12 or later is required. From a clone:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/r_quickstart.py
```

For the optional Bayesian runtime:

```sh
python -m pip install -e '.[bayesian]'
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu python examples/gp_map_quickstart.py
```

The base install does not require JAX, NumPyro, ArviZ or plotting libraries. The Bayesian extra pins the runtime used by the source handoff; enabling float64 is required. These source-install commands do not depend on a PyPI release.

## Data and imports

```python
from multimodalsrm import MultimodalSRM, TimeSeries, Identity, Gaussian, Response
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors

# values: (observations, features); times: (observations,).
# Each modality retains its own timestamps. A mask marks observed entries.
# data = {participant: {run: {modality: TimeSeries(values, times, mask)}}}
```

Use consistent timestamp units across streams. Separate runs have separate latent responses; participants viewing the same run share one response. `features=K` chooses the latent dimensionality rather than estimating K. The [runnable tutorials](docs/tutorials.md) demonstrate fitting, explicit target exclusion, independent new runs and GP archive replay on small synthetic data.

## Documentation and development

- [Tutorials](docs/tutorials.md) and [capabilities](docs/capabilities.md)
- [Migration and archive compatibility](docs/migration.md)
- [Source integration and provenance](docs/model-integration.md)
- [CI and test policy](docs/testing.md)
- [Release setup](docs/releasing.md), [contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md)

The development workflow tests installed wheels and source distributions, with separate core and Bayesian suites. See the current CI run and integration record for test evidence. Passing CI does not certify scientific validity or authorize a new release.

## License and citation

MIT licensed; see [LICENSE](LICENSE). The implementation was extracted from Luke J. Chang's [shared-response-models](https://github.com/ljchang/shared-response-models) research repository with contributor provenance retained in the [integration record](docs/model-integration.md). Citation metadata is in [CITATION.cff](CITATION.cff). No package DOI or associated publication is claimed. Research observations and fitted archives are not distributed.
