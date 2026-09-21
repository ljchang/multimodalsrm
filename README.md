# MultimodalSRM

[![CI](https://github.com/ljchang/multimodalsrm/actions/workflows/ci.yml/badge.svg)](https://github.com/ljchang/multimodalsrm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Shared response models for observations measured across people, modalities and native sampling times.

**Status: distribution infrastructure under construction.** This repository currently contains packaging, CI and release infrastructure. The R-MSRM and GP-MSRM implementations have not yet been imported. Version `0.1.0.dev0` is a local development scaffold, not a usable model release. A passing CI badge currently describes infrastructure checks; model suites are explicitly pending and package publication is blocked.

## Planned model families

- **R-MSRM (`MultimodalSRM`)**: regularized estimation of one shared latent response per run, with participant–modality mappings and modality response kernels.
- **GP-MSRM (`BayesianMultimodalSRM`)**: a continuous shared latent Gaussian process with explicit priors, MAP estimation and supported posterior workflows.

Both families will retain native observation times, masks, missing streams and explicit information boundaries for held-out prediction. Bayesian capabilities are being completed in the source research repository before a verified handoff. Learned FIR responses are deferred.

The initial distribution will focus on these two models. Legacy neighborhood, graph, mixture and reference models remain in their research repository. Implementation tests, convergence, calibrated uncertainty, response recovery and empirical usefulness are separate claims.

## Contributing to the scaffold

Python 3.12 is the initial CI target. From a clone of this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/infrastructure
ruff check .
ruff format --check .
python -m build
python -m twine check --strict dist/*
```

Current usable import: `from multimodalsrm import __version__`. Model classes are not exported yet. Once integrated and released, the planned installation is `pip install multimodalsrm` for R-MSRM and `pip install 'multimodalsrm[bayesian]'` for GP-MSRM. The provisional Bayesian dependency versions will be reconciled with the final verified handoff.

## Development and releases

- [Model integration and extraction plan](docs/model-integration.md)
- [CI and test policy](docs/testing.md)
- [Release and Trusted Publisher setup](docs/releasing.md)
- [Tutorial scope](docs/tutorials.md)
- [Contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md)

CI builds a wheel and source distribution and tests both installed artifacts on Linux and macOS. After model integration, core and Bayesian suites run separately on both platforms. PyPI publication uses a versioned GitHub Release; manual release workflow runs target TestPyPI. Neither route can publish the scaffold.

## License and citation

MIT licensed; see [LICENSE](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff). There is no package DOI or associated publication claimed by this scaffold. Research data and fitted model archives are not distributed.
