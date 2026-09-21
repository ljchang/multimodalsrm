# Installation

Use Python 3.12 or later. Python 3.12 is the version currently exercised by package CI. Until the first PyPI release, install from the public repository.

## R-MSRM

```sh
git clone https://github.com/ljchang/multimodalsrm.git
cd multimodalsrm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/r_quickstart.py
```

The activation command above is for macOS/Linux shells. On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The base package provides the regularized model and shared data/response types:

```python
from multimodalsrm import MultimodalSRM, TimeSeries, Identity, Gaussian, Response
```

## GP-MSRM

Install the optional Bayesian dependencies into the same environment:

```sh
python -m pip install -e '.[bayesian]'
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu python examples/gp_map_quickstart.py
```

In PowerShell, set `$env:JAX_ENABLE_X64="true"` and `$env:JAX_PLATFORM_NAME="cpu"` before running the script. Float64 must be enabled before fitting. The Bayesian extra installs the numerical runtime used by the package; the base R installation does not require JAX or NumPyro.

```python
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors
```

The GP quickstart performs MAP fitting and archive replay. It does not run posterior sampling. Read [capabilities](capabilities.md) before selecting a sampling or computational backend.

## Record the version you use

An editable installation follows changes in your checkout. For a research analysis, record the source commit (`git rev-parse HEAD`), model configuration, random seed, and dependency versions. Keep the original environment for replaying historical fitted artifacts.

Continue with [data preparation](data.md) and the [runnable tutorials](tutorials.md).
