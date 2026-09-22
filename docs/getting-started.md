# Installation

Use Python 3.12 or later. Python 3.12 is the version currently exercised by package CI. Install version `0.1.0` from PyPI, or use a source checkout for development and the example scripts.

## Install the release

Create and activate a dedicated environment, then install the base model:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "multimodalsrm==0.1.0"
```

For GP-MSRM, add the Bayesian extra:

```sh
python -m pip install "multimodalsrm[bayesian]==0.1.0"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The following examples require a source checkout because the scripts are not installed as commands.

## Source checkout and R-MSRM example

```sh
git clone https://github.com/ljchang/multimodalsrm.git
cd multimodalsrm
git checkout v0.1.0
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

## GP-MSRM example

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
