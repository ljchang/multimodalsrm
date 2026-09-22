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

## Thread settings

Set BLAS and OpenMP thread counts before starting Python; the package never changes process-global thread limits during a fit.

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` reproduces CI and is the safe choice on a laptop or a shared machine.
- For GP-MSRM fits on a many-core workstation, 8 to 16 threads was the measured sweet spot. The CPU factorization runs LAPACK inside a JAX host callback while JAX's own thread pool already occupies the cores, so the BLAS default of one thread per core oversubscribes the machine. On a 64-core workstation that default slowed a 2,727-dimension factorization by up to 10x relative to 16 threads; see the [Linux GPU opportunities note](performance/2026-09-21-linux-cuda-gpu-opportunities.md). The package warns once per process when the BLAS pool exceeds half the physical cores.
- R-MSRM restarts with `n_jobs > 1` run in separate processes that limit themselves to one BLAS thread each, so the parent process setting matters only for serial fits.

Use the same settings when comparing timings; thread count and background load change CPU results by several-fold.

## Record the version you use

An editable installation follows changes in your checkout. For a research analysis, record the source commit (`git rev-parse HEAD`), model configuration, random seed, and dependency versions. Keep the original environment for replaying historical fitted artifacts.

Continue with [data preparation](data.md) and the [runnable tutorials](tutorials.md).
