# Installation

Use Python 3.12 or later. Python 3.12 is the version currently exercised by package CI. Install version `0.2.0` from PyPI, or use a source checkout for development and the example scripts.

## Install the release

Create and activate a dedicated environment, then install the base model:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "multimodalsrm==0.2.0"
```

For GP-MSRM, add the Bayesian extra:

```sh
python -m pip install "multimodalsrm[bayesian]==0.2.0"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The following examples require a source checkout because the scripts are not installed as commands.

## Source checkout and R-MSRM example

```sh
git clone https://github.com/ljchang/multimodalsrm.git
cd multimodalsrm
git checkout v0.2.0
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

In PowerShell, set `$env:JAX_ENABLE_X64="true"` and `$env:JAX_PLATFORM_NAME="cpu"` before running the script. Float64 must be enabled before fitting. The Bayesian extra installs the CPU numerical runtime used by the package; the base R installation does not require JAX or NumPyro. Pinning the platform to `cpu` is optional on a machine without a GPU, and on a machine with an NVIDIA card it records that the CPU backend is intended (see [NVIDIA GPUs on Linux](#nvidia-gpus-on-linux)).

```python
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors
```

The GP quickstart performs MAP fitting and archive replay. It does not run posterior sampling. Read [capabilities](capabilities.md) before selecting a sampling or computational backend.

GP MAP searches can optionally start from a bounded R-MSRM fit with
`SearchConfig(r_init=True)`, including the MAP search that precedes posterior
sampling. One CPU R-MSRM fit supplies loadings, offsets, residual noise variances
and learned response parameters for the first GP restart. The remaining
prior-based restarts are unchanged. This changes starting values only; GP priors,
native timestamps and convergence checks stay the same.

It is **off by default**: the measured benefit is scale-dependent and did not
transfer to a real recording, so it is a per-analysis decision rather than a
general recommendation. To enable it:

```python
from multimodalsrm.bayesian import SearchConfig

model = BayesianMultimodalSRM(
    priors=priors,
    inference="map",
    search=SearchConfig(r_init=True),
)
```

The preliminary R fit uses one hybrid start, at most 40 alternating iterations
and 20 iterations per kernel optimization. Its grid spacing is half the smaller
of the shortest median native sampling interval and initial GP timescale,
coarsened if necessary to approximately 10,000 cells across runs. This grid is
used only for initialization. Feature scales, masks, all training runs and the
single-factor positive anchor are accounted for when translating to GP units;
starting values are clipped to interior prior quantiles. Only Gaussian response
parameters are optimized in the preliminary R fit: other response families are
held at their configured initial values to avoid expensive finite differences.
The GP still learns all requested response parameters. The R fit need not
converge. Its status and total initialization time (including GP score validation) are recorded in
`restart_diagnostics_[0]["initialization"]`. If it fails numerically, the search
warns and retains the previous data-based start. Conditional participant
calibration and posterior updates retain their existing initialization.

See the [initialization timing comparison](performance/r-map-initialization.md)
for a reproducible benchmark and the limits of the measured benefit.

## NVIDIA GPUs on Linux

GP-MSRM runs on CUDA in float64 without code changes. Install the pinned JAX CUDA plugin with the `bayesian-cuda` extra:

```sh
python -m pip install -e '.[bayesian-cuda]'
JAX_ENABLE_X64=true python examples/gp_map_quickstart.py
```

The extra adds the CUDA wheels only on Linux x86_64, which is the only platform they exist for; elsewhere it resolves to the same CPU runtime as `bayesian`, so one install command works in a mixed lab. Both extras pin the same JAX version. Apple MPS is not supported: it has no float64 JAX backend, and the [Apple MPS findings](performance/2026-09-19-apple-mps-feasibility.md) recorded CPU float64 as faster than the tested substitutions.

What to expect, from the [Linux GPU opportunities note](performance/2026-09-21-linux-cuda-gpu-opportunities.md):

- Consumer and workstation cards run float64 at 1/64 of their float32 rate. At the recorded one-eighth empirical scale the GP likelihood gradient took 66 ms on an RTX PRO 6000, 172 ms on an RTX 3090 and 644 ms on the best contended CPU row; against an idle, well-threaded CPU the 3090 is expected to sit within 2 to 4x. Float32 is outside the package's numerical contract.
- R-MSRM stays on the CPU; a GPU would accelerate only a 30 ms residual.
- Posterior sampling is where a GPU pays most, and it is still expensive: one NUTS leapfrog step cost 2.0 s on the CPU, 0.37 s on the 3090 and 0.16 s on the PRO 6000.

Environment variables read by JAX before Python starts:

- `CUDA_VISIBLE_DEVICES` selects the card on a multi-GPU machine.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from reserving most of the card's memory at startup, which matters on a shared card.
- `JAX_PLATFORMS=cpu` (or `JAX_PLATFORM_NAME=cpu`) pins the CPU backend even when the plugin is installed.

The package checks the environment once per process at the start of a fit. On Linux x86_64 with an NVIDIA driver visible, it warns when JAX initialized its CPU backend without the CUDA plugin, naming the install command above, and when the plugin is installed but JAX still fell back to the CPU. It stays silent when the platform was pinned through the environment, on macOS, and on machines without a driver. JAX prints its own fallback message in the first case; the package's warning adds the install command. The package never installs anything or changes device selection itself.

## Thread settings

Set BLAS and OpenMP thread counts before starting Python; the package never changes process-global thread limits during a fit.

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` reproduces CI and is the safe choice on a laptop or a shared machine.
- For GP-MSRM fits on a many-core workstation, 8 to 16 threads was the measured sweet spot. The CPU factorization runs LAPACK inside a JAX host callback while JAX's own thread pool already occupies the cores, so the BLAS default of one thread per core oversubscribes the machine. On a 64-core workstation that default slowed a 2,727-dimension factorization by up to 10x relative to 16 threads; see the [Linux GPU opportunities note](performance/2026-09-21-linux-cuda-gpu-opportunities.md). The package warns once per process when the BLAS pool exceeds half the physical cores.
- R-MSRM restarts with `n_jobs > 1` run in separate processes that limit themselves to one BLAS thread each, so the parent process setting matters only for serial fits.

Use the same settings when comparing timings; thread count and background load change CPU results by several-fold.

## Record the version you use

An editable installation follows changes in your checkout. For a research analysis, record the source commit (`git rev-parse HEAD`), model configuration, random seed, and dependency versions. Keep the original environment for replaying historical fitted artifacts.

Continue with [data preparation](data.md) and the [runnable tutorials](tutorials.md).
