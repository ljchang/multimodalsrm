# R initialization for GP MAP

`SearchConfig(r_init=True)` initializes the first GP MAP restart with a bounded
CPU R-MSRM fit. The [configuration guide](../getting-started.md) describes its
budgets, parameter translation, diagnostics and fallback. This preliminary fit
is separate from NUTS warmup and does not replace sampler adaptation.

## Synthetic comparison

The committed [benchmark script](https://github.com/ljchang/multimodalsrm/blob/main/scripts/benchmark_r_initialization.py)
compares one historical data-based start with one R start. It generates three
participants on unequal modality clocks with feature masks. The single-factor
case learns a Gaussian response lag; the two-factor case uses Identity responses.
Neither case includes empirical observations or tests SCR response recovery.

Each fit runs in a fresh process, with the same seed, priors, optimizer limits
and physical-gradient tolerance. Both configurations receive up to 1,200 search
iterations and 200 refinement iterations. Total time includes preparation,
compilation, R initialization when enabled, optimization and diagnostics. The
table reports medians across seeds 31, 32 and 33 for grouped algebra; dense and
state-space checks use seed 31. Iterations include search and refinement.

| Case | Algebra | Historical iterations | R-start iterations | Historical time | R-start time |
| --- | --- | ---: | ---: | ---: | ---: |
| One factor, Gaussian | grouped | 81 | 60 | 7.33 s | 7.61 s |
| Two factors, Identity | grouped | 124 | 85 | 7.07 s | 7.17 s |
| One factor, Gaussian | dense | 116 | 67 | 7.48 s | 7.51 s |
| Two factors, Identity | state_space | 117 | 88 | 4.86 s | 4.52 s |

All 16 fits passed the physical-gradient threshold of `1e-3`. For Gaussian
seed 31, the R start found a lower negative log posterior: approximately
`-84.79` versus `-64.56`, with agreement between dense and grouped algebra.
The other paired fits agreed in objective to numerical precision. A single
start does not establish global mode coverage; production searches retain
the other configured prior-quantile starts.

These small cold fits show reduced optimizer work, with similar total time
for dense/grouped algebra. Compilation dominates at this size. The state-space
pair was about 7% faster, but one timing pair is insufficient for a general
speedup claim. Larger data, GPU execution, SCR filters and empirical analyses
require their own measurements. Fewer iterations do not guarantee a faster fit
or better predictions.

Measurements were made on Linux CPU with float64, affinity restricted to eight
cores and one BLAS/OpenMP thread. Exact runtime versions, CPU model and individual
fit records are in the [results file](r-map-initialization-results.json). Timings
are observations from this run, not performance test thresholds.

## Reproduce

Install the Bayesian extra, then run each configuration in a fresh process:

```sh
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_r_initialization.py --case gaussian --seed 31 --no-r-init
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_r_initialization.py --case gaussian --seed 31 --r-init
```

Repeat for seeds 32 and 33 and `--case multifactor`. Use `--algebra dense` for
the Gaussian backend check or `--algebra state_space --case multifactor` for
the Identity state-space check. On Linux, prefix the command with an appropriate
`taskset -c` CPU list to reproduce a fixed affinity. Compare convergence and
objectives alongside wall time, and keep thread settings and background load
consistent.
