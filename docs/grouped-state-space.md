# Grouped state-space filtering and smoothing

Select `linear_algebra="state_space"` with fixed or learned response parameters and a
strictly positive noise prior to use grouped observation updates automatically.
This is a computational improvement to the same MAP target. `linear_algebra="grouped"`
continues to select the existing dense covariance over unique functionals.

```python
import numpy as np

from multimodalsrm import BatemanSCR, Identity, Response
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior

model = BayesianMultimodalSRM(
    features=2,
    responses={
        "reference": Response(Identity(), estimate=False, pooling="shared"),
        "scr": Response(BatemanSCR(), estimate=False, pooling="shared"),
    },
    priors=BayesianPriors(noise=Prior.lognormal(np.log(0.1), 0.7)),
    inference="map",
    linear_algebra="state_space",
)
```

The noise prior describes observation **variance** in supplied data units; the
numbers above are illustrative. Grouping applies to all supported responses, including Identity, Gaussian,
Gamma, DoubleGamma, BachSCR and BatemanSCR. Parameter restrictions are unchanged:
Gamma shapes and Bach shape parameters remain fixed; supported widths, scales,
ratios, Bateman rise/decay and response lags may be learned. Gaussian approximations and finite-response
tail qualifications retain their existing covariance error bounds.

## Numerical contract

Each exact `(modality, native timestamp)` pair is one observation node. All
eligible feature measurements across participants contribute to that node's
loading information and precision-weighted observations. These statistics are
recomputed for every parameter evaluation. Masks remove individual observations,
and noise remains specific to its participant/modality group. Each run starts
from its own stationary distribution; loadings and other shared parameters keep
their existing cross-run meaning.

The filter uses a small positive-definite solve in whitened functional
coordinates and a Joseph covariance update. It never inverts the loading
information matrix, so zero and rank-deficient loadings remain supported. The
likelihood evaluates original observation residuals after each node update and
includes all noise and determinant terms. This avoids subtracting large, nearly
equal quadratic terms at high signal-to-noise ratios. No covariance jitter or
time binning is added.

The RTS smoother runs over the observation nodes and requested query events.
It retains covariance between factors and accepts repeated queries, exact ties,
near ties and queries outside the observation range. Query events add no
observations. The same projection and target-exclusion workflows apply to
training and frozen new-run predictions.

## Scope and compatibility

For learned responses, node membership remains fixed at native modality/time
pairs. Each parameter evaluation realizes the current response, shifts node and
query clocks, and sorts the resulting events. Different modalities remain
separate nodes even when their shifted clocks coincide. Gaussian widths also
change internal clock shifts; these are included in the same calculation.
Exact-time transition derivatives and the existing differentiable RTS tie
update preserve sensitivities when event order changes or queries meet nodes.
A lognormal noise prior or a prior with a strictly positive lower bound admits
grouping. Priors including zero use scalar updates for the entire fit, even at
positive current parameter values, so noiseless evaluations remain supported.
State-space posterior sampling and learned GP timescales remain outside the
current backend's supported scope.

Fitted archive formats and saved configuration/diagnostic evidence are unchanged.
Historical fixed- and learned-response archives can use grouped prediction after loading;
floating-point results may differ slightly because measurement arithmetic is
regrouped. Tests compare such replay against scalar predictions and verify
that the original fit evidence survives loading and resaving.

## Cost and measurement

For `N` scalar measurements, `U` unique modality/time nodes and state dimension
`d`, transition storage becomes proportional to `U d²` instead of `N d²`.
Filtering work depends on `U` plus an observation pass to construct statistics
and score residuals. Smoothing stores forward moments at nodes and query events.
Checkpointing the filter scan body reduces saved intermediate operations during
differentiation, but its retained state history is still linear in node count;
this does not provide constant-memory gradients for arbitrarily long runs.

The [benchmark script](https://github.com/ljchang/multimodalsrm/blob/main/scripts/benchmark_grouped_state_space.py)
compares full MAP fits using the previous scalar recurrence and the grouped
implementation, then times warm likelihood/gradient and smoother calls at
identical physical parameter vectors. Run each arm in a fresh process:

```sh
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_grouped_state_space.py --backend scalar --response bateman
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_grouped_state_space.py --backend grouped --response bateman
```

Use `--response identity`, `--times 128`, or `--channels 32` to vary the
workload. Compare fit convergence, endpoint objectives and probe agreement as
well as timing. These seeded synthetic checks measure computation and numerical
agreement; empirical prediction and response recovery require separate studies.

### Measured CPU comparison

Eight complete synthetic fits used two factors, two participants, 16 features
per modality, two modality clocks, masks and one run. Both arms used identical
seeds, priors and search budgets. All eight fits passed the `1e-3` physical
projected-gradient threshold and paired final objectives agreed within `1e-6`.
Each full-fit timing is one cold process; warm operation timings are medians
of five calls at identical physical parameter vectors.

| Responses | Times per stream after the initial gap | Scalar fit | Grouped fit | Fit ratio | Gradient ratio | Smoother ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 32 | 6.16 s | 5.01 s | 1.23× | 7.91× | 8.57× |
| Identity | 128 | 12.94 s | 5.74 s | 2.26× | 5.95× | 8.95× |
| Identity + fixed Bateman | 32 | 9.57 s | 7.03 s | 1.36× | 15.34× | 16.93× |
| Identity + fixed Bateman | 128 | 19.59 s | 8.00 s | 2.45× | 8.24× | 9.23× |

Ratios are scalar time divided by grouped time. The larger Bateman case reduced
filtering events from 8,002 to 257 and stored transition arrays from 8.19 MB
to 0.263 MB. Across all cases, the largest absolute differences at matched
parameter probes were `8.8e-11` in objective, `6.5e-10` in physical gradients,
`1.9e-13` in smoother means and `3.0e-15` in factor covariances.

These measurements used Linux CPU float64, eight-core affinity and one
BLAS/OpenMP thread. Other tests ran on separate CPU affinity sets, so timings
are illustrative rather than dedicated-machine estimates. The full-fit timings
include cold setup and compilation. These are comparisons with the previous
scalar state-space implementation; they do not establish superiority over
dense grouped GP on every workload or measure GPU speed. See the
[individual results and environment](grouped-state-space-results.json).
