# Repeated factor blocks, transition reuse, and smaller Gaussian banks

The clearest improvement in this round is **lower compilation cost and memory**.
The block prototype cuts six-factor gamma compilation from 23.3 to 7.6 seconds
and estimated live memory from 1.70 to 0.81 GB. For three-factor Gaussian,
compilation falls from 71.1 to 14.2 seconds. Warm speedups are modest: about
1.34x for gamma and 1.04x for Gaussian without transition reuse.

Reusing repeated transitions raises warm gains to approximately 1.38x and
1.14x, respectively, but increases compilation cost relative to the simpler
block prototype. All variants fail the extreme empirical low-noise gradient
gate. These results do not support changing production defaults yet.

The smaller Gaussian order-16 bank preserves the previously fitted MAP's
projected-gradient qualification **at that point**, but does not satisfy the
existing full-box covariance bound or broader accuracy gates. Order 12 also
fails the local MAP check. Supported production orders remain unchanged.

This follows [higher-factor parallel filtering](2026-09-22-gp-parallel-scaling.md).
See the [portable results](2026-09-23-gp-block-filter-results.json) for individual
points, timings, numerical errors, and source hashes. The experiments change
research scripts only; no new fits or predictive scores are reported here.

## Paired GPU measurements

<!-- TIMINGS:START -->

| Model / calculation | Warm speedup | Compile reference / candidate (s) | Estimated memory reference / candidate (GB) | Accuracy points passing |
| --- | ---: | ---: | ---: | ---: |
| Gamma K=6, block prediction | 1.16x | 23.3 / 8.6 | 1.70 / 0.81 | 7/8 |
| Gamma K=6, compact Joseph | 1.34x | 23.3 / 7.6 | 1.70 / 0.81 | 7/8 |
| Gaussian K=3, block prediction | 1.01x | 71.1 / 22.2 | 5.52 / 4.60 | 7/8 |
| Gaussian K=3, compact Joseph | 1.04x | 71.1 / 14.2 | 5.52 / 4.60 | 7/8 |
| Gamma K=6, compact Joseph, reuse | 1.38x | 23.4 / 14.8 | 1.70 / 0.86 | 7/8 |
| Gaussian K=3, compact Joseph, reuse | 1.14x | 71.1 / 67.9 | 5.52 / 4.64 | 7/8 |
| Gamma K=6, original update, reuse | 1.02x | 23.1 / 12.3 | 1.70 / 0.86 | 7/8 |

<!-- TIMINGS:END -->

Gamma runs use the RTX 3090; Gaussian runs use the RTX PRO 6000. Each speedup
compares against the original sequential filter in the **same process and on
the same GPU**, at the same initialization. Cross-family seconds are not a
controlled family comparison. Warm times are medians of three completed
float64 value-and-gradient evaluations. Compilation excludes lowering and
first execution. Memory is the compiler's estimated argument/output/temporary
storage, not measured process/device peak or total compiler/runtime usage.

The dataset is the same two-subject, 500-second, 100-parcel training set from
the previous qualification: 57,671 scalar observations at 942 nodes. Brain
uses fixed DoubleGamma, EDA uses learned BatemanSCR, and face/ratings use
gamma-3 or Gaussian with matched FWHM/peak priors. Pulse and respiration are
excluded. The full cross-factor covariance is retained in every variant.

## Algebraic changes

The [prototype](../../scripts/gp_block_filter.py) tests three changes separately:

1. **Block prediction.** Apply the same single-factor transition to each pair
   of factor covariance blocks. Store transitions once per time instead of
   repeating them on a large block diagonal. Keep the full Joseph covariance
   multiplication for the observation update.
2. **Compact Joseph products.** Associate the same Joseph update using the
   rank-K observation products. This removes large dense multiplications but
   changes floating-point cancellation; it requires its own numerical checks.
3. **Transition reuse.** Compute repeated transitions once and gather them back
   to the original node order. Both endpoint modalities and the native time
   difference identify the parameter dependence. Exact elapsed time is also
   part of the key, preserving differences from floating-point time arithmetic.
   Equal elapsed values alone are insufficient: their lag derivatives can differ.

Reuse evaluates a batch of up to 64 representatives. If the number of groups
exceeds capacity, it uses the original per-node transition calculation. There
is no timestamp rounding, latent-factor independence assumption, dropped
observation, noise floor, or covariance jitter. Both branches are compiled,
which helps explain why reuse can increase startup cost and compiler memory.

Separate GPU diagnostics confirm reuse is active on the benchmark points.
For Gaussian, 35–45 groups cover the 942 nodes. That large reduction in repeated
transition work does not translate into a similar reduction in end-to-end
gradient time. The remaining filtering/derivative work and execution overhead
need profiling before pursuing more transition-specific optimization.

A final gamma probe reuses transitions while reconstructing the full matrices
inside the scan and calling the original production update. It gives only
about 1.02x warm speedup and also fails the extreme empirical gate. This checks
that compact Joseph contraction is not the only possible source of differences;
batching and derivative accumulation can matter in this difficult regime.

## Accuracy and its limits

Each empirical case checks two initializations, near-lower/upper response
bounds, observation variances of `1e-3` and `1e-6`, rank-one loadings, and zero
loadings. The predeclared gate is:

- Objective difference: at most `1e-6 + 1e-9*abs(reference)`.
- Every gradient component: at most `1e-5 + 1e-7*abs(reference component)`.
- All values and gradients must be finite.

Every variant passes seven of eight points and fails at variance `1e-6`.
Those empirical stress observations are generally implausible at such low
noise, and gradients can reach `1e16`. These failures establish disagreement
with the production recurrence under the gate; they do not by themselves
prove which full-data gradient is more accurate. No dense full-data oracle
was evaluated at that extreme point.

The small-fixture [dense covariance checks](../../scripts/check_gp_parallel_numerics.py)
use the identical response realization to isolate numerical algebra from
kernel approximation. Unlike the earlier prefix filter, both block variants
pass the componentwise gradient comparison on the coherent low-noise gamma
and Gaussian draws. This is encouraging, but does not resolve the larger
empirical disagreement.

The [transition reuse check](../../scripts/check_gp_transition_reuse.py) covers
equal/near-equal clocks, equal elapsed values with different lag derivatives,
large shifts, a nonlinear lag parameterization, and forced capacity fallback.
Its eight comparisons agree in objective exactly at displayed precision and
in gradients to less than `8e-15`. Independent finite-difference errors are
below `1e-9`. This checks the grouping logic, not all numerical behavior of
a long, high-dimensional filter.

## Smaller Gaussian bank at the saved MAP

The [order probe](../../scripts/qualify_gp_gaussian_map_orders.py) compares
orders 16 and 12 with the existing 60-state Gaussian reference at K=1.
It reuses the earlier five-parcel training fold, canonical priors, and saved
Gaussian MAP. All observations, parameter names, and bounds are checked equal.

<!-- ORDERS:START -->

| Gaussian order | States (K=1) | Full-box covariance bound | Objective error at existing MAP | Largest gradient change | Projected gradient | Local MAP gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 52 | 0.000153 | 3.34e-05 | 9.02e-05 | 8.6e-05 | Pass |
| 12 | 44 | 0.0188 | 0.0103 | 0.00695 | 0.00697 | Fail |

<!-- ORDERS:END -->

The local gate, declared before evaluation, requires largest gradient change
`<= 1e-4` and projected physical gradient `<= 1e-3`. Order 16 passes this local
criterion but fails the stricter componentwise comparison. The production
covariance tolerance is `1e-6`; both smaller-order full-box bounds exceed it.
Preparation explicitly uses a **research-only tolerance of 0.1** to evaluate
these otherwise rejected representations. This is not a proposed tolerance
change. The probe restores the original template factory immediately after
preparing each candidate.

Two initializations and the same six stress points are also evaluated. Passing
the saved MAP does not establish accuracy throughout optimization, predictions,
the parameter posterior, or a higher-factor model. Order 16 remains a candidate
for narrower-domain qualification. Order 12 is not supported by this MAP check.
Single-call timings in this order probe are diagnostics, not a controlled
GPU or full-fit speedup benchmark.

## Reproduce and review

Use float64, the existing Bayesian/CUDA environment, and the ignored empirical
loader/dataset. Each timed GPU process uses eight CPU cores and one BLAS/OMP
thread. Python is 3.12.13 and JAX is 0.11.2.

```sh
PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/qualify_gp_block_filter.py --candidate gamma3 --features 6 \
  --output local_data/gp-block-filter-2026-09-23/gamma3-k6.json
```

Select the GPU using `CUDA_VISIBLE_DEVICES`. Repeat with `--candidate gaussian
--features 3`. For reuse, add `--joseph compact --transition-cache 64` and use a
new output path. `--inspect-only` records group counts without timing the filter.
`--joseph original` retains the original observation-update arithmetic.

The synthetic covariance check accepts `--implementation block`. The transition
reuse check needs no participant data. The smaller-order probe additionally
requires the saved Gaussian fit specified by `--fit`. Raw observations and
loadings remain in ignored local storage; portable results contain aggregates.

Review this alongside the [research overview](gp-response-research.md).
The next useful scientific step is the already prepared larger-factor,
full-parcellation held-out comparison using the existing backend. For further
engineering work, profile the remaining GPU filter/gradient cost and establish
an independent reference for the extreme low-noise disagreement before
promoting a new arithmetic path. No NUTS/Gibbs convergence or VI comparison is
established by these likelihood benchmarks.
