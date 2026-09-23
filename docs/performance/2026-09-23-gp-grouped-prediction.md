# Reusing grouped GP prediction calculations

Grouped marginal prediction now factors the conditioning system once per
parameter draw and output stream, and shares native-time response covariance
across that stream's features. This implements the optimization identified by
the [temporal-noise research scorer](2026-09-23-gp-temporal-refits.md).
It changes the production prediction calculation, with the same fitted model,
response approximation, conditional means, and marginal variances.

The independent-noise Gaussian empirical benchmark on RTX PRO 6000 falls from
52.2 to 6.6 seconds per repeated call, approximately **8 times faster**. Its
first call falls from 54.5 to 10.5 seconds. These calls predict eight brain
features at sixteen times from the existing three-factor fit with 52,964
conditioning observations. They include compilation and transfers as well as
numerical work; they are not kernel-only or model-fitting timings.

The correlated-noise gamma empirical case on RTX 3090 improves from 110.2 to
7.7 seconds on the repeated call, **14.4 times faster**, with maximum moment
differences of `2.3e-9`. A synthetic six-factor/four-draw control on RTX PRO 6000
improves 7.5 times. These comparisons support the implementation for the tested
workloads without claiming a general posterior-inference speedup.

<!-- BENCHMARKS:START -->

## Matched before/after timings

Seconds per complete marginal-prediction call. Repeated calls create fresh caches; the table reports the median. All queries use sixteen times and include observation noise. All cases request eight output features except the single-output control.

| Workload | Device | First: before → after | Repeated: before → after | Repeated speedup | Max moment error |
| --- | --- | ---: | ---: | ---: | ---: |
| Empirical Gaussian / independent | RTX PRO 6000 | 54.55 → 10.48 | 52.22 → 6.56 | 7.96× | 2.8e-13 |
| Empirical gamma / OU | RTX 3090 | 113.50 → 9.93 | 110.25 → 7.68 | 14.36× | 2.2e-09 |
| Synthetic Gaussian / OU | CPU | 117.92 → 5.22 | 117.02 → 4.99 | 23.47× | 1.4e-12 |
| Synthetic Gaussian / independent, one output | CPU | 4.26 → 3.50 | 3.74 → 3.11 | 1.20× | 4.4e-15 |
| Synthetic Gaussian / independent, K=6 / draws=4 | RTX PRO 6000 | 53.55 → 10.37 | 51.12 → 6.85 | 7.46× | 2.3e-14 |

The [aggregate results](2026-09-23-gp-grouped-prediction-results.json) contain all timings, dimensions, source hashes, and agreement checks. There are two repeated calls after the first call in each case except the RTX 3090 empirical case, which has one. These small timing samples describe the tested workloads, not a general speed guarantee.

The empirical cases use three factors, 841 temporal nodes, and quadrature order 384. The synthetic OU case has 3,017 observations and 160 nodes at order 384. The single-output and six-factor controls use order 96; the six-factor case repeats one parameter point four times to exercise draw processing. Maximum mean/variance disagreement across all five comparisons is **2.3e-9**, below the unchanged `1e-7` gate.

<!-- BENCHMARKS:END -->

## Calculation and scope

Previously, each feature's prediction independently evaluated the response
covariance and factored the grouped conditioning system inside each query/draw
chunk. With OU residuals, shared response covariance was also evaluated against
all observation rows, repeating the same temporal functionals across features.

The [grouped prediction implementation](../../src/multimodalsrm/bayesian/grouped_prediction.py)
now reuses that conditioning state across the output features in a result.
It evaluates temporal covariance at grouped training nodes and unique query
times, then gathers the appropriate feature loadings. All cross-factor terms
are retained. Observed OU residuals still contribute their private conditional
mean and variance; white noise remains an independent replicate. An excluded
feature with no conditioning observations in the queried run has no private
OU cross-covariance, so that observation-sized calculation is avoided.

One draw's factorization is retained at a time. Query work uses at most 128
unique times per temporal chunk and 32 time/feature pairs per solve batch.
The returned draw/time/feature arrays still scale with the requested output.
No persistent cache is attached to the estimator, and different calls, runs,
draws, and conditioning datasets cannot reuse stale values. Separate subject
or modality results still construct their own conditioning state.

The production `predict` and `infer_latent` paths use this calculation for
grouped algebra without run baselines. Dense, spectral, state-space, and
run-baseline calculations retain their existing paths. Inference defaults,
fitting objectives, quadrature orders, and the scientific response/noise
specification are unchanged. This prediction speedup does not improve the
poor brain/face validation scores found in the preceding model comparison.

## Correctness checks

New regression tests compare multi-feature, multi-draw results with dense
Gaussian conditioning. They cover independent, full OU, and partial OU noise;
one and three latent factors; changed response/noise/loading parameters between
draws; masked feature observations; zero loadings; duplicate and unsorted query
times; and both query batching boundaries. Rotated latent directions retain
cross-factor covariance. Public frozen-conditioning checks compare dense and
grouped results, preserve metadata, and check that new donor values affect only
the new conditioned estimator.

Existing regression checks also cover noisy OU interpolation at observed times,
white-noise replicas, mixed DoubleGamma/Gamma/Gaussian/Bateman responses, learned
GP timescales, latent reporting coordinates, and run-baseline fallback.
The targeted checks comprise 92 passing cases, including nine new cases;
74 infrastructure checks, Ruff, API-reference consistency, release readiness,
and the strict documentation build also pass. The CI smoke suite now includes
multi-feature white/OU prediction and draw-aligned latent rotation checks.

The empirical and synthetic benchmarks require maximum absolute before/after
errors of `1e-7` in both means and variances. The initial RTX 3090 attempt stopped
because a stricter `1e-10` repeatability check rejected approximately `1.7e-9`
variation in the **original** implementation. The rerun records repeat variation
and applies the same `1e-7` moment gate to repeatability and before/after
agreement. The aborted run is retained locally and excluded from timing results.

## Reproduction and limits

The [benchmark runner](../../scripts/benchmark_gp_grouped_prediction.py) loads
the original prediction implementation from commit
`16e6574007018bf140b4ddad25b2b8e2d7b02880` and compares identical parameters,
observations, queries, and response quadrature. That commit must be available
in the local Git checkout. Caches are cleared between implementations; each
repeat makes a fresh prediction call. The baseline feature loop matches the
former production result assembly. Timings include NumPy materialization and
any compilation, rather than measuring an already-compiled accelerator kernel.

All measurements use float64 and one BLAS/OMP thread, with eight pinned CPU
cores per process on the AMD Ryzen Threadripper PRO 3995WX host. GPU jobs have
exclusive use of their respective card; CPU jobs may overlap on disjoint cores.
Compare before/after within each row. Rows with different devices, noise models,
quadrature orders, or data do not establish CPU/GPU or family rankings.
Repeated parameter points in the scaling benchmark exercise prediction cost;
they are not a posterior sample or evidence of posterior convergence. Different
draw values are covered by the numerical regression tests.

For a synthetic comparison, no participant data are required:

```sh
PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_gp_grouped_prediction.py \
  --noise ou --query-features 8 \
  --output local_data/prediction-benchmark.json
```

Use `--fit` with a private refit artifact to reproduce an empirical comparison;
the saved fit supplies the family, factor count, and noise specification rather
than the synthetic CLI defaults. Empirical reproduction also requires the
separately available data and historical loader. Fitted parameters and
participant observations remain local; the public result JSON contains only
aggregate dimensions, errors, timings, and provenance.
