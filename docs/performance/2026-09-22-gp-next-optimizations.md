# Next GP optimizations: parallel filtering, blocked Gibbs, and structured VI

Research on September 22, 2026, against `main` at `d066357` (v0.2.0).
Production code and defaults are unchanged. The new
[parallel-filter probe](../../scripts/benchmark_gp_parallel_filter.py) is a research
implementation. Measurements and provenance are in
[the results file](2026-09-22-gp-next-results.json).

Follow-up: [Gaussian response efficiency and alternative families](2026-09-22-gp-gaussian-followup.md)
adds lower-order measurements and isolates a Gaussian compilation bottleneck.
It prioritizes the shared likelihood/filter improvements while leaving the
sampler choice open pending a converged NUTS comparison.

**Recommendation:** develop a numerically qualified parallel state-space filter
and simulation smoother, and reduce Gaussian-response state size. These shared
computations are the best-supported targets for large NVIDIA GPU gains. Finish
blocked Gibbs as a candidate and obtain a converged NUTS comparison before
selecting a sampler. Keep structured variational inference as another inference
mode built on the same Gaussian machinery. Changing response families also
changes the scientific model and needs a separate fit-quality comparison.

State-space algebra and variational inference answer different questions.
The former changes how the latent Gaussian distribution is computed; the latter
approximates the posterior over unknown quantities. With the present Matérn-3/2
latent kernel and Identity responses, the state-space representation is exact.
Gaussian response banks introduce an explicitly qualified approximation.
Either representation can support MAP, Gibbs, or structured VI.

## What the merged work already bought

The earlier notes proposed several changes that are now implemented:

| Change | Current evidence | Remaining limitation |
| --- | --- | --- |
| Grouped state-space observations, including learned responses | Fresh learned-Bateman CPU probe: 129.8 ms to 12.5 ms per objective/gradient, a 10.4× gain over the old scalar recurrence | Time recurrence is still sequential; Gaussian banks can make states large |
| Vectorized diagnostics | Fresh paired comparison, 2 chains × 15 draws × 57,758 quantities: former ArviZ path 90.8 s, current path 0.665 s, 136× faster | Does not accelerate sampling or fix mixing |
| R initialization | Now available as opt-in `SearchConfig(r_init=True)` | No general empirical full-fit speedup established |
| CUDA installation and CPU thread guidance | Supported installation extra and warnings are present | Moving the same dense factorization to CUDA retains cubic scaling |

The diagnostic comparison uses identical arrays and ArviZ 1.3.0 in one process,
with imports outside the timer. Maximum absolute discrepancies were below
`1e-13` for ESS, zero for R-hat, and below `4e-16` for MCSE. The old path was
timed once; the new time is the median of three. These results are more directly
comparable than the earlier profiled, contended-machine timings.

The [committed complete-fit state-space study](../grouped-state-space.md) already
reports 1.23–2.45× gains for fixed responses and 2.25–6.54× for learned Bateman
responses against scalar filtering, with qualified optimizer endpoints. Those
are CPU results on small synthetic problems. The
[R-initialization study](r-map-initialization.md) found fewer iterations but
similar total times. The saved real-data investigation found two different lag
basins and no iteration reduction. The earlier synthetic 6.5× warm-start result
should therefore not be used as a general prediction.

## New NVIDIA result: parallelize time

`state_space_grouped.update` uses stable rank-K observation statistics, but
`nll` still calls a sequential `lax.scan`. Independent groups of temporal
messages can instead be combined in an associative prefix scan.
[Särkkä and García-Fernández](https://arxiv.org/abs/1905.13002) derive this
algorithm. It has logarithmic dependency depth with sufficient processors;
total work and storage still grow with recording length and state size.

The new probe constructs Gaussian information messages without inverting the
loading information or process covariance, so zero loadings and tied events are
allowed. It reuses production observation statistics and stable likelihood
scoring. It currently duplicates some update arithmetic to obtain that score.
It changes neither priors nor float64 precision and adds no covariance jitter
or timestamp rounding.

Warm objective-and-gradient medians on the RTX PRO 6000:

| Synthetic case | Nodes | State dimension | Dense grouped | Current sequential state-space | Parallel prototype | Gain over dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity, 450 s, K=3 | 925 | 6 | 42.9 ms | 486.5 ms | 5.56 ms | 7.7× |
| Identity, 1,800 s, K=3 | 3,700 | 6 | 1,929 ms | 1,991 ms | 18.88 ms | 102× |
| Bateman, learned lags, 450 s, K=3 | 924 | 12 | Not measured | 494.0 ms | 7.10 ms | 69.6× versus sequential |

On the RTX 3090, the same 925-node Identity case took 133.0 ms with dense
grouped and 5.77 ms with parallel filtering, a 23.1× gain. The 450 s CPU case
took 54.0 ms sequentially and 70.9 ms in parallel: use a sequential CPU path
when it is faster. These CPU runs used eight-core affinity and one BLAS thread;
they are not a search for the best possible CPU configuration.

At the saved fitted parameters for the empirical Identity case (1,346 nodes,
2,787 parameters), the RTX 3090 took **363.1 ms dense versus 7.05 ms parallel**,
a 51.5× gain. Its sequential GPU filter took 940.6 ms. The corresponding
eight-thread CPU probe took 812.9 ms dense, 100.1 ms sequential and 120.3 ms
parallel. At this near-optimum point, the 3090 parallel gradient differed from
the sequential gradient by at most `2.9e-12` absolutely; the dense gradient
differed by at most `2.2e-10`. An earlier empirical default-vector probe on the
PRO 6000 took 140.3 ms dense versus 7.06 ms parallel.

Repeating at the **same saved fitted parameters on the PRO 6000** gave
**120.3 ms dense versus 6.28 ms parallel, a 19.2× gain**. The sequential filter
took 702.5 ms. The parallel/sequential maximum absolute gradient difference
was `2.7e-12`. Its parallel first evaluation took 12.34 s, compared with 4.11 s
for dense; the extra first-call cost amortizes after roughly 73 evaluations at
these warm timings, assuming other costs are equal. This is an illustration,
not a complete-fit measurement.

Relative gradient differences between parallel and sequential evaluation were
below `7e-15` in these ordinary synthetic probes. Additional checks reused the
repository's masked, two-run cases with exact ties, adjacent floating-point
timestamps, large gaps, and full/rank-one/zero loading matrices. Value and
gradient agreement was near machine precision. At noise variance `1e-7`, the
high-SNR test had a maximum gradient difference of `0.0011` but relative gradient
error `4.6e-12`; it passed the existing high-SNR comparison tolerance.

These are **evaluation speedups**, not demonstrated complete-fit or ESS-per-second
speedups. First evaluations include compilation: 10.4–14.3 s for the successful
parallel GPU probes above. No new posterior chain was run using this filter.
A faster gradient does not reduce NUTS trajectory lengths by itself.

### Gaussian response costs

A synthetic case with two different Gaussian widths reached a 659 ms warm
sequential gradient but exceeded the four-minute process budget during the
first parallel evaluation. Compilation and execution were not separated, so
there is no parallel warm-time result for that case. Preserve this failed
attempt rather than extrapolating the Identity result to Gaussian banks.

The empirical Gaussian case with learned lags (1,298 nodes, state dimension 78)
also exceeded its seven-minute process budget during the first parallel
evaluation. The sequential gradient completed: first call 54.7 s, warm median
797 ms. Thus neither a Gaussian parallel warm-speed claim nor a new dense versus
parallel Gaussian comparison is established. The retained partial results mark
the timeout explicitly.

Large-state improvements should target:

1. **Transitions in factor blocks.** For K factors and d states per factor,
   propagate each of the K² covariance blocks with d-dimensional transitions,
   instead of a generic Kd-dimensional dense multiplication. Cross-factor
   posterior covariance must remain present. This saves about a factor K in
   the propagation arithmetic, not the entire objective.
2. **Low-rank or square-root observation updates.** The measurement has rank K,
   but the current Joseph form explicitly multiplies full state matrices.
   Exploit that structure while retaining its high-SNR stability. A subtractive
   covariance formula is not automatically a numerically acceptable replacement.
3. **Smaller response banks with measured error.** Current compact Gaussian
   templates use orders 20 or 24. Test qualified reduced realizations over the
   complete allowed width/timescale range, then compare likelihoods, gradients,
   response recovery and predictive intervals. The model currently has a
   128-state-per-factor limit. Shared fixed widths already reuse banks;
   independently learned widths generally require separate banks.
4. **Compilation and differentiation.** Profile large closed-over transition
   arrays and reverse-mode storage before attributing the timeout to arithmetic.
   Passing arrays explicitly, chunking the prefix scan, and a smoother-based
   score/Fisher-identity derivative are candidates. None was measured here.

The [2025 GPU scan study](https://arxiv.org/abs/2511.10363) shows that scan choice
and kernel implementation affect throughput and introduces a two-GPU two-filter
smoother. Start with a qualified JAX implementation; consider fused CUDA/Pallas
kernels after profiling. With this workstation's unequal GPUs, separate chains
or independent runs are the simpler first use of both cards.

## Resume blocked Gibbs, with the shift move

The interrupted investigation is recoverable from branch
`docs/empirical-evaluation-2026-09-22`, commit `7d48a21`, and ignored
`local_data/empirical-eval-2026-09-22/`. Its report is not on current `main`.
The sampler is `gibbs_proto2.py`; `gibbs_proto.py` is the earlier version.

I re-ran **diagnostics on its saved draws**, using current main, rather than
claiming a new sampling run. On the full 2,787-parameter, 1,346-node real-data
Identity case, the two shift-enabled chains used 445.3 seconds of sampling
time for 5,000 sweeps each, retaining every fifth sweep:

| Reported quantity | Minimum bulk ESS | Minimum tail ESS | Maximum R-hat |
| --- | ---: | ---: | ---: |
| Marginal log likelihood | 1,961 | 2,047 | 1.001 |
| Noise variances | 1,730 | 1,660 | 1.004 |
| Squared loading norms | 1,145 | 1,377 | 1.007 |
| Offsets | 1,521 | 1,283 | 1.005 |

Without the shift move, the offsets had bulk ESS 5.18 and R-hat 1.315 at
essentially the same sampling time. Shifting the latent level and compensating
the offsets removes a serious mixing problem. Scale and rotation moves also
need to be retained or independently assessed, not dropped just to shorten a
sweep. Interweaving is well established in related factor models;
[Kastner et al.](https://arxiv.org/abs/1602.08154) provide a useful comparison.

This evidence strongly supports further development, but it is narrower than
general posterior qualification. The prototype fixes responses and timescale,
assumes the current independent Gaussian observation model, and lacks the K=1
positive anchor, production multi-run behavior, persistence and diagnostics
contracts. Loading norms do not exhaust all rotation-invariant scientific
quantities: additionally check cross-feature covariance, predicted observations,
latent subspaces and response parameters. The historical small NUTS comparison
was only moderately converged; the real-data NUTS chains were not converged.
They are evidence of expensive sampling at those budgets, not an exact oracle
or a defensible universal ESS speedup ratio.

The existing prototype also adds `1e-12` jitter in simulation smoothing and
treats intervals below `1e-9` as ties. The prior investigation fixed a major
scale-move defect caused by inconsistent treatment of almost-tied times.
Those choices require replacing/qualifying against the production exact-time
contract before claiming the sampler targets the unmodified model.

### Correct extension to learned responses and timescale

A sensible sweep is:

1. Update the small response/timescale block using a Metropolis or HMC kernel
   targeting its **latent-marginal** conditional, holding loadings, offsets
   and noise fixed.
2. Immediately redraw the complete latent path conditional on those new
   parameters and the data.
3. Draw each feature's loading/offset block jointly from its Gaussian
   conditional, using statistics summed over all runs where the loading is shared.
4. Update noise under the existing lognormal variance prior, then apply valid
   shift, scale and rotation moves.

The ordering is part of correctness. A marginal hyperparameter update followed
by a loading update using the old path generally does not preserve the joint
posterior. [Van Dyk and Jiao](https://arxiv.org/abs/1309.3217) explain the hazards
of mixing marginal and conditional transitions. Reuse observation information
statistics across multiple hyperparameter proposals within the sweep.

Response changes can reorder shifted event times, so rebuild the ordered event
system and redraw the path at the new parameters. Gaussian-response sampling
targets the qualified state-space approximation unless a separate exact
correction is added. The current public backend still excludes state-space
posterior sampling and learned GP timescales; these are real implementation
tasks, not options that can simply be switched on.

For GPU path draws, compare parallel FFBS with conditional simulation using a
prior path plus a smoothed residual correction. The latter follows the
[simulation-smoother literature](https://doi.org/10.1093/biomet/89.3.603) and can
reuse the filter/smoother developed for MAP. It must include simulated
observation noise and preserve joint temporal covariance; drawing independent
pointwise marginals is not a path sampler. Multi-run shifts and filtered-response
shifts need the correct joint compensation, and the K=1 anchor needs a truncated
Gaussian block, not an absolute-value transformation of an unconstrained draw.

## Where variational inference fits

**Prefer structured VI over a generic diagonal guide on the current marginal
parameter vector.** A diagonal guide would still pay for dense likelihood
evaluations unless the algebra changes, and it cannot express the loading/path
scale dependence or lag multimodality well. Dense covariance over roughly 60,000
parameters is also unattractive.

A useful first family is a joint Gaussian Markov distribution for each entire
latent path, Gaussian blocks for each feature's loading and offset, and small
distributions for noise and response/timescale parameters. Keep temporal and
cross-factor covariance inside the path block. The Gaussian updates use
expected second moments such as `E[W W.T]` and `E[x_t x_t.T]`; replacing them by
products of means silently changes the optimization.

[Fast variational state-space GP learning](https://arxiv.org/abs/2007.04731)
supports natural-parameter updates, and
[Hansen et al.](https://arxiv.org/abs/2305.13188) demonstrate fast VI for related
factor models. Neither establishes calibration or speed for GP-MSRM. Our
Gaussian observation likelihood is already conjugate conditional on parameters;
the proposed approximation is chiefly to parameter/path dependence, not a need
to approximate a non-Gaussian observation likelihood.

Two reasonable product scopes are:

- **Fast exploratory fits/predictions:** structured VI or EM, with response
  parameters optimized at a point if that limitation is explicit. Point
  hyperparameters do not provide posterior uncertainty for lags or timescale.
- **Scientific uncertainty:** Gibbs as the reference; accept VI only after
  simulation-based coverage, held-out predictive checks, and lag/timescale and
  covariance comparisons across multiple seeds and lag basins.

Do not minibatch time points as though GP observations were independent.
Feature or independent-run minibatches are more natural, conditional on global
latent variables and with correct likelihood scaling. No VI implementation or
VI speedup was measured in this investigation.

Inducing-function VI is a backup if Gaussian banks remain too large:
[Álvarez et al.](https://proceedings.mlr.press/v9/alvarez10a.html) treat convolved
multi-output GPs directly. The inducing representation must retain response
operators and approximation corrections. A plain low-rank covariance
substitution is not the same objective.

[Gokcen et al. (2025)](https://users.ece.cmu.edu/~byronyu/papers/GokcenNeuralComput2025.pdf)
is the closest application-level comparison: inducing-variable and
frequency-domain approximations accelerate multigroup GP factor models, while
finite-window frequency approximations can bias delays and timescales. Our
unequal native clocks and feature masks prevent assuming that a Fourier
transform diagonalizes the observation problem. This is a lower-priority
fork than exploiting the Markov structure already available.

## Decision gates

1. **Qualify parallel filtering and smoothing.** Require density, physical
   gradient and joint posterior-moment agreement across the existing masks,
   ties, high-SNR, zero-loading, multi-run and learned-response cases. Measure
   cold setup/compilation, warm operations, peak memory and complete fits.
2. **Finish fixed-response Gibbs on that implementation.** Require exact
   conditional checks, multiple overdispersed chains and predictive/covariance
   diagnostics. Report minimum bulk/tail ESS per wall second including
   compilation and warmup, and time to R-hat below 1.01 on chosen invariants.
3. **Add learned hyperparameters with the valid collapsed ordering.** Test
   lag-order crossings, boundaries and timescale/response confounding. This is
   the main unresolved statistical risk for transferring the Gibbs gains.
4. **Then compare structured VI at matched predictive/uncertainty quality.**
   Reuse the same smoother and sufficient statistics. Promote it for a clearly
   stated approximate-inference use case if it wins.

Prioritize algorithm changes over reduced precision or hardware purchases.
The cards' compute capabilities (8.6 and 12.0) have limited ordinary FP64
throughput relative to FP32 in
[NVIDIA's table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
This investigation achieved its successful gains in float64 by changing the
dependency structure, without relying on tensor-core precision changes.

## Reproduction and limits

Synthetic benchmark example; choose the GPU by UUID from `nvidia-smi -L`, because
CUDA and `nvidia-smi` index order differed on this machine:

```sh
PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda \
  CUDA_VISIBLE_DEVICES=GPU-... XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_gp_parallel_filter.py --duration 1800 --repeats 3 \
    --output local_data/parallel-identity-1800.json
```

For learned Bateman responses add `--response bateman --learned --skip-dense`.
The dense Bateman alternative requires separate response quadrature qualification;
the table deliberately compares it only with the same state-space realization.
Use `--scalar --duration 40 --channels 8` for the old recurrence comparison.

Runs used Python 3.12.13, JAX 0.11.2 and NumPy 2.5.3. GPU processes used one
BLAS thread and CPU affinity 8–15, except the empirical 3090 check used 24–31.
CPU probes used 0–7; the fitted-point CPU check used eight BLAS threads.
The GPUs were checked
idle before the campaign and benchmark jobs on a given GPU were sequential.
Some CPU checks ran concurrently on separate affinity sets. Warm timings use
synchronized device completion and three or five repetitions, with first calls
recorded separately. Load averages and all timing repetitions are retained.

Synthetic observations are seeded workloads, not response-recovery simulations.
Empirical probes reuse the previous local loader's unconfirmed splice-volume
alignment and preprocessing assumptions and therefore address computation only.
No participant arrays are included in the report or results file. Local scripts,
logs, checks and aggregate results are under ignored
`local_data/gp-next-2026-09-22/`. No new long NUTS fit was launched.
