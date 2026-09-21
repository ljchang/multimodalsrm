# Prior Apple MPS feasibility findings

**Experiment date:** September 19, 2026. **Documentation added:** September 21, 2026.

The tested Apple MPS substitutions were slower than the optimized CPU float64 paths. GP-MSRM's MPS float32 factorization also changed physical gradients enough to fail the existing numerical qualification. Retain CPU float64 for these tested Mac workloads.

**Planning decision:** Do not repeat these MPS substitutions in the [post-release Linux evaluation](../superpowers/plans/2026-09-21-post-release-linux-performance-evaluation.md). Investigate the available Linux CPU and NVIDIA CUDA hardware. Reconsider MPS only through a separate proposal identifying a concrete new capability or algorithm and how it addresses the recorded precision or performance limitation.

These are historical saved-state operation measurements. They do not establish full-fit throughput, a fully GPU-resident implementation, current performance under newer runtimes, or the suitability of NVIDIA CUDA.

## Baseline and environment

The initial synthetic GPU campaign used an older maintained checkout that omitted newer uncommitted CPU optimizations. Its suggestion to add CPU batching duplicated work already completed. That earlier campaign is superseded for evaluating incremental improvements to the optimized empirical paths.

The corrected campaign froze **123 production Python files** from the empirical worktree at Git HEAD `a97cf693d5df33c3feb68c3e221c047c236686c0` **plus its recorded uncommitted changes**. The file-hash manifest identifies the baseline; that Git commit alone cannot reproduce it. It included:

- R CPU passes 1–3: batched loading updates, feature contraction before latent assembly, observation/operator caching, kernel sufficient statistics, analytic Gaussian derivatives, and parallel restart/CV support. The timed operations did not exercise parallel restarts.
- GP CPU passes 1–2: grouped sufficient statistics, analytic gradients, and the guarded symmetric factorization using CPU LAPACK POTRF/POTRI.

Hardware/runtime: Apple M5 Max, macOS 26.5 arm64, PyTorch 2.14.0, NumPy 2.5.3 with Apple Accelerate, and JAX 0.6.2. A direct hardware probe confirmed working MPS float32 operations and rejection of float64 tensors. This records the tested environment, not a claim about every future Apple runtime.

One CPU thread was requested through BLAS/OpenMP/Accelerate settings, threadpoolctl, and PyTorch; actual internal Accelerate thread use was not independently verified. MPS fast math and fallback were disabled. The frozen CPU snapshot passed **52 scoped regression tests in 18.77 seconds**. These historical checks were not rerun when this note was added.

## R-MSRM loading update

The saved model used K=3, 19 participant–modality pairs, 252 timestamps per block, and approximately 3,600 brain features per participant. Feature coefficients were shared within each observed block. The comparison preserved feature-count-normalized penalties and the original pair count.

Data and coefficients stayed cached on MPS; changing designs were formed on CPU and transferred, and results returned to CPU. Timings include those transfers and are medians of nine shuffled/interleaved warm updates.

| Complete loading update | Median time | Time / CPU | Relative loading L2 difference |
| --- | ---: | ---: | ---: |
| Optimized CPU float64 | 2.994 ms | 1.00 | 0 |
| MPS float32, all pairs | 12.155 ms | 4.06 | 2.48e-7 |
| MPS float32, brain pairs only | 3.907 ms | 1.30 | 2.45e-7 |

Per-block reconstruction RMSE differences were below 1.8e-7 at this saved latent state. The empirical case did not exercise feature-specific masks or pair-scaled penalties. This was not a complete alternating fit, held-out prediction comparison, or recovery experiment.

## GP-MSRM likelihood and physical gradient

Only the symmetric factorization was replaced. CPU float64 formed `S = K + D^-1`; an MPS float32 callback computed the Cholesky inverse, solve, and log determinant. The remaining objective and analytic gradient computation stayed on CPU float64. Casting returned values to float64 did not restore precision lost in the factorization.

The learned/fixed response models had approximately 72,000 parameters, 3.68/3.96 million observed scalar values, and factorization dimensions 2,520/2,700 respectively. Their different supports were not used to compare model quality. Timings are medians of seven warm evaluations per backend, measured CPU then MPS sequentially with synchronization.

| Saved response configuration | CPU time | MPS time | Time / CPU | CPU projected gradient | MPS projected gradient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned | 115.98 ms | 202.54 ms | 1.75 | 0.0003487 | 0.0956015 |
| Fixed | 136.14 ms | 240.98 ms | 1.77 | 0.0006967 | 0.1671628 |

The unchanged projected-gradient threshold was **0.001**. Both CPU reevaluations passed; both MPS substitutions failed. Maximum absolute gradient differences were approximately **0.0956** and **0.1672**, while relative objective differences were only **7.18e-11** and **5.94e-10**. Both MPS arms also failed the strict gradient comparison (`atol=1e-6`, `rtol=1e-7`). Objective agreement alone would have missed the numerical failure.

Each GP arm recorded **eight successful MPS callbacks and zero failed callbacks**. The harness explicitly checked callback execution because a failure could otherwise produce NaNs and trigger the guarded CPU LU fallback. A fallback result could not count as successful GPU parity. No jitter, priors, masks, scaling, or qualification thresholds were changed.

This comparison changed both precision and backend arithmetic, so it does not isolate the cause to float32 alone. It did not run a full optimizer trajectory or posterior sampler. Uncontrolled background activity and the sequential GP timing design limit timing generalization.

## Lessons for the Linux campaign

1. Freeze and hash the actual optimized baseline, including relevant dirty changes. An older branch can make an already completed optimization appear new.
2. Measure preparation, compilation, transfers, numerical operations, and whole fits separately. GPU residency and callback/fallback behavior must be observed.
3. Compare complete physical gradients and convergence qualification, not only objective values or predictions.
4. Begin with the released float64 contract. Mixed precision needs its own qualification rather than a silent backend switch.
5. Preserve failed and resource-stopped cases. These historical results do not imply that all GPUs or all larger workloads are unsuitable.

The Linux plan should evaluate R fitting, GP-MAP, and GP posterior sampling independently, using the intended PyPI release as the new baseline. The old Mac timing ratios are not Linux performance predictions.

## Evidence retained locally

The corrected campaign is retained under `shared-response-models/local_data/gpu-optimized-baseline-20260919/`. The local report, scalar summary, raw operation results, frozen sources, callback-guarded probes, and provenance were inspected when preparing this note. Source/archive immutability and the 52-test result are recorded by that campaign's audit; no experiment was rerun for publication.

This note contains a selected aggregate summary. Raw result JSON, local paths/provenance details, fitted archives, research arrays, and source snapshots remain local. SHA-256 identifiers allow the retained evidence to be matched later:

| Local artifact | SHA-256 |
| --- | --- |
| `REPORT.md` | `aab19ecf0c517a64fd8f90bf447661d8ac49f8b1a9f9e48016043b93cb6e23cb` |
| `summary.json` | `c7f5c92571fceed82236ef0a30044408a2d37709b796e1e71070e804ddd63d53` |
| `provenance.json` | `828919026f8c9a52f48533e473091267a5619398c324498e8964e2c7efb17ea9` |
| `loading.json` | `5e32743694ba25f0d7ec1ce1f172575896f2e494377079ea66e971bfb983f87c` |
| `gp_learned.json` | `259479090cfd9f808ebac5a7b3846ca5fea3b3be8057e0714d7f01df79a8d1f4` |
| `gp_fixed.json` | `8b25fa2a2f6ee203924c6c974a284dbd11b9522320ae76637c351c30daab98f4` |
