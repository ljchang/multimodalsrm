# Linux CUDA GPU opportunities for R-MSRM, GP-MAP and GP posterior

**Measurement date:** September 21, 2026. **Status:** research note on the development source at Git HEAD `5a49a54`; no package release, default change, or production optimization is made by this note.

This note builds on the [prior Apple MPS findings](2026-09-19-apple-mps-feasibility.md) and the [post-release Linux evaluation plan](../superpowers/plans/2026-09-21-post-release-linux-performance-evaluation.md). It answers three questions with measurements on the user's Linux workstation: where GPU acceleration could help each model, how much float64 throughput the installed NVIDIA cards actually provide, and which bottlenecks are algorithmic rather than hardware limits.

## Summary

1. **R-MSRM does not need a GPU.** A bounded fit at the recorded one-eighth empirical scale spends its time in many small per-block NumPy/SciPy operations and Python-level loops. The largest single operation, the dense reconstruction residual, is about 30 ms. Restarts already run as parallel processes across the 64 CPU cores. Keep R on CPU.
2. **GP-MSRM already runs on CUDA with no code change.** The grouped and dense likelihoods are JAX; the guarded symmetric factorization selects the CPU LAPACK callback only on the CPU platform and native Cholesky elsewhere. Installing `jax[cuda13]` and starting Python with the CUDA platform is sufficient to run the released code on a GPU in float64. GPU support in the package is therefore an installation, configuration, evidence-recording and testing task, not a port.
3. **Float64 throughput on these cards is limited by design, and the PRO 6000 is the useful one.** Both the RTX 3090 (Ampere, compute capability 8.6) and the RTX PRO 6000 Blackwell (compute capability 12.0) execute float64 at 1/64 of their float32 rate. Measured float64 matrix multiplication peaks near 0.55 TFLOPS on the 3090 and 2.1 TFLOPS on the PRO 6000, against roughly 1 TFLOPS from the 64-core CPU under the day's background load. For the model's actual objective-and-gradient at the one-eighth scale the PRO 6000 took 66 ms, the 3090 172 ms and the best CPU row 644 ms. Float32 is 30 to 80 times faster on the GPUs (and JAX's default float32 matmul on CUDA silently uses TF32) but is outside the float64 numerical contract, and the MPS note already recorded a float32 factorization failing gradient qualification.
4. **The current CPU GP path had a large avoidable cost in this environment.** The CPU LAPACK callback took 0.4 to 2.9 s for a 2,727-dimension factorization depending on OpenBLAS thread count and background load, while direct POTRF+POTRI with 16 threads took 60 ms and the 3090 took 135 ms. Thread control is the first fix for the CPU path; it needs no GPU.
5. **For full recordings the dense grouped algebra is the wall, not the device.** Cost grows with the cube of unique observation nodes times factors. A single run eight times longer than the one-eighth case at K=5 implies a 36,000-dimension dense factorization per gradient, which is tens of seconds on any device here and tens of gigabytes of memory. If the full data are eight separate runs, cost grows only linearly and runs are independent. The state-space formulation, with grouped observation updates and a parallel-in-time scan, is the candidate that changes the scaling; the plan already names it.
6. **The posterior is where a GPU pays, and it is still expensive.** One NUTS leapfrog step for the K = 3 one-eighth case cost 2.0 s on the CPU, 0.37 s on the 3090 and 0.16 s on the PRO 6000, in each case two to three times the bare gradient. Vectorizing two chains on one card gained about 1.35x throughput, not 2x. At realistic budgets (a thousand warmup and twelve hundred draws at a few hundred steps each) one chain at this scale is on the order of a day on the 3090 and half a day on the PRO 6000, versus about a week on the CPU. The ArviZ diagnostics phase took 70 to 100 s for only 15 draws over 57,757 parameters on every device and grows with draw count; it needs its own profile.
7. **The state-space backend cannot be measured at empirical scale as written.** Two Gaussian responses give 50 states per factor, so the state dimension is 150 at K = 3. The implementation stages one transition and one process-noise matrix per scalar observation, 17.6 GB for a 48,000-observation run, and reverse-mode differentiation through the scan stores per-step covariances on top: the 96 GB card requested a single 214 GB allocation for the gradient on a 36-feature reduction, and the 3090 failed earlier. At 3.6 million observations the same arrays would be terabytes. Grouping observations by unique node (909 here) reduces the transition arrays to 327 MB and the scan length by a factor of 50. Grouped observation updates, with gradient checkpointing across the scan, are therefore a prerequisite, not an optimization, for state-space at empirical scale on any device.

The rest of this note records the evidence and its limits.

## Hardware and runtime inventory

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen Threadripper PRO 3995WX, 64 cores / 128 threads, one NUMA node |
| Host memory | 503 GB |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), compute capability 12.0, PCIe 4.0 x16 |
| GPU 1 | NVIDIA GeForce RTX 3090, 24 GB (24,576 MiB), compute capability 8.6, PCIe 4.0 x16 |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Default Python | Anaconda 3.11.7 with NumPy 2.4.6 and torch 2.3.1 (CUDA 12.1, no `sm_120` kernels, so torch cannot target the Blackwell card); no JAX; below the package's Python 3.12 floor |
| Measurement environment | `uv venv --python 3.12`, package installed from the checkout with the `bayesian` and `test` extras, `jax==0.11.2`, `jaxlib==0.11.2`, `jax[cuda13]` plugin, `numpyro==0.22.0`, NumPy 2.5.3, SciPy 1.18.1 with OpenBLAS 0.3.34 |

Both GPUs were in use by another lab member's jobs during the first part of the session (about 10 GB resident on the 3090, 5 to 8 GB on the PRO 6000, intermittent utilization), and the CPU load average was between 25 and 65 from other users' work throughout. All timings below are therefore upper bounds on contended hardware, recorded with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. At the user's request, measurement first moved to the 3090 only; the full PRO 6000 series ran later once that card showed zero utilization and under 1 GB resident, with the user's authorization. The CPU rows remain the least reliable and need an idle-machine repeat.

The MPS note's lessons were followed: one frozen source tree, separate preparation, compilation and warm timings, device synchronization before stopping timers, callback and fallback behavior observed, and float64 throughout.

## Raw float64 linear algebra

`bench_linalg.py` times a square matrix product, Cholesky, LU and Cholesky-inverse (`cho_solve` against the identity, the pattern used by the native symmetric solve) with JAX on each device. Median of three warm repeats after one compile.

| Device | dtype | n | matmul | TFLOPS | Cholesky | LU | Cholesky inverse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU (XLA, all cores) | float64 | 2,700 | 41 ms | 0.95 | 57 ms | 140 ms | 115 ms |
| CPU | float64 | 4,096 | 119 ms | 1.16 | 141 ms | 442 ms | 465 ms |
| CPU | float64 | 8,192 | 1,958 ms | 0.56 | 3,086 ms | 9,064 ms | 4,690 ms |
| RTX 3090 | float64 | 2,700 | 77 ms | 0.51 | 20 ms | 51 ms | 141 ms |
| RTX 3090 | float64 | 4,096 | 246 ms | 0.56 | 55 ms | 130 ms | 417 ms |
| RTX 3090 | float64 | 8,192 | 2,032 ms | 0.54 | 432 ms | 894 ms | 3,094 ms |
| RTX PRO 6000 | float64 | 2,700 | 20 ms | 1.92 | 8.2 ms | 22 ms | 41 ms |
| RTX PRO 6000 | float64 | 4,096 | 66 ms | 2.10 | 16 ms | 46 ms | 116 ms |
| RTX PRO 6000 | float64 | 8,192 | 583 ms | 1.88 | 119 ms | 277 ms | 872 ms |
| RTX PRO 6000 | float64 | 16,384 | 4,677 ms | 1.88 | 807 ms | 1,702 ms | 6,608 ms |
| RTX 3090 | float32 | 2,700 | 1.2 ms | 32 | 1.4 ms | 8 ms | 7 ms |
| RTX 3090 | float32 | 8,192 | 30 ms | 36 | 22 ms | 50 ms | 105 ms |
| RTX PRO 6000 | float32 | 8,192 | 6.7 ms | 165 | 6.3 ms | 24 ms | 36 ms |
| RTX PRO 6000 | float32 | 16,384 | 73 ms | 121 | 35 ms | 102 ms | 253 ms |

Readings:

- In float64 the 3090 is at rough parity with the loaded CPU for the factorization sizes the grouped model produces today, and slower for matrix products. The PRO 6000 is about 3 to 4 times faster than the CPU in float64 at these sizes, sustains its rate to 16,384 dimensions, and is 3.6x the 3090 at 8,192. Neither approaches the float32 numbers, which are excluded by the numerical contract.
- The PRO 6000 float32 matrix product exceeds the card's nominal float32 rate, which shows that JAX's default float32 matmul precision on CUDA uses TF32 tensor cores. Any float32 experiment must set `jax_default_matmul_precision="highest"` or it measures a reduced-precision path.
- The CPU rows at n = 8,192 are anomalous (matmul throughput halves) and reflect the background load, not a hardware limit.
- Memory, not time, bounds single-run scaling on the 3090: a 36,000-dimension float64 matrix is 10.4 GB and the inverse path holds several. The PRO 6000's 96 GB removes that bound for a single run, at about 70 s per factorization-plus-inverse by cubic extrapolation from 16,384.

## R-MSRM: CPU profile at the recorded scale

Synthetic workload shaped like the recorded one-eighth empirical case: four participants with a 3,600-feature brain stream sampled every 1.8 s (252 samples), five participants with rating and physiology streams on 1.2 s and 1.5 s clocks, one run of about 454 s, K = 3, `latent_dt = 0.9` (505 grid points), Identity brain response, Gaussian rating lag and Gaussian physiology width and lag estimated with shared pooling. `profile_r.py` ran one restart for three alternating iterations with 8 OpenBLAS threads under cProfile.

| Quantity | Value |
| --- | --- |
| Total fit time (3 iterations) | 2.65 s, about 0.7 s per alternating iteration after 0.5 s of preprocessing |
| `complete_objective` | 35 calls, 1.06 s total, 30 ms each: dense residual over 3.6 million observed values |
| Gaussian operator construction with derivatives | 250 calls, 0.58 s total, 2.3 ms each |
| `make_blocks` + `fit_preprocessing` | 0.46 s once |
| `solve_latents` (sparse normal equations, SuperLU) | 4 calls, 0.19 s total, 21 ms in the solve itself |
| Loading solves, gradient terms, scale balance | under 0.2 s combined |

A repeat at twice the recording length (1,009 grid points, 7.3 million observed values, load average 47 to 58) took 6.6 s for the same three iterations: 1.7 s of one-time block preparation, then about 1.6 s per iteration with the reconstruction residual at 40 ms per call and the Gaussian operator construction at 5 ms per call. Growth is close to linear in recording length once the background load is discounted.

Extrapolation: a 100-iteration restart costs on the order of one to two minutes at this scale, and the per-iteration cost scales roughly linearly with recording length because every term is per-observation-row or banded. Eight times the recording with 30 participants implies roughly 10 to 30 minutes per restart, and restarts are independent processes. A GPU would accelerate only the 30 ms dense residual (a memory-bound weighted GEMM) and the loading solves; the operator construction, sparse assembly, L-BFGS-B kernel loop and objective bookkeeping are Python and SciPy sparse code with no device path. The prior MPS note measured the same loading update as 3 ms on CPU. **Recommendation: no GPU work for R.** The worthwhile CPU items, if profiling of the real data confirms them, are (a) reusing the reconstruction residual between `complete_objective` calls inside one alternating iteration, since the objective is evaluated up to a dozen times per iteration on unchanged loadings, and (b) a banded Cholesky (`solveh_banded`) in place of SuperLU for the latent normal equations. Neither changes the objective.

## GP-MSRM grouped likelihood: where the time goes

Same synthetic data, `linear_algebra="grouped"`, K = 3, 909 unique (modality, time) nodes, factorization dimension 2,727, 57,757 parameters, 3.64 million observed scalars. `bench_gp_parts.py` times each stage under `jax.jit` with warm medians of three.

| Stage | CPU, 64 OpenBLAS threads, load 36, F = 360 | CPU, 16 OpenBLAS threads, load 50 to 65, F = 3,600 | RTX 3090 float64, F = 3,600 | RTX PRO 6000 float64, F = 3,600 |
| --- | ---: | ---: | ---: | ---: |
| Covariance over unique pairs and gathers (`operands`) | 20 ms | 25 ms | 0.9 ms | 0.4 ms |
| Block sufficient statistics (`segment_sum` over observations) | 17 ms | 146 ms | 2.2 ms | 0.7 ms |
| Guarded symmetric solve alone, dimension 2,727 | 2,890 ms | 2,523 ms | 153 ms | 42 ms |
| `gaussian_terms` primal (guarded Cholesky path) | 2,727 ms | 3,223 ms | 145 ms | 42 ms |
| LU reference path | 410 ms | 118 ms | 52 ms | 23 ms |
| Negative log likelihood forward | 154 ms | 227 ms | 59 ms | 26 ms |
| Objective and full gradient | 3,366 ms (8,167 ms at F = 3,600) | 3,720 ms | 172 ms | 70 ms |
| Log prior | 0.06 ms | 0.3 ms | 0.1 ms | 0.2 ms |

Two findings matter more than the device comparison:

- **The CPU callback path was pathological here.** `symmetric_solve` on the CPU platform calls SciPy's POTRF and POTRI through `jax.pure_callback`. Direct POTRF+POTRI at this size takes 420 ms with one OpenBLAS thread and 60 ms with 16 threads (`potri_threads.py`). Through the callback in a JAX process the same work took 655 ms with one thread, 386 ms with 16 threads at load 42, 2.5 s with 16 threads at load 60, and 2.9 s with the default 64 threads at load 36. OpenBLAS threads, XLA's 128-thread pool and the other users' processes oversubscribe the cores, and the callback's serialization with XLA's thread pool amplifies contention. The CI convention of one BLAS thread is safe but slow; 8 to 16 threads is the right region on this machine when it is otherwise idle, and it must be set before Python starts. An uncontended CPU measurement is still needed. The Mac note's 116 ms whole-gradient timing was obtained with one thread on an idle machine, which is why the Linux CPU number looks worse than the Mac number.
- **The observation-count part is cheap on the GPU and not on XLA CPU.** Going from 360 to 3,600 brain features (0.37 to 3.64 million observations) added 4.8 s to the CPU gradient and 30 ms to the 3090 gradient. XLA's CPU scatter-add and its transpose are effectively single-threaded; on the GPU they are memory-bound and negligible. This is the one place where the GPU wins by a large factor even in float64, and it grows with feature count.

Gradient agreement: the objective and gradient fingerprints agreed between CPU and GPU at the float64 level in these runs (differences reported in `bench_gp.jsonl`). This is algebraic agreement at one parameter vector, not convergence qualification.

## GP objective-and-gradient, NUTS and state-space timings

`bench_gp.py` builds the synthetic problem, evaluates the jitted objective-and-gradient at one seeded parameter vector (warm median of five), and optionally runs a short NUTS (15 warmup, 15 draws, diagonal mass matrix, tree depth at most 5) through the package's own `sample` entry point so that initialization, adaptation, draw conversion, log-likelihood and diagnostics phases are all exercised.

### Objective and gradient

| Device | K | Recording scale | Dimension | Parameters | Observations | Prepare | Compile + first | Warm gradient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3090 | 3 | 1/8 | 2,727 | 57,757 | 3.64 M | 8.9 s | 4.9 s | 196 ms |
| RTX 3090 | 5 | 1/8 | 4,545 | 86,627 | 3.64 M | 7.5 s | 5.4 s | 552 ms |
| RTX 3090 | 5 | 1/4 as one run | 9,205 | 86,627 | 7.28 M | 16.4 s | 13.6 s | 4,653 ms |
| RTX PRO 6000 | 3 | 1/8 | 2,727 | 57,757 | 3.64 M | 7.4 s | 5.0 s | 66 ms |
| RTX PRO 6000 | 5 | 1/8 | 4,545 | 86,627 | 3.64 M | 6.2 s | 6.6 s | 204 ms |
| RTX PRO 6000 | 5 | 1/4 as one run | 9,205 | 86,627 | 7.28 M | 15.6 s | 10.3 s | 1,218 ms |
| RTX PRO 6000 | 5 | 1/2 as one run | 18,530 | 86,627 | 14.56 M | 45.4 s | 25.3 s | 9,308 ms |
| CPU, default threads, load 25 to 35 | 3 | 1/8 | 2,727 | 57,757 | 3.64 M | 9.0 s | 12.7 s | 8,167 ms |
| CPU, 16 threads, load 43 | 3 | 1/8 | 2,727 | 57,757 | 3.64 M | 4.7 s | 3.3 s | 3,201 ms |
| CPU, 16 threads, load 27 (NUTS run) | 3 | 1/8 | 2,727 | 57,757 | 3.64 M | 7.4 s | 5.4 s | 644 ms |
| CPU, 16 threads, load 51 | 5 | 1/8 | 4,545 | 86,627 | 3.64 M | 6.9 s | 13.7 s | 8,420 ms |
| CPU, 16 threads, load 32 | 5 | 1/4 as one run | 9,205 | 86,627 | 7.28 M | 15.7 s | 41.1 s | 33,070 ms |

The objectives agreed between CPU and 3090 to 16 significant digits in all three cases (for example 5323210.5539216 at K = 3) and the sampled gradient entries agreed to about 13 digits. The two K = 3 CPU rows differ by 5x with identical settings and only the background load changed (43 versus 27), which is the clearest evidence that CPU numbers on this shared machine need an idle-machine repeat. Against the better CPU row the 3090 was 3.3x faster at K = 3 and the PRO 6000 9.8x; against the contended rows the 3090 was 16x at K = 3, 15x at K = 5 for the one-eighth case, and 7x at the doubled length. The PRO 6000 was 2.6 to 3.8x faster than the 3090 across the matched cases. On the PRO 6000 each doubling of recording length as one run raised the K = 5 gradient cost 6.0x and then 7.6x, so a full recording as one run extrapolates to roughly 70 s per gradient there, and the 96 GB card held the 18,530-dimension case that the 3090 could not. Doubling the recording length as one run raised the K = 5 gradient cost 8.4x on the GPU and 3.9x on the CPU, consistent with cubic scaling on the GPU and with the CPU still being dominated by the contended callback at the smaller size. Against an idle, well-threaded CPU the expected ratio is 2 to 4x at these sizes, from the raw factorization sweep. Peak GPU memory was not captured by the script (the allocator statistic returned nothing with preallocation disabled); capture it with `nvidia-smi` sampling in the next campaign.

### Short NUTS, K = 3, one-eighth scale

| Device | Chains, method | Leapfrog steps | Warmup + sampling | Per leapfrog | Bare gradient in the same process | Initialization | Log-likelihood of draws | Diagnostics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3090 | 1, sequential | 465 | 173 s | 0.37 s | 0.17 s | 15 s | 3.4 s | 72 s |
| RTX 3090 | 2, vectorized | 930 | 525 s | 0.56 s per step pair, 0.28 s per chain-step | 0.17 s | 16 s | 7.1 s | 100 s |
| CPU, 16 threads, load 27 | 1, sequential | 465 | 950 s | 2.04 s | 0.64 s | 27 s | 4.6 s | 70 s |
| RTX PRO 6000 | 1, sequential | 465 | 75 s | 0.16 s | 0.065 s | 26 s | 3.4 s | 72 s |
| RTX PRO 6000 | 2, vectorized | 930 | 218 s | 0.23 s per step pair, 0.12 s per chain-step | 0.066 s | 21 s | 7.0 s | 99 s |

The same seed produced the same 465-step trajectory length on all three devices. Per leapfrog step the PRO 6000 was 2.3x the 3090 and 13x the CPU, and vectorizing two chains gave 1.37x throughput on either card. At 15 draws the diagnostics phase already exceeds warmup plus sampling on the PRO 6000. The per-leapfrog figures include NUTS kernel compilation amortized over a short run, so they overstate the steady-state cost; the ratio of leapfrog cost to bare gradient (2.2x on the GPU, 3.2x on the CPU) also includes the diagonal mass-matrix adaptation and NumPyro's tree building, which are serial small operations that suit the CPU less than the dense algebra does. The 3090 was 5.5x faster than the CPU per leapfrog step in this pairing. Sampling ran with `XLA_PYTHON_CLIENT_PREALLOCATE=false` on a card shared with other jobs. Diagnostics run on the CPU through ArviZ over every parameter, cost about 70 s on either device at 15 draws, and are the second-largest phase.

### State-space backend

| Device | K | Brain features | Observations | State dimension | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 3090 | 3 | 36 | 48,438 | 150 | Out of memory: XLA staged 42.6 GB of scan arguments; allocations of 8 to 20 GB failed |
| RTX PRO 6000 (96 GB) | 3 | 36 | 48,438 | 150 | Out of memory: the objective-and-gradient executable requested a single 213.6 GB allocation for the reverse-mode residuals of the scan |
| CPU | 3 | 36 | 48,438 | 150 | Not run: the same 17.6 GB staging fits in host memory but would occupy the shared machine for the full 25-minute cap to repeat a memory finding |

`ResponseStateSpace.prepare` gives 50 states per factor for this response configuration (two Gaussian responses on rational or Laguerre banks plus the Matérn pair). `StateSpaceSystem.prepare` materializes `(observations, 150, 150)` transition and process-covariance arrays, 17.6 GB at K = 3 and 48.8 GB at K = 5 for this reduced run, and reverse-mode differentiation through `lax.scan` additionally stores the per-step predicted and filtered covariances, which is what the 214 GB request on the PRO 6000 reflects. Checkpointed differentiation (`jax.checkpoint` over scan chunks) bounds that residual memory, as the plan's section 7 already requires. At the empirical 3.6 million observations the same arrays would be 1.3 TB. With observations grouped to the 909 unique nodes the arrays are 327 MB and 909 MB. This is the plan's section 7 "exact grouped state-space observation updates" candidate, and it is required before any state-space timing at empirical scale is meaningful, on GPU or CPU. A parallel associative scan would additionally replace 909 sequential 150-dimension updates with a logarithmic-depth combination, which is where a GPU would help this backend.

## Scaling to the full recordings

Let N be the number of unique observation nodes per run and K the factor count. The grouped algebra forms and factorizes an NK-square dense matrix per run per gradient, and the guarded path also materializes its inverse for the implicit derivative. Costs scale as (NK)^3 in time and (NK)^2 in memory per matrix.

| Case | Nodes per run | Dimension at K = 5 | Dense matrix | Factorization + inverse, RTX 3090 float64 | Same, RTX PRO 6000 float64 | Same on CPU (16 threads, uncontended, extrapolated) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One-eighth (measured shape) | 909 | 4,545 | 165 MB | about 0.5 s | 0.13 s | about 0.4 s |
| One-quarter as one run | 1,818 | 9,090 | 660 MB | about 4 s | about 1.1 s | about 3 s |
| One-half as one run | 3,636 | 18,180 | 2.6 GB | about 30 s (memory-limited) | about 9 s | about 25 s |
| Full recording as one run | 7,272 | 36,360 | 10.6 GB | does not fit | about 70 s | about 3 min |
| Full recording as eight runs | 8 × 909 | 8 × 4,545 | 8 × 165 MB | about 4 s, or 0.5 s per run in parallel | about 1 s | about 3 s |

These extrapolations use the measured n = 2,700 to 16,384 cubic trend (the PRO 6000 column interpolates its measured 4,096, 8,192 and 16,384 rows) and should be replaced by measurements. Their point is structural: with one long run, no device in this machine makes a NUTS chain feasible at full length (a thousand warmup and twelve hundred draws at up to 1,023 leapfrog steps each would need on the order of a year of gradients). With runs at their natural boundaries, the full data costs about eight times the one-eighth case, runs are independent and can be spread over devices, and the GPU speedup of the per-run gradient carries over.

Whether the user's recordings are one run or several is the single most important fact to verify in the plan's Stage A inventory before choosing implementation work. The plan's Stage D duration ladder should be built at run boundaries, as it already says.

## What "GPU support" would mean in the package

For GP-MSRM the code already dispatches by platform. The remaining work is contractual:

1. **Installation extra and documentation.** Add an optional `bayesian-cuda` extra (or documented `jax[cuda13]` instructions) with the same pinned JAX version, plus a getting-started section on `JAX_PLATFORMS`, `XLA_PYTHON_CLIENT_PREALLOCATE`, thread variables and `CUDA_VISIBLE_DEVICES`. Keep CPU-only installation the default.
2. **Evidence recording.** `execution_info` already records backend and devices for sampling; MAP diagnostics and fitted archives should record the same, plus dtype, so a fit made on a GPU is identifiable. Warmup checkpoints already bind to device identity.
3. **Numerical qualification on CUDA.** Run the existing factorization, analytic-score, multifactor and posterior smoke tests on the CUDA platform and compare objectives, physical gradients and MAP endpoints against CPU at identical parameters, as the plan's Stage C requires. Pin `jax_default_matmul_precision="highest"` in the test harness; it only affects float32 but prevents a silent TF32 path if any float32 array ever appears.
4. **Thread and device configuration guard.** Emit a warning when the CPU platform runs with more BLAS threads than physical cores or when `pure_callback` is on the hot path with an oversubscribed pool; document 8 to 16 threads as the tested CPU setting on this hardware.
5. **Chain scheduling.** Keep `chain_method="sequential"` as default; qualify `vectorized` on one GPU (padded trajectories waste work when tree depths differ across chains) and document running separate processes per device for independent chains, each with its own seed and archive.
6. **Memory guard for the inverse path.** `symmetric_solve` materializes the full inverse. On a 24 GB card the practical dimension ceiling is about 15,000 to 18,000 in float64; document it and keep the LU reference path as the fallback. A reverse-mode rule that uses solves against the cotangent instead of the explicit inverse would remove that ceiling and reduce work; it is an algebraic change to the derivative rule, not to the density, and would need the existing finite-difference and higher-order derivative tests.

For R-MSRM, no GPU backend. A NumPy array-API port for the two dense operations would touch a third of the fitting code for a few percent of runtime.

## Candidate ranking for the plan's Stage C

| Track | Candidate | Evidence here | Recommendation |
| --- | --- | --- | --- |
| R-MSRM | GPU batched loading or latent solves | Largest dense op 30 ms; Python and sparse code dominate | Reject |
| R-MSRM | CPU residual reuse, banded latent solver, restart parallelism | Profile above | Proceed on CPU after real-data profile |
| GP-MAP | Released grouped path on CUDA | Runs unmodified with 13-digit gradient agreement; gradient 66 ms on the PRO 6000, 172 ms on the 3090, 644 ms to 8.2 s on the contended CPU; 96 GB holds an 18,530-dimension case | Proceed: qualify and document; prefer the PRO 6000 |
| GP-MAP | CPU thread policy fix | 2.9 s to 0.4 s from thread count alone | Proceed first; no GPU needed |
| GP-MAP | Reverse-mode rule without explicit inverse | Inverse path is 60 percent of GPU gradient time and the memory ceiling | Prototype after qualification |
| GP-MAP | State-space with grouped node updates, checkpointed differentiation and associative scan | Per-observation formulation needs 214 GB for the gradient of a 36-feature reduction and fails on both cards; grouped updates cut arrays and scan length by the feature count | Prototype under the plan's section 7 gates; required before any state-space timing |
| GP posterior | NUTS on GPU, one chain per device, CPU chains in parallel | Per leapfrog: 0.16 s PRO 6000, 0.37 s 3090, 2.0 s CPU; vectorized chains gain 1.35x; diagnostics phase is device-independent CPU work | Proceed once MAP is qualified; profile diagnostics separately |
| GP posterior | float32 or mixed precision | Excluded by contract; MPS evidence of gradient failure | Separate labeled study only |

## Evidence retained locally

Scripts and raw outputs are retained under ignored `local_data/linux-performance-v1/2026-09-21-gpu-opportunities/`: `bench_linalg.py` with `bench_linalg.log` (CPU and partial PRO 6000 rows), `bench_linalg_3090.json` and `bench_linalg_pro6000.json`; `synth.py`; `bench_gp.py` with `bench_gp.jsonl` (every objective, gradient, NUTS and error row); `bench_gp_parts.py`; `potri_threads.py`; `callback_probe.py`; `profile_r.py` with `profile_r_F3600.log`; `run_gp.sh`, `sequence.sh` with `sequence.log` (3090 and CPU series, including the scale-2 R profile) and `sequence_blackwell.sh` with `sequence_blackwell.log`. No empirical data were used; all inputs are synthetic and seeded.

## Limits

- Synthetic data of the recorded shape, not the empirical arrays; feature count, node count and parameter count were matched from the MPS note's description, not from the actual inventory.
- Contended hardware throughout; the CPU rows especially vary by a factor of two to five with background load. The GPU rows were taken with other users' contexts resident on the cards and are consistent across repeats, but not idle-machine measurements.
- Single parameter vector timings; no optimizer trajectories, no convergence qualification, no posterior diagnostics beyond the short NUTS timing run.
- One run per dataset; the multi-run parallel case is extrapolated, not measured.
