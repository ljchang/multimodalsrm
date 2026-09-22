---
search:
  exclude: true
---

# Post-release Linux performance evaluation plan

> **For agentic workers:** Use `superpowers:executing-plans` when the user resumes this evaluation after the release prerequisite is satisfied. This draft authorizes documentation only. Do not start experiments, optimize implementations, access the workstation, or publish a package from this plan alone.

**Status:** Draft, September 21, 2026. Execution deferred at the user's request until the intended new package version is published on PyPI.

**Goal:** Determine the best CPU/GPU optimization strategy for both regularized R-MSRM and GP-MSRM on the user's Linux workstation, including the feasibility of full Bayesian inference for longer recordings.

**Architecture:** Preserve the released models and compare measured computational alternatives against that frozen baseline. Evaluate R fitting, GP-MAP, and GP posterior sampling as separate tracks; select implementation work from the Linux results. Grouped state-space inference and parallel filtering are candidates, with their numerical and performance value still to be established for this workload.

**Tech stack:** Released `multimodalsrm`; its qualified NumPy/SciPy and optional JAX/NumPyro runtime; Linux CPU and NVIDIA CUDA. Candidate R acceleration libraries are selected after profiling. Keep CPU-only R installation supported.

**Spec:** The user-approved scope is recorded below. This document is an evaluation roadmap; detailed implementation plans follow the measured decision for each model.

## 1. Agreed scope and sequencing

- Publish the intended R/GP package version through the existing release process first. GPU work is not a new release requirement.
- Resume exploration directly on the Linux workstation after verifying the published package installs and runs there.
- Cover R-MSRM regularized fitting, GP-MSRM MAP fitting, and GP-MSRM posterior inference. Keep results and recommendations separate for each.
- Target approximately 30 participants with more than an hour of recording per participant. The user describes the current test as roughly one-eighth of the full data per participant. Verify actual duration, runs, rates, features, and masks during inventory.
- Use K=5 as the principal multi-component workload, with K=1 and K=3 correctness controls. Preserve the general-K API; neither K=5 qualification nor a hardware benchmark establishes convergence at every K.
- Retain one common latent response per run, individual participant mappings, and existing modality response semantics. R uses its existing regularized objective; GP uses its existing priors and likelihood.
- Preserve native timestamps, missing-data masks, training-only preprocessing, protected splits, target exclusion, and participant-isolated transforms.
- Keep empirical arrays, fitted models, checkpoints, and research payloads local. Publish code and reviewed scalar summaries only.
- Read the [prior Apple MPS findings](../../performance/2026-09-19-apple-mps-feasibility.md) before choosing candidates. Repeating the tested MPS substitutions is out of scope; use Linux CPU/CUDA measurements for this campaign. Reopening MPS requires a concrete new capability or algorithm and a separate proposal.
- This draft creates no code changes, benchmark results, remote jobs, schedules, release actions, or default changes.

## 2. Hardware target and unresolved inventory

User-reported workstation: Linux, 64 CPU cores, approximately 512 GB system RAM, an NVIDIA RTX PRO 6000, and an RTX 3090.

The Blackwell RTX PRO 6000 specification lists 96 GB VRAM and the RTX 3090 lists 24 GB. Verify exact model/edition and available memory on the machine. Do not infer the installed card's memory solely from its abbreviated name. System RAM and each GPU's VRAM are separate budgets; the GPUs are not assumed to provide one pooled allocation.

Initial scheduling candidates:

| Resource | Evaluation role |
| --- | --- |
| CPU and host RAM | Float64 reference, CPU performance baseline, preprocessing, independent restarts/folds/chains |
| RTX PRO 6000 | Primary GPU profiling and larger GPU workloads |
| RTX 3090 | Independent smaller workloads or complete chains whose memory fits |

Start with each GPU isolated in its own process. Evaluate concurrent use after individual measurements; avoid coupling the faster device to the slower one without evidence that synchronization helps. CPU worker count, BLAS threads, JAX threads, and GPU allocations must be recorded and controlled. Sixty-four cores do not imply that 64 workers improve a single fit.

## 3. Repository and release boundary

The future implementation home is this `multimodalsrm` repository after its model handoff. Use [model integration](../../model-integration.md) and [releasing](../../releasing.md) for the release prerequisite. Recheck their live state when execution resumes; this draft does not certify an upload.

The development source reference for this discussion is `shared-response-models` merge commit `67128c7a562cab67499bdc0604357731a628aa9e` (PR #18), whose tested feature head was `dd009e519919d38e978d54de9eedc1b1c8b14a9f`. This is provenance, not a substitute for the actual released version and artifact hashes.

At evaluation time, pin the released package and freeze its source and runtime. Ensure the baseline includes the CPU optimizations actually shipped in that release. Candidate changes use isolated source snapshots with exact commits and any dirty patches hashed. Do not compare a new GPU prototype against an obsolete CPU implementation.

Current source landmarks, to be mapped through the completed extraction manifest:

| Area | Development source beneath `src/personalized_srm/multimodal/` |
| --- | --- |
| R fitting and numerical checks | `estimator.py`, `optimization.py`, `kernel_optimization.py`, `operators.py`, and the actual fitting dependency closure |
| GP grouped likelihood and gradients | `bayesian/grouped.py`, `multifactor.py`, `grouped_score.py`, `multifactor_score.py`, `factorization.py` |
| GP state-space likelihood and response realizations | `bayesian/state_space.py`, `state_space_responses.py`, `state_space_parameters.py`, `state_space_compact.py`, `state_space_delays.py` |
| GP posterior and persistence | `bayesian/fitting.py`, `gp_hyperparameters.py`, `trajectories.py`, `warmup_checkpoint.py`, posterior archive modules |

These are inspection landmarks. The future optimization files and public interfaces are chosen after profiling and verification of the released layout.

## 4. Stage A — establish the released Linux baseline

- [ ] Verify the intended PyPI version exists; record version, wheel/sdist hashes, source tag/commit, and package provenance.
- [ ] Install the wheel in a clean Linux environment outside a source checkout. Verify imports resolve to that installation and run small released R and GP workflows, including persistence.
- [ ] Inventory CPU model, physical/logical core counts, NUMA layout, RAM, GPU names/UUIDs/VRAM, driver, CUDA compatibility, storage, and background device use.
- [ ] Record Python, NumPy, SciPy, BLAS, JAX, JAXlib, CUDA plugin, NumPyro, and diagnostics versions. Preserve the release-qualified numerical versions while adding compatible CUDA components. Treat a necessary runtime upgrade as a separate qualification variable.
- [ ] Confirm device placement and float64 operation with a small workload. Record compilation and actual completion, including device synchronization.
- [ ] Inventory the real observations: participants, runs and durations, modality clocks, feature counts, masks, number of unique response/time events, sampled parameter count, and proposed state dimension.

**Deliverable:** A reproducible environment and workload manifest, successful installed-package smoke results, and resource estimates. Do not begin timing a full empirical fit when the small workflow or numerical checks fail.

## 5. Stage B — profile both released models

Use the existing one-eighth case as an initial workload after confirming what it contains. Profile the same values, timestamps, masks, parameters, and seeds across competing implementations. Freeze preprocessing and per-case support before comparing arms.

| Track | Measure separately |
| --- | --- |
| R-MSRM | Preparation and response operators; latent updates; loading updates; response optimization; objective/stationarity checks; reconstruction/readout; complete multistart fit |
| GP-MAP | Preparation/compilation; likelihood and full gradient; factorization or filtering; conditioning; optimizer evaluations and convergence; complete multistart fit |
| GP posterior | MAP initialization; likelihood/potential gradients; warmup; retained sampling; diagnostics; posterior queries and joint trajectories; archive/checkpoint I/O |

- [ ] Measure cold and warm operations separately; synchronize GPU work before stopping timers.
- [ ] Count host/device transfers and CPU callbacks/fallbacks in the numerical path. Device availability alone is not evidence that the expensive work ran there.
- [ ] Measure host memory across the complete worker process tree and GPU peak memory per device. Include compiler/runtime and concurrent-chain costs.
- [ ] Establish CPU scheduling baselines with a small prespecified worker/thread sweep. Use one outer parallelization level for restarts or folds and prevent nested oversubscription.
- [ ] Retain failures, timeouts, out-of-memory cases, compilation failures, and unsupported configurations alongside completed cases.

**Deliverable:** A phase-level cost report identifying the dominant bottleneck for each track. A faster matrix operation alone does not qualify a faster complete fit.

## 6. Stage C — bounded candidate evaluation

Choose the smallest prototypes that address the measured bottlenecks. Do not implement every row automatically.

| Track | Candidates | What decides whether to proceed |
| --- | --- | --- |
| R-MSRM | Existing CPU batching/caching and parallel restarts; GPU batched loading solves; GPU latent normal-equation/operator work; persistent device data; response-gradient acceleration | Preserved objective and stationarity, qualified full-fit speed, transfer overhead, memory, implementation/dependency cost |
| GP-MAP | Released grouped CPU path; native JAX grouped GPU path; sequential state-space; exact grouped state-space observation updates; parallel filtering across time | Matched likelihood/gradient checks, physical-gradient convergence, response qualification, complete fit and query cost |
| GP posterior | Released grouped posterior on CPU/GPU; sequential versus vectorized independent chains; separately scheduled devices; state-space posterior feasibility | Raw and invariant diagnostics, effective samples per second, memory, initialization/warmup cost, coherent trajectory support |

For R, compare a narrow GPU numerical backend with a broader accelerator port only if profiling warrants it. Backend selection must account for sparse/native-time operators, transfer frequency, and optional dependencies. Changing regularization, reducing feature count, resampling observations, or weakening solver tolerances is a separate model/workload change.

For GP, retain grouped inference as a serious GPU candidate. State-space is a linear-algebra choice; MAP versus posterior is a separate estimation choice. A performant GP-MAP path need not be the best posterior path.

Use float64 as the initial numerical contract. Any float32 or mixed-precision proposal gets a separately labeled experiment with full gradient, convergence, and posterior checks; objective agreement alone is insufficient. Do not infer workload speed from advertised AI throughput.

**Deliverable:** A short list of measured candidates, with rejected candidates and reasons retained. This selection precedes production optimization work.

## 7. State-space posterior decision, if supported by profiling

The current source backend is MAP-only. Mark state-space posterior cells as unsupported until a separately scoped prototype has passed its correctness checks; do not obtain benchmark numbers merely by bypassing its admission guard.

The candidate design is:

- Marginalize the shared latent process with the Kalman likelihood; reuse parameter priors, transformations/Jacobians, and NUTS diagnostics.
- Combine observations of the same response functional/time using exact sufficient statistics, retaining likelihood normalization and residual terms. Recompute parameter-dependent statistics for every proposal. Preserve masks and exact clocks without time binning.
- Keep one shared process per run across participants. Integrate all participant observations into that process; splitting participants into independent latent fits changes the model.
- Qualify fixed-timescale Identity first, then learned shared timescale and supported responses. Exercise delay-order crossings, simultaneous events, irregular clocks, small intervals, and long gaps.
- Compare sequential grouped filtering with a parallel associative formulation. Verify likelihoods and gradients, including derivatives through response/timescale-dependent transitions. Parallel filtering has extra memory and arithmetic costs.
- Bound differentiation memory with checkpointing or chunking that carries the complete filter state and preserves gradients and temporal dependence across chunk boundaries.
- Add conditional simulation smoothing for joint trajectory draws, preserving cross-time/component dependence and existing reporting rotations. Marginal smoother variances alone do not provide joint draws.
- Carry qualified support through queries, updates, persistence, and checkpoints in explicit increments. Existing warmup checkpoints are not transferable across package, backend, hardware, or runtime identities; fit initialization and exact checkpoint continuation are separate contracts.

Maintain current response restrictions in the candidate matrix: exact Identity; fixed-integer Gamma shapes with finite-tail qualification; qualified Gaussian approximations; admitted fixed-shape BachSCR cases. Full response-family parity requires additional work. Compare approximate paths against the same declared finite-response target and report approximation error separately from sampling error.

**Deliverable:** A go/no-go recommendation for a separate state-space posterior implementation plan. GPU optimization of released R and grouped GP can proceed independently of that decision.

## 8. Stage D — length scaling and representative fits

- [ ] Build a duration ladder near 1/8, 1/4, 1/2, and full recordings using actual run boundaries and native support. Keep all 30 available participants, feature representation, K, priors/penalties, and masks matched between implementations within each case.
- [ ] Run K=1/K=3 synthetic parity controls and the representative K=5 workload. Record response-state size and parameter count for every cell.
- [ ] Use fixed and learned response configurations as separate workloads with matched support within each comparison. Include the intended empirical response configuration early enough to expose its state-size cost.
- [ ] Run inexpensive operation probes before bounded complete fits. Run a posterior pilot only after initialization and likelihood/gradient qualification pass.
- [ ] Declare time caps, host/GPU memory caps, repeated timing count, seeds, and termination rules in the campaign manifest before launching that campaign. Derive the full-scale budget from measured smaller cases and the user's acceptable turnaround; hardware capacity alone does not determine the budget.
- [ ] Run timed candidates without competing benchmark jobs, then measure concurrent scheduling as a separate experiment.
- [ ] Stop an arm that exceeds its declared cap and retain its failure. Do not turn a resource-stopped posterior into a convergence result or automatically extend its budget.

A performance winner must pass the applicable numerical gates and improve complete-workflow cost or enable a previously infeasible workload. Report R/GP-MAP time to a qualified solution separately from posterior effective samples per second. Use the existing posterior R-hat, ESS, divergence, tree-depth, BFMI, and orientation checks; fix the exact thresholds in the campaign protocol before sampling. Compare posterior summaries using Monte Carlo uncertainty, not bitwise sample equality across devices.

Compare algebra and physical gradients at identical parameter vectors before comparing optimized endpoints. Retain existing released numerical tolerances; any proposed revision requires a separately justified numerical study. Optimization paths and local optima can differ, so report qualified endpoint objectives and predictions as well as elapsed time.

## 9. Evidence, CI, and final decisions

Future local campaign outputs belong under ignored `local_data/linux-performance-v1/`, with separate environment/workload/campaign manifests, logs, per-case metrics, failures, frozen candidate identities, and a final decision report. Record CPU/GPU/backend/precision/chain configuration, cold and warm timing, complete-fit timing, memory, qualification, and numerical discrepancies. Hash the baseline and relevant inputs; do not overwrite existing scientific archives.

Existing synthetic CPU correctness and installed-package checks remain in ordinary CI. After a production optimization is selected, add small focused parity/workflow regressions for the changed path. Real-data scaling, long posterior runs, and GPU benchmarks run on demand on the workstation. No new 100-case calibration campaign or long performance campaign is added to standard PR CI by this plan.

The final evaluation report must select independently:

1. The preferred R optimization and scheduling strategy.
2. The preferred GP-MAP backend and scheduling strategy.
3. The preferred GP posterior backend, chain scheduling, and precision.
4. Whether state-space posterior sampling merits implementation now, needs a narrower feasibility step, or should remain deferred.

For each decision, record tested workload limits, unsupported responses, failed cases, setup cost, dependency implications, and the smallest production implementation with meaningful tests. Split R and GP implementation work into separately reviewable plans. Do not change defaults solely from a favorable microbenchmark, or treat performance/convergence as evidence of physiological recovery or broad posterior calibration.

## References

- [Package model handoff](../../model-integration.md) and [release process](../../releasing.md).
- [Prior Apple MPS feasibility findings and scalar evidence](../../performance/2026-09-19-apple-mps-feasibility.md), including the corrected optimized CPU baseline and limits of the historical tests.
- [Linux CUDA GPU opportunities](../../performance/2026-09-21-linux-cuda-gpu-opportunities.md): September 21, 2026 synthetic measurements on the workstation's RTX 3090, partial RTX PRO 6000 rows, CPU thread findings, and the state-space memory prerequisite. These inform Stage C candidate selection but are not the release-baseline measurements this plan requires.
- Historical development reports: `shared-response-models/docs/empirical/cpu-optimization-v3.md`, `gp-cpu-optimization-v4.md`, `docs/GP_STATE_SPACE.md`, `docs/GP_POSTERIOR_WORKFLOW.md`, and `docs/GP_WARMUP_CHECKPOINTS.md`. Read the released equivalents when available; earlier timings are context, not Linux results.
- [JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html) and [associative scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html).
- [NumPyro MCMC execution modes](https://num.pyro.ai/en/latest/mcmc.html).
- [Parallel Kalman GPU performance study](https://arxiv.org/abs/2511.10363).
- [RTX PRO 6000 specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/), [RTX 3090 specifications](https://www.nvidia.com/en-eu/geforce/graphics-cards/30-series/rtx-3090/), and [CUDA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
