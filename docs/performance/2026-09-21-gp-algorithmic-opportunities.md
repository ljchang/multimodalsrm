# Algorithmic opportunities for substantially faster GP-MSRM

Research date: September 21, 2026. Source inspected: `e47fdda031e098acbf4510157a54e5e00311dd5a`.
The user needs both continuous long recordings and multiple independent runs.
This investigation adds two synthetic research probes; production model code and defaults are unchanged.

**Recommendation:** first build a state-space likelihood that assimilates exact observation groups, then investigate an alternative posterior that integrates out loadings. For MAP, add EM/ECME updates using Gaussian posterior moments. These exploit this particular model's structure. A multifactor spectral approximation is a separate, promising option when approximation error is acceptable.

The existing [CUDA investigation](2026-09-21-linux-cuda-gpu-opportunities.md) establishes useful hardware gains, but the largest remaining opportunities change what is computed. MAP versus posterior describes the inference target; dense, spectral, and state-space describe the representation. They are separate choices.

## What the implementation currently spends time doing

Let `N` be observed scalars, `U` unique modality/time functionals per run, `K` latent factors, and `q` response states per factor. Write `d = Kq`.

| Path | Current computation | Consequence |
| --- | --- | --- |
| Grouped likelihood | Exact sufficient statistics followed by a dense `UK × UK` factorization; the gradient path also obtains a dense inverse | Roughly `O((UK)^3)` time and `O((UK)^2)` memory per run |
| State-space likelihood | One transition and scalar measurement update for each of `N` scalars; dense covariance prediction and Joseph-update products | Repeated work at identical times, `N × d × d` transition arrays, large differentiation intermediates |
| MAP search | L-BFGS-style joint optimization over loadings, offsets, noise, and response parameters | Repeated expensive gradients despite conditional Gaussian structure |
| Posterior | Latent GP marginalized; NUTS samples the remaining physical parameters | A high-dimensional Hamiltonian trajectory can require many expensive likelihood gradients |

Relevant source: [state-space filtering](../../src/multimodalsrm/bayesian/state_space.py), [block statistics](../../src/multimodalsrm/bayesian/multifactor.py), [analytic grouped score](../../src/multimodalsrm/bayesian/multifactor_score.py), [symmetric factorization](../../src/multimodalsrm/bayesian/multifactor_cholesky.py), [fitting](../../src/multimodalsrm/bayesian/fitting.py).

State-space currently admits **MAP only**, and learned GP length scale currently requires dense/grouped inference. The existing spectral backend supports **one factor**. Those are actual API restrictions, not restrictions of the underlying methods.

## 1. Exact grouped state-space updates: strongest immediate candidate

Observations sharing a response functional have the form

\[
y_i-b_i=w_i^T z_j+\epsilon_i,\qquad \epsilon_i\sim N(0,v_i).
\]

At node `j`, retain

\[
D_j=\sum_i w_iw_i^T/v_i,\quad
h_j=\sum_i w_i(y_i-b_i)/v_i,\quad
c_j=\sum_i[(y_i-b_i)^2/v_i+\log(2\pi v_i)].
\]

The observation log density is exactly `−(zᵀDz − 2hᵀz + c)/2`. Thousands of feature measurements can therefore be replaced by one `K`-dimensional information update. Retaining `c` is essential for the likelihood and noise gradients. Statistics depend on the proposed parameters and must be recomputed at each evaluation.

This requires no time bins, resampling, feature truncation, or independent participant fits. Missing entries are simply excluded. Distinct modalities remain distinct nodes even when their timestamps coincide. The algebra also works when `D` is singular: a production implementation must not assume it can invert `D`.

I implemented a narrow probe in [benchmark_gp_node_filter.py](../../scripts/benchmark_gp_node_filter.py). It uses existing response realizations, recomputes exact statistics, advances once per node, and checkpoints the scan body. All comparisons include the same parameter priors and all physical parameter gradients.

Warm median objective-and-gradient times on this workstation's CPU, float64:

| Synthetic case | Dense grouped | Current scalar state-space | Node state-space prototype |
| --- | ---: | ---: | ---: |
| Identity, K=3, 128 nodes, 8,192 scalars | 15.73 ms | 158.50 ms | **6.99 ms** |
| Identity, K=3, 128 nodes, 65,536 scalars | 22.63 ms | 1,425.97 ms | **13.15 ms** |
| Identity, K=3, 512 nodes, 32,768 scalars | 919.03 ms | 650.27 ms | **21.63 ms** |
| Identity, K=5, 128 nodes, 8,192 scalars | 52.30 ms | 144.01 ms | **7.34 ms** |
| Identity + fixed Gaussian, K=3, short case, 66 states | **2.72 ms** | 179.85 ms | 14.28 ms |
| Identity + fixed Gaussian, K=3, 504 nodes, 8,064 scalars, 66 states | 787.00 ms | Not run | **88.76 ms** |

These are **microbenchmarks, not complete-fit or posterior speedups**. The first four Identity comparisons preserve the exact model. Gaussian compression preserves the existing state-space realization; its approximation to the dense Gaussian-response model remains separately visible.

Findings and qualifications:

- Node compression was approximately **13–108× faster than scalar filtering** in the ordinary cases above. It was about **42× faster than this CPU dense-grouped baseline** at 512 Identity nodes, and about **9× faster** in the longer Gaussian case. These ratios cannot be transferred to a tuned GPU dense baseline.
- Identity gradients agreed with dense grouped to about `1e-11` or better at ordinary noise. Additional masked, all-zero-loading and low-noise cases passed assertions; the largest low-noise gradient difference was `4.2e-7` in absolute units.
- The short Gaussian case agreed with the current scalar state-space likelihood to `1.2e-13` and its gradients to `6.1e-14`. Relative to dense grouped, maximum gradient differences were `1.7e-8` in the short case and `8.5e-8` in the longer case. This is evidence at fixed parameters, not a bound on all likelihood gradients.
- For 65,536 Identity observations at 128 nodes, transition storage fell from **37.75 MB to 73.73 KB**, exactly **512×**. Observed-value storage and statistics construction remain proportional to the data size.
- Small dense problems can win: the short Gaussian case favored dense grouped by about **5×**. Response-state size matters as much as recording length.
- CPU timings were variable, especially the existing callback-based dense path. An eight-BLAS-thread repeat of the 512-node case took 1.280 s versus 22.49 ms; restricting CPU affinity to eight logical CPUs with one BLAS thread gave 1.161 s versus 20.23 ms. Neither is a claim of optimal CPU tuning. Retained repeats show the variation.

The probe is deliberately limited to fixed responses/timescale, positive white noise, and one synthetic run. It uses covariance subtraction and a small nonsymmetric solve, not a fully qualified square-root filter. It has no prediction, persistence, learned-lag, or sampling integration. Production work must retain the package's numerical safeguards.

### What should follow observation grouping

1. **Preserve transition structure.** Independent prior factors share the same `q × q` transition. Store that once and propagate the `K × K` covariance blocks using it. Dense propagation can drop from `O(K^3 q^3)` to `O(K^2 q^3)` arithmetic. Posterior cross-factor covariance must still be retained.
2. **Use low-rank measurement updates.** Each functional observes only `K` directions in a `Kq` state. Avoid dense `d × d` Joseph products for each scalar. Use stable square-root or factored updates and compare their gradients at high signal-to-noise ratios.
3. **Control differentiation memory.** Checkpoint scan chunks, not only individual steps, or implement a smoother-based likelihood score. Scan-body rematerialization alone still retains state inputs across time. The [JAX checkpointing documentation](https://docs.jax.dev/en/latest/gradient-checkpointing.html) explains the compute/storage tradeoff; it does not promise constant-memory scans.
4. **Then evaluate parallel filtering.** Associative filtering reduces sequential dependency depth to logarithmic in the number of events; total arithmetic and storage do not become logarithmic. It may add expensive dense operations at large `d`. Compare sequential CPU, sequential GPU, and parallel GPU on actual response sizes. [Särkkä and García-Fernández](https://arxiv.org/abs/1905.13002) provide the algorithm; [Dynamax](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_parallel_inference.html) provides an inspectable implementation.
5. **Address large response banks if they remain dominant.** Semiseparable/factored state representations can reduce constants and, for suitable structure, dependence on state rank. [Celerite's structured derivatives](https://arxiv.org/abs/1801.10156) are relevant. Celerite itself is not a drop-in solver for this model's arbitrary response operators and loading blocks.

Learned lags require sorting shifted **nodes**, handling coincident events and crossings, and preserving the existing delay-gradient semantics. Learned widths require differentiating the response realization. Neither was qualified by this probe.

## 2. Posterior: integrate out loadings instead of always integrating out trajectories

This is a particularly promising alternative when there are many more observed channels than time points. It is an adaptation of Gaussian marginalization and [dual probabilistic PCA](https://jmlr.org/papers/v6/lawrence05a.html) to this package's temporal prior, not a claim that the cited paper implements GP-MSRM.

For a participant/modality block, let `Y` be `T × F` and `Z` the `T × K` filtered latent trajectories. The package uses independent isotropic normal loading priors, normal offset priors, and one noise variance for that participant/modality. For K>1, integrating loadings and offsets gives

\[
C(Z)=vI+\sigma_w^2ZZ^T+\sigma_b^2\mathbf1\mathbf1^T,
\]
\[
-\log p(Y\mid Z,v)=\tfrac12\{F\log|C|+
\operatorname{tr}(C^{-1}YY^T)+TF\log(2\pi)\}.
\]

`C` is diagonal plus rank `K+1`. Woodbury and the determinant lemma reduce its factorization to **`(K+1) × (K+1)`**. Cache the exact Gram matrix `YYᵀ` when beneficial; no feature information relevant to this marginal density is discarded. Alternatively, multiply by the raw data when a long run makes the Gram matrix unattractive.

The independent [loading-collapse probe](../../scripts/benchmark_gp_loading_collapse.py) checks this density three ways for `T=252`, `F=3,600`, `K=3`:

| Same loading-marginalized observation density and gradient | Warm median |
| --- | ---: |
| Direct `252 × 252` covariance and raw observations | 76.23 ms |
| Woodbury `4 × 4` solve, raw observations | 1.212 ms |
| Woodbury `4 × 4` solve, cached Gram matrix | **0.165 ms** |

The Gram matrix took 6.46 ms to prepare and occupied 0.51 MB, versus 7.26 MB for the data block. Maximum absolute gradient disagreement was `5.9e-11`. The cached version was about **7.3× faster than raw-data Woodbury**, a more meaningful comparison than its approximately 460× advantage over the deliberately direct covariance oracle.

This single block eliminates **14,400 loading/offset sampling coordinates**. It retains 756 latent-functional values and one noise coordinate in the probe. A full model must add the shared GP prior, the other streams, and response/hyperparameter inference. These timings do **not** include those terms and are **not** a comparison against current NUTS.

For fixed response parameters, a dense whitened latent representation can cache one `U × U` temporal factorization shared across factors, rather than refactor a `UK × UK` observation posterior at each proposal. State-space or finite-basis prior representations offer other options. The resulting latent posterior is non-Gaussian, so this route still needs a sampler and a mixing study.

Important boundaries:

- Marginalizing and then conditionally drawing loadings preserves the **joint posterior**. Maximizing this loading-marginalized density does **not** reproduce the current parameter MAP; modes depend on what was integrated out.
- Arbitrary feature-specific masks require grouping columns by observed-time pattern or using separate masked calculations. A single full-block Gram matrix is not sufficient in that case. Different native clocks across modalities are compatible with separate blocks.
- Loadings and offsets are shared across runs. Their collapse requires concatenating eligible observations across those runs and retaining the resulting cross-run terms. The latent priors remain independent by run, but the collapsed observation likelihood no longer factors by run.
- K=1 has a positive loading anchor with a truncated prior; its marginalization needs separate treatment. The probe covers K>1.
- A large number of time points can make `YYᵀ` expensive; evaluate the raw-data form or structured alternatives. This is especially relevant to the continuous-run requirement.

## 3. Posterior alternative: Gaussian block sampling

A second route retains explicit trajectories and alternates conditional draws:

1. Draw a complete shared latent path conditional on all participants using a simulation smoother.
2. Draw each loading/offset vector with a small Gaussian regression solve, batched across features. Handle the K=1 anchor's truncation explicitly.
3. Update noise and response/GP parameters with suitable lower-dimensional kernels while preserving the existing priors.

The foundation is established [simulation smoothing](https://academic.oup.com/biomet/article-abstract/89/3/603/251989?login=false). The inference proposal here follows from the package's normal loading/offset priors. Its noise prior is generally lognormal or another admitted prior, so **noise does not automatically have an inverse-gamma conditional**.

This can replace hundreds of large collapsed-likelihood gradients per NUTS draw with one state-path draw and many tiny solves. However, loading/path dependence may slow mixing; centered/noncentered moves or interweaving may be necessary. Compare effective samples per second and predictive/rotation-invariant summaries, not sweeps per second. If combining collapsed hyperparameter updates with explicit paths, update ordering and path redraws must preserve the target.

The current state-space MAP guard should remain until sampling, joint trajectory draws, diagnostics, updates, and persistence are implemented and checked. Removing the guard alone does not provide a complete posterior backend.

## 4. MAP: EM/ECME can exploit the same Gaussian structure

The current MAP maximizes the parameter density after integrating the latent GP. A compatible EM scheme uses the latent posterior's **means and covariances** in the E-step, followed by prior-regularized regression updates for loadings and offsets. Noise updates can be small numerical optimizations under the existing physical priors. Response parameters can be updated with the actual marginal objective in ECME steps.

This is established in [GP factor analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/) and [time-delay GPFA](https://pmc.ncbi.nlm.nih.gov/articles/PMC4545403/); extending it to the exact response/mask/prior conventions here is implementation work.

Use a hybrid: EM to reach a good region, then the current physical-gradient convergence checks and marginal optimizer. EM can converge slowly near an optimum. Replacing posterior second moments with just a fitted path is a different procedure and can change the MAP result. No speedup for complete EM fits was measured in this investigation.

## 5. Explicit approximations worth evaluating

| Candidate | Why it fits this model | Main limitation |
| --- | --- | --- |
| Multifactor extension of the existing spectral backend | Sine bases evaluated at native timestamps; Gaussian convolution and delay already have analytic basis responses | Currently K=1 only; rank, padding, finite-response discrepancy and gradients need validation |
| Variational inducing functionals | Flexible native clocks and convolution observations; minibatching observations conditional on inducing variables | Approximate posterior; residual covariance must be accounted for, not just discarded |
| Frequency-domain/Whittle formulation | Response convolution becomes frequency-wise multiplication, delays become phases, and frequency blocks can be small | Regular/common sampling and missingness matter; independent-frequency likelihood is an approximation |

[Solin and Särkkä's Hilbert-space method](https://arxiv.org/abs/1401.5508) is the basis of the current spectral implementation. Generalizing it to K factors reduces the dense system from `UK` to `mK` for rank `m`. That can be substantial if `m << U`. However, preserving frequency resolution as duration grows generally requires increasing `m`; fixed-rank linear scaling is not an accuracy guarantee. Padding changes the cutoff as well as boundary behavior.

[Álvarez et al.'s variational inducing kernels](https://proceedings.mlr.press/v9/alvarez10a.html) directly address convolutional multi-output GPs. This is a more appropriate sparse-GP reference than applying an ordinary scalar inducing-point package without the response operators.

The closest recent empirical reference is **Gokcen et al. (2025), Fast Multigroup Gaussian Process Factor Models**. Their frequency-domain method achieved a median **25× complete-fitting speedup** on recordings from three visual areas, with much larger gains in selected simulations. They also document finite-window biases, including underestimated delays/timescales, and mitigation by tapering. Those are results for their model and data, not expected GP-MSRM speedups. Our irregular clocks and masks can couple Fourier frequencies, so an FFT does not automatically diagonalize this likelihood. [Author-hosted paper](https://users.ece.cmu.edu/~byronyu/papers/GokcenNeuralComput2025.pdf).

Approximate posterior options such as a structured variational distribution or a Laplace approximation could also replace long MCMC runs when their uncertainty is adequate. They require handling rotational nonidentifiability and checking uncertainty against a small exact reference; a dense Hessian over tens of thousands of raw loading coordinates is unattractive.

## Both recording configurations

| Requirement | Preferred first experiments |
| --- | --- |
| One long continuous run | Node state-space, structured transitions and bounded gradient memory; compare exact versus controlled spectral approximation |
| Many separate runs | Batched/parallel per-run likelihoods and smoother E-steps, aggregating shared-parameter updates; retain dense grouped for short runs when it wins |
| Many features but relatively few time points | Loading-collapsed posterior with raw-data versus Gram-matrix likelihood |
| Full posterior in either configuration | Compare loading collapse against Gaussian block sampling and current collapsed NUTS using effective samples per second |

Computational chunks inside a continuous run must carry the filtering distribution and gradients across boundaries. Resetting each chunk to stationarity would change the model. Likewise, fitting independent participant latents would lose the shared response.

## Prioritized implementation decision

1. **Production node state-space likelihood and smoother.** Start with fixed Identity and response parity, then learned delays/widths, masks, rank-deficient loadings, high signal-to-noise cases, and multiple runs. Keep a size-dependent comparison with dense grouped.
2. **A bounded loading-collapsed posterior pilot.** First test complete K>1 blocks with fixed response parameters. Restore conditional loading draws, reporting rotations, and prediction; compare to the existing posterior on a small problem. Extend masks, shared runs, and learned responses deliberately.
3. **EM/ECME MAP plus Gaussian block-sampling pilot.** Reuse the grouped smoother and posterior sufficient statistics. Decide by time to the same objective tolerance and by posterior effective samples per second.
4. **Multifactor spectral approximation if needed.** Establish rank/padding convergence on response parameters and posterior predictions, including run boundaries and extrapolation.
5. **Parallel scan and GPU tuning after profiling the new computation.** Neither should substitute for removing scalar-event repetition.

One correction to the earlier GPU note: replacing an explicit inverse by a custom reverse rule does not by itself remove dense cubic scaling. The exact log-determinant score requires `tr(S^-1 dS)`; the current code already uses analytic contractions. Streaming selected solves can reduce storage, and structured derivatives can do more, but “avoid the inverse” alone is not an orders-of-magnitude solution.

## Reproduction and evidence limits

The two scripts are standalone synthetic research probes with assertions, not package backends. Run from the checkout with Python 3.12 and the pinned Bayesian dependencies:

```sh
export PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/benchmark_gp_node_filter.py --times 128 --channels 32
python scripts/benchmark_gp_node_filter.py --times 128 --channels 256
python scripts/benchmark_gp_node_filter.py --times 512 --channels 32
taskset -c 0-7 python scripts/benchmark_gp_node_filter.py --times 128 --channels 32 --factors 5
python scripts/benchmark_gp_node_filter.py --times 32 --channels 4 --gaussian
taskset -c 0-7 python scripts/benchmark_gp_node_filter.py --times 256 --channels 8 --gaussian --skip-scalar
python scripts/benchmark_gp_node_filter.py --times 64 --channels 16 --mask --zero-loadings
python scripts/benchmark_gp_node_filter.py --times 64 --channels 16 --mask --noise 0.001
taskset -c 0-7 python scripts/benchmark_gp_loading_collapse.py
```

Runtime: Python 3.12.13, JAX 0.11.2, NumPy 2.5.3, SciPy 1.18.1, CPU float64. Seed 721. Five warm repeats for node filters, seven for loading collapse; device completion is synchronized. Compilation/first evaluation and individual repeats are recorded separately. Hardware was shared; observed load averages were approximately 5–14. No GPU jobs or empirical-data fits were launched. The scripts explicitly import this checkout via `PYTHONPATH=src`.

Raw JSON, source hashes, and the workload manifest are retained locally under ignored `local_data/gp-algorithm-research-2026-09-21/`. Compiled temporary-buffer estimates are not measured process peak memory. No full optimizer convergence, posterior mixing, learned-response derivative qualification, or empirical recovery is claimed. The evidence supports specific implementation experiments, with especially strong evidence for exact observation grouping.
