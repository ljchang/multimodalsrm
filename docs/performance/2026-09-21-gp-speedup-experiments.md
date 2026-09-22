# GP-MSRM speedup experiments: MAP, posterior and grouped state-space

**Measurement date:** September 21 to 22, 2026, on the Linux workstation. **Status:** research note on development source `e47fdda`; no production code, default or release changes. Companion to the [algorithmic opportunities note](2026-09-21-gp-algorithmic-opportunities.md), which derives the same candidates and benchmarks them on small cases, and to the [CUDA note](2026-09-21-linux-cuda-gpu-opportunities.md). This note measures the candidates at the recorded empirical shape, with structured synthetic data whose generating latent, loadings, responses and noise are known.

## Summary

| Candidate | Evidence | Measured effect | Kind of gain |
| --- | --- | --- | --- |
| Warm-start GP MAP from a fast R-MSRM fit | Full-scale synthetic (57,757 parameters), same optimum reached | 176 iterations and 81 s versus 1,200 iterations and 526 s from the default start: 6.5x, plus 13 s for the R fit | Fewer gradients; works with any backend and device |
| Grouped-node state-space likelihood | Exact to 1e-16 in value and gradient against dense grouped for Identity responses at 934 to 7,500 nodes | CPU: 3.4x at the one-eighth length, 9x at double, 25x at four times; linear growth versus cubic | Changes the scaling; the only route that makes long single runs cheap |
| Same on GPU | Sequential scan is latency-bound at about 0.4 ms per node | 8x slower than dense grouped on the PRO 6000 at the one-eighth length | Needs a parallel (associative) scan or stays on CPU |
| Blocked Gibbs posterior (simulation smoother plus conjugate loadings) | Every block validated against exact conditionals; agrees with NUTS on invariants (cross-sampler R-hat at most 1.10); one sweep costs 0.22 s at full parameter count on the PRO 6000 | 26x to 920x more effective samples per second than NUTS on a 311-parameter exact problem | Replaces hundreds of gradients per draw with one smoother pass |
| NUTS at full scale | 57,756 parameters, diagonal metric, 100 warmup | Every draw saturated the depth cap (255 leapfrog steps), minimum ESS 1.5 after 160 iterations and 6,300 s | Sets the multiplier: 40 to 160 s per draw on the best card |
| ArviZ diagnostics phase | 57,758 quantities, 15 draws | 106 s; per-parameter Python loop, about 1 to 3 ms per quantity | Replace with a vectorized rank-normalized ESS |

Everything below uses the same synthetic shape as the CUDA note: four participants with a brain stream of F features every 1.8 s (252 samples at the one-eighth length), five participants with a 5-feature rating stream every 1.2 s and a 2-feature physiology stream every 1.5 s, K = 3 unless stated, one run. The generator draws one shared Matérn-3/2 latent per run with length scale 3 s, Gaussian rating and physiology responses with lags 0.4 s and 1.0 s, known loadings and offsets, and independent noise. The one-eighth length has 934 unique modality-time nodes; the recorded empirical case had about 909.

## 1. MAP: warm start from R-MSRM

Rationale: the regularized model fits in seconds on CPU and estimates the same objects the GP needs as a starting point (mappings, latent scale, offsets, noise, response lags). The GP MAP search then only has to refine. Translation used here: GP loading = R loading × R feature scale × latent factor standard deviation (the GP latent has unit stationary variance); GP offset = R feature mean; GP noise = residual variance of the R reconstruction in supplied units; filter lags = R group kernels, clipped inside the prior bounds. Rotation is left as is, since the GP loading prior is isotropic.

`warm_start.py` runs the package's own `search` with one start from either the default data-based start or the translated R fit, with `maxiter` 1,200 and `gtol` 1e-5.

| Scale | R fit | Start | Objective at start | Iterations | Function evaluations | Time | Final objective | Recovered lags (true 0.4, 1.0) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| F = 360, 5,914 parameters, PRO 6000 | 2.5 s, 21 iterations | default | 1,123,896 | 246 | 272 | 21 s | 406,583.5287 | 0.408, 1.040 |
| | | R warm | 407,897 | 96 | 109 | 7.9 s | 406,583.5288 | 0.408, 1.040 |
| F = 3,600, 57,757 parameters, 3090 shared with another job | 13 s, 23 iterations | default | 10,771,490 | 1,200 (budget) | 1,328 | 526 s | 3,914,192.4866 | 0.391, 0.994 |
| | | R warm | 4,021,507 | 176 | 198 | 81 s | 3,914,192.4861 | 0.391, 0.994 |

Both starts reach the same optimum to ten significant digits with the same lags. Neither full-scale run met the 1e-3 physical projected-gradient tolerance (1.5 and 3.0 remained), which at an objective of 4 million is an absolute tolerance the package's refinement and polishing steps are meant to close. A separate 600-iteration run from the same default start on the same data ended in a different basin with the rating lag at its 1.5 s bound and a worse objective, so the default start is fragile at this scale. That divergence between two runs of a deterministic optimizer is consistent with nondeterministic float64 scatter-add ordering on the GPU changing gradients at the last bits; it is worth recording rather than assuming.

The warm start is cheap, backend-independent, and stacks with every other gain. It also gives a principled way to cut `starts` from 16 to a few, since the R restarts are the cheap exploration.

## 2. Grouped-node state-space likelihood

`grouped_kalman.py` folds every observation at a (modality, time) node into K×K information statistics and a K-vector score, exactly the block statistics the grouped dense path already computes, then runs one Kalman information update per node over a factor-major state built from the package's Matérn-3/2 transitions and, for other responses, its `ResponseStateSpace` realization. The marginal likelihood is the sum of per-node log-determinant and quadratic terms with the observation constants. `jax.grad` through `lax.scan` gives the gradient. The companion note's `benchmark_gp_node_filter.py` is the same idea with per-step checkpointing; the two were written independently and agree.

### Exactness

Identity responses make the state-space form exact, so the comparison is against the grouped dense likelihood at a random parameter vector:

| K | Length | Nodes | Dense dimension | Relative objective difference | Relative gradient difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1/8 | 934 | 2,802 | 1.3e-16 | 8.0e-16 |
| 3 | 1/4 | 1,866 | 5,598 | 0 | 1.5e-15 |
| 3 | 1/2 | 3,731 | 11,193 | 1.3e-16 | 2.6e-15 |

### Cost

Objective and gradient, warm medians of three, at a random parameter vector; 360 brain features per participant (0.38 million observations at the one-eighth length) so that the observation-side cost stays small and the node-side cost is isolated. Load average 5 to 32 from other work throughout. "Exact" columns are relative differences against dense grouped in objective and gradient.

CPU, 8 OpenBLAS threads:

| K | Length | Nodes | Dense dimension | Dense grouped | Grouped Kalman | Speedup | Exact (objective / gradient) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 1/8 | 934 | 2,802 | 0.36 s | 0.09 s | 3.9x | 1e-16 / 8e-16 |
| 3 | 1/4 | 1,866 | 5,598 | 1.60 s | 0.18 s | 8.9x | 0 / 2e-15 |
| 3 | 1/2 | 3,731 | 11,193 | 17.6 s | 0.31 s | 56x | 1e-16 / 2e-15 |
| 3 | full | 7,461 | 22,383 | not run | 0.55 s | | |
| 5 | 1/8 | 934 | 4,670 | 1.08 s | 0.17 s | 6.5x | 0 / 1e-15 |
| 5 | 1/4 | 1,866 | 9,330 | 5.26 s | 0.32 s | 16x | 0 / 4e-16 |
| 5 | 1/2 | 3,731 | 18,655 | 66.7 s | 0.51 s | 131x | 2e-16 / 2e-15 |

RTX PRO 6000:

| K | Length | Nodes | Dense dimension | Dense grouped | Grouped Kalman | Ratio | Exact (objective / gradient) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 1/8 | 934 | 2,802 | 0.045 s | 0.36 s | 0.12x | 0 / 6e-15 |
| 3 | 1/4 | 1,866 | 5,598 | 0.27 s | 0.73 s | 0.37x | 1e-16 / 2e-14 |
| 3 | 1/2 | 3,731 | 11,193 | 1.99 s | 1.46 s | 1.4x | 1e-16 / 1e-14 |
| 3 | full | 7,461 | 22,383 | not run | 3.02 s | | |
| 5 | 1/8 | 934 | 4,670 | 0.17 s | 0.36 s | 0.48x | 0 / 1e-14 |
| 5 | 1/8, 3,600 features (3.64 M observations) | 934 | 4,670 | 0.19 s | 0.38 s | 0.50x | 2e-16 / 2e-14 |
| 5 | 1/4 | 1,866 | 9,330 | 1.16 s | 0.71 s | 1.6x | 2e-16 / 9e-15 |
| 5 | 1/2 | 3,731 | 18,655 | 9.06 s | 1.47 s | 6.2x | 0 / 5e-15 |
| 5 | full | 7,461 | 37,305 | not run | 3.02 s | | |

Readings:

- On CPU the Kalman gradient is linear in nodes (0.09, 0.18, 0.31, 0.55 s at K = 3) and nearly independent of K (0.51 s at K = 5 and half length). Dense grouped is cubic. The full recording as one run at K = 5 costs 0.5 to 0.6 s per gradient on CPU where dense would need a 37,000-dimension factorization per gradient, which the CUDA note put at minutes and 10 GB per matrix.
- On the GPU the scan costs 0.39 ms per node whatever the state size (6, 10, 150 or 250 dimensions all give 0.36 s at 934 nodes), because each node is a chain of tiny dependent kernels. The crossover with dense grouped is near 10,000 dense dimensions. Below that, dense on the PRO 6000 remains the fastest measured gradient (45 ms at the one-eighth length); above it, the Kalman scan wins on either device, and the CPU scan is 3 to 6x faster than the GPU scan.
- Going from 360 to 3,600 brain features (0.38 to 3.64 million observations) added 20 ms on the GPU, since the observation side is one pass of segment sums.
- The companion note's next step, an associative (parallel-in-time) scan, is what would make the GPU competitive at short lengths; it is not needed to realize the long-recording gain, which the CPU already delivers.

### Gaussian responses

The package's compact rational realization gives 24 states per Gaussian response per factor, so with two Gaussian responses the state has 50 dimensions per factor, 150 at K = 3 and 250 at K = 5. The grouped Kalman likelihood using that realization agrees with the dense grouped likelihood, which uses the analytic Gaussian-response covariance, far inside the declared covariance tolerance:

| Device | K | Length | Nodes | State dimension | Dense grouped | Grouped Kalman | Relative difference (objective / gradient) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PRO 6000 | 3 | 1/8 | 921 | 150 | 0.044 s | 0.36 s | 6e-13 / 8e-12 |
| PRO 6000 | 5 | 1/8 | 921 | 250 | 0.17 s | 0.37 s | 8e-13 / 2e-12 |
| PRO 6000 | 3 | 1/2 | 3,719 | 150 | not run | 1.39 s | |
| CPU | 3 | 1/8 | 921 | 150 | 0.37 s | 1.65 s | 6e-13 / 8e-12 |
| CPU | 5 | 1/8 | 921 | 250 | 1.04 s | 2.99 s | 8e-13 / 2e-12 |
| CPU | 3 | 1/2 | 3,719 | 150 | not run (17.6 s for the same dimension with Identity) | 5.40 s | |

On the GPU the response banks are free (the scan is latency-bound). On CPU each node costs a dense (K·50)-dimension covariance prediction, so the state size dominates: with Gaussian responses the Kalman gradient is 4.5x slower than dense at the one-eighth length and about 3x faster at half length, with the crossover near a quarter of a recording. The likelihood stays linear in nodes (1.65 s to 5.40 s for 4x the nodes). Two structural reductions apply: the transition is block-diagonal across factors, so the prediction can be done per factor in 50-dimension blocks (K² blocks of 50×50 instead of one (50K)² matrix, about 3x fewer operations at K = 3), and the measurement update is rank K. Beyond that, the bank order itself (24 per Gaussian response in the compact rational realization) is the lever, and it trades against the declared covariance tolerance.

## 3. Posterior: blocked Gibbs against marginal NUTS

`gibbs_proto.py` is a prototype of the companion note's "Gaussian block sampling" route for Identity responses with fixed filters. One sweep draws the whole latent state path by forward filtering and backward sampling over the grouped nodes (the same recursion as section 2, so it costs about one likelihood pass), then draws every loading and offset from its exact Gaussian conditional in one batched (K+1)-dimensional solve per feature, updates each group's noise variance with five random-walk Metropolis steps on the log scale under the lognormal prior, and finishes with two moves that leave the likelihood invariant and only have to satisfy the priors: a scale move (path × c, loadings ÷ c) and a Haar rotation of the factor coordinates. Its target is the joint posterior over parameters and path, whose parameter marginal is exactly the density the package's NUTS samples.

### Validation

Each block was checked separately, which paid off:

- **Simulation smoother.** 4,000 path draws at fixed parameters against the exact Gaussian conditional from the dense grouped algebra (376 latent quantities): z-scores of the means have root-mean-square 1.00 and maximum 3.2, standard-deviation ratios lie in [0.97, 1.03], and neighbouring-node correlations match to 1e-15.
- **Noise step.** 200,000 Metropolis steps at a fixed state against the numerically integrated one-dimensional conditional: mean and standard deviation of log variance agree (z = 0.3).
- **Rotation move.** Disabling it leaves every invariant's posterior mean within 0.2 posterior sd, as a correct move must.
- **Scale move.** Disabling it changed the total loading norm from 394 to 180. The move's acceptance uses the prior quadratic form of the path, and that form was 1e28 times too large: node times from different native clocks (1.2 s and 1.8 s grids) differ by 1e-16 s from floating-point rounding, which gives process-noise matrices of order 1e-48 that the quadratic form inverts, while the smoother's 1e-12 covariance jitter makes the corresponding residuals far larger than exact arithmetic would. The likelihood never inverts those matrices, which is why it was exact all along. Treating node times closer than 1e-9 s as tied fixed it: full versus no-scale-move now agree within 0.013 posterior sd on every quantity, and the likelihood stays exact to 1e-16 and 2e-15.

The first NUTS comparison (599 parameters, diagonal mass matrix, 400 warmup) ran before that fix and is superseded; its Gibbs posteriors were biased along the scale ridge, which appeared as loading norms 31 posterior sd away from NUTS. It is retained in `gibbs_compare.jsonl` as a record of the failure mode.

### Cost per sweep at realistic scale

| Device | K | Length | Nodes | Observations | Parameters | Gibbs sweep | Kalman objective + gradient | NUTS leapfrog (from the CUDA note) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU, 8 threads | 3 | 1/8 | 934 | 3.64 M | 57,754 | 0.33 s | 0.42 s | 2.0 s |
| PRO 6000 | 3 | 1/8 | 934 | 3.64 M | 57,754 | 0.22 s | 0.37 s | 0.16 s |
| PRO 6000 | 5 | full, one run | 7,461 | 29.1 M | 86,624 | 1.83 s | 4.64 s | not feasible with dense grouped |

One sweep costs roughly one to two NUTS leapfrog steps, and a NUTS draw at this scale needs tens to hundreds of leapfrog steps (section 4). Whether that translates into effective samples depends on mixing, measured next.

### Effective samples per second

`gibbs_compare3.py`: K = 2, 16 brain features, a 0.3-length recording (281 nodes, 8,624 observations, 311 parameters), Identity responses, both samplers started from the same single-start MAP. NUTS through the package's `sample` with a dense mass matrix, target acceptance 0.9, depth cap 8, two chains of 300 warmup and 300 draws. Gibbs: two chains of 500 warmup and 10,000 sweeps thinned by 5, the second chain started from a perturbed point (loadings scaled by 1.3 with noise, offsets perturbed, variances doubled). Quantities are the marginal log-likelihood, the noise variances, the squared loading norms per feature and the offsets, all invariant to factor rotation. Effective sample sizes are ArviZ bulk ESS over both chains; time is sampling wall time on the loaded CPU.

| Quantity | NUTS min ESS (R-hat) | NUTS ESS per s | Gibbs min ESS (R-hat) | Gibbs ESS per s | Gibbs / NUTS | Cross-sampler R-hat, max | Max mean difference | Median sd ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Log-likelihood | 139 (1.017) | 0.048 | 2,370 (1.000) | 20.1 | 419x | 1.009 | 0.09 sd | 1.00 |
| Noise variances (19) | 276 (1.010) | 0.095 | 2,187 (1.002) | 18.5 | 195x | 1.007 | 0.09 sd | 0.99 |
| Loading norms (99) | 16 (1.107) | 0.005 | 594 (1.005) | 5.0 | 920x | 1.062 | 0.25 sd | 0.96 |
| Offsets (99) | 66 (1.037) | 0.023 | 69 (1.028) | 0.58 | 26x | 1.101 | 0.35 sd | 0.99 |

NUTS took 2,914 s for 153,000 leapfrog steps: every iteration ran to the depth cap even with a dense metric, which is what a flat rotation direction does to a Hamiltonian trajectory (the package's Haar refresh addresses reporting, not the trajectory length). Gibbs took 59 s per chain. The two samplers agree: pooling the two NUTS and two Gibbs chains gives R-hat below 1.11 on every quantity, posterior means differ by at most 0.35 of a posterior sd, and posterior spreads match to 4 percent. Offsets are the slowest Gibbs block, because an offset and the mean of the latent path are nearly confounded within a run; an interweaving step for offsets would address that directly.

Combined with the per-sweep costs above, the blocked Gibbs route is the largest posterior speedup measured in this investigation: two to three orders of magnitude in effective samples per second on a small exact problem, with per-sweep cost at full scale equal to one or two leapfrog steps. What it does not yet have: learned response parameters and timescale (a small Metropolis or NUTS block on the marginal Kalman likelihood, which section 2 makes cheap), the K = 1 positive anchor, masks with feature-specific missingness (the conjugate solves already handle them through the segment sums), multi-run sharing of loadings (the conjugate statistics simply sum over runs), and the package's diagnostics, orientation and persistence contracts.

## 7. What to build, in order

1. **Warm start GP MAP from R-MSRM** (section 1). A translation function and a `starts` policy. Days of work, backend-independent, 6.5x on the full-scale MAP with the same optimum.
2. **Grouped-node state-space likelihood as a production backend** (section 2), sequential scan on CPU first. It is exact for Identity, within 1e-12 of dense for Gaussian responses through the existing realization, linear in nodes, and the only path that makes a full recording as one run cheap (0.5 to 0.9 s per gradient at K = 3 to 5). Per-factor block propagation and rank-K updates come next for the Gaussian-response CPU cost; an associative scan only if GPU execution at short lengths matters.
3. **Blocked Gibbs posterior on that backend** (section 3), starting from fixed responses, then a marginal Metropolis block for the response and timescale parameters, then the anchor, masks, multi-run and diagnostics contracts. Validate as here: each block against its exact conditional, then invariants against NUTS on a small problem.
4. **Vectorized diagnostics** (section 5). Independent of everything else and worth doing first if long posterior runs start before item 3.

Keep dense grouped on the PRO 6000 for short single runs with MAP, where it remains the fastest gradient measured (45 ms), and keep the CUDA note's thread policy for every CPU run.

## 4. Posterior: NUTS geometry at full scale

`nuts_depth.py` ran the package's `search` (one start, 600 iterations) and then `sample` on the full-scale synthetic case (K = 3, 3,600 brain features, 57,756 parameters, diagonal mass matrix as the package requires above 1,024 parameters, target acceptance 0.9, 100 warmup and 60 retained draws, tree depth capped at 8) on the RTX 3090 while that card was shared with the warm-start run for part of the time.

| Quantity | Value |
| --- | --- |
| MAP after 600 iterations | objective 3,917,564, projected gradient 3,919, rating lag at its 1.5 s bound (true 0.4): not converged, wrong basin |
| Warmup + sampling | 6,332 s for 160 iterations |
| Leapfrog steps per retained draw | 255 in every draw (all 60 saturated the depth-8 cap) |
| Mean acceptance | 0.97 (step size adapted too small in 100 warmup iterations) |
| Per leapfrog | 0.41 s |
| Minimum bulk ESS over 57,756 parameters | 1.5 |
| Diagnostics phase | 82 s |

Read with the caveats that warmup was one tenth of the package default and the start was a poorly converged MAP, the result still fixes the order of magnitude: at this dimension NUTS with a diagonal metric takes hundreds of leapfrog steps per draw, so a draw costs 40 s on the PRO 6000 (0.16 s per step) and up to 160 s at the package's default depth of 10. The default budget of 2,200 iterations is then one to four days per chain on the best card, before mixing is established. Every leapfrog step is a full gradient, so the section 1 and 2 gains apply to the step cost but not to the step count. Reducing the step count needs either a metric the diagonal cannot provide, a reparameterization, or a different sampler, which is what section 3 measures.

## 5. Diagnostics phase

`diagnostic_summary` calls ArviZ `summary` and tail `ess` over every quantity. Profile at the measured shapes:

| Chains | Draws | Quantities | Time | Dominant cost |
| ---: | ---: | ---: | ---: | --- |
| 1 | 15 | 57,758 | 106 s | 115,516 calls into ArviZ's per-quantity ESS/R-hat functions |
| 2 | 15 | 57,758 | 146 s | same |
| 1 | 200 | 5,000 | 17 s | same, 3.5 ms per quantity |

The cost is per quantity, not per draw, so at full budgets it stays in the minutes-to-tens-of-minutes range rather than exploding, but it is pure Python overhead. A vectorized rank-normalized split R-hat and ESS over a (chains, draws, quantities) array, as NumPy or JAX batched FFT autocorrelations, would take seconds and could keep the existing thresholds and tail definition. This is the cheapest of all the items here.

## 6. Other measurements

- **Batched restarts with `vmap`.** On the 3090, evaluating eight parameter vectors in one call cost 0.24 s per gradient versus 0.31 s for one at a time, a 1.3x gain, because one 2,727-dimension factorization already saturates the card. Batching restarts on the GPU is not a leap.

## Evidence and limits

Scripts and logs are retained under ignored `local_data/linux-performance-v1/2026-09-21-gp-speedups/`. All data are synthetic and seeded. The machine was shared throughout (load average 5 to 16, and the 3090 ran two of these experiments concurrently for part of the time), so absolute timings are upper bounds; iteration counts, exactness comparisons and effective-sample counts do not depend on load. No empirical data were used, and no claim is made about convergence, calibration or recovery beyond the reported lags on one synthetic realization.
