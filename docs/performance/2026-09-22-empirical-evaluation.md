# Empirical evaluation of the GP-MSRM speedup candidates on a real recording

**Measurement date:** September 22, 2026, on the Linux workstation (idle, load average 1 to 9). **Status:** research note; no production code, default or release changes. This note checks the candidates from the [algorithmic note](2026-09-21-gp-algorithmic-opportunities.md) and the [synthetic experiments note](2026-09-21-gp-speedup-experiments.md) against a small real multimodal recording. The data are not part of the repository; the loader and scripts live under ignored `local_data/empirical-eval-2026-09-22/`.

## Summary

| Candidate | Real-data result | Verdict |
| --- | --- | --- |
| Blocked Gibbs posterior | On the full 2,787-parameter problem, two chains of 5,000 sweeps take 7.4 min on the CPU and give at least 1,145 effective samples for every rotation-invariant quantity, with R-hat at most 1.007. NUTS on a problem half the size ran 47 min on the PRO 6000, saturated the tree-depth cap on every iteration and did not converge (R-hat 2.7, minimum ESS 2). | Most promising by a wide margin. One defect found and fixed: offsets did not mix until a shift move was added. |
| Grouped-node state-space likelihood | Exact on the real clocks (1e-15 with Identity responses, 1e-11 with Gaussian responses). CPU: 9x faster than dense grouped at K=3 and 16x at K=5 with Identity responses, 1.8x with three Gaussian responses. PRO 6000: slower than dense. | Confirmed; the Gaussian-response state size is the remaining cost. |
| Warm start of GP MAP from R-MSRM | The translated start has a much better objective (248,663 versus 297,841) but took as many iterations as the default (695 versus 652), reached a different lag basin with two lags at their bounds, and neither run met the gradient tolerance. | Does not transfer as measured; the lag landscape is multimodal on this data. |
| GPU for the dense grouped gradient | PRO 6000 129 ms, 3090 360 ms, idle 16-thread CPU 435 ms at K=3; 3,600 voxels and 72,790 parameters add 15 percent on the PRO 6000 and 57 percent on the CPU. | Confirmed; the node count sets the cost, not the feature count. |

## The recording

Five participants watched one 503 s clip. Streams on their native clocks: brain (fMRI at 2 s, 100 Neurosynth parcels, or 3,600 voxels for the high-feature case), 20 facial action units at 2 s, 16 continuous emotion ratings at 1 Hz, and three physiology channels averaged into 1.5 s bins from a 500 Hz stitched clock; one participant has no physiology. Censored volumes are masked. Every feature is z-scored. Loader assumptions that still need confirmation from the data preparer: the eight splice volumes are dropped to align the 260-volume brain files with the 252 TR-binned rows, and only censored volumes are masked.

Problem shapes at K = 3 with the grouped backend:

| Configuration | Observations | Unique nodes | Dense dimension | Parameters |
| --- | ---: | ---: | ---: | ---: |
| 100 parcels, Gaussian responses on behaviour and physiology | 189,292 | 1,298 | 3,894 | 2,790 |
| 100 parcels, Identity responses everywhere | 192,167 | 1,346 | 4,038 | 2,787 |
| 3,600 voxels, Gaussian responses | 4,480,292 | 1,298 | 3,894 | 72,790 |
| Reduced: 30 parcels, first 150 s, Identity responses | 31,890 | 403 | 1,209 | 1,387 |

This is one short run, so it says nothing about the long-single-run question that decides whether the state-space backend is mandatory.

## 1. Objective-and-gradient cost

The package's own jitted `value_gradient` at the default start, warm median of five:

| Device | 100 parcels, K=3 | 3,600 voxels, K=3 | 100 parcels, K=5 (dimension 6,490) |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 129 ms | 148 ms | 490 ms |
| RTX 3090 | 360 ms | 404 ms | 1,503 ms |
| CPU, 16 OpenBLAS threads, idle | 435 ms | 682 ms | 1,514 ms |

With the thread policy applied and the machine idle, the CPU sits within 1.2x of the 3090 at K=3, as the CUDA note predicted for an uncontended CPU. The PRO 6000 is 2.8x the 3090 and 3.4x the CPU.

## 2. Grouped-node state-space likelihood on real clocks

The prototype from the synthetic campaign, unchanged, against the dense grouped likelihood at a random parameter vector:

| Device | Responses | K | State dimension | Dense grouped | Grouped Kalman | Ratio | Objective / gradient relative difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | Identity | 3 | 6 | 836 ms | 89 ms | 9.4x | 0 / 3e-15 |
| CPU | Identity | 5 | 10 | 2,361 ms | 145 ms | 16x | 0 / 2e-15 |
| CPU | Gaussian on three streams | 3 | 78 | 795 ms | 451 ms | 1.8x | 1e-13 / 1e-11 |
| PRO 6000 | Identity | 3 | 6 | 120 ms | 426 ms | 0.28x | 0 / 4e-15 |
| PRO 6000 | Identity | 5 | 10 | 479 ms | 497 ms | 0.96x | 0 / 2e-15 |
| PRO 6000 | Gaussian on three streams | 3 | 78 | 117 ms | 506 ms | 0.23x | 1e-13 / 1e-11 |

The real clocks (2 s, 1 s, 1.5 s grids and masked volumes) raise no new issue: node grouping and the tied-time rule carry over. The Gaussian-response state size, 26 states per factor for three responses here, is what limits the CPU gain, and the per-factor block propagation and rank-K updates proposed in the experiments note remain the next step for that case.

## 3. MAP warm start from R-MSRM

Gaussian responses with learned lags, 100 parcels, K = 3, one start each, `maxiter` 1,200 and `gtol` 1e-5, on the 3090. The R-MSRM fit took 4 s and 22 iterations.

| Start | Objective at start | Iterations | Function evaluations | Time | Final objective | Projected gradient | Lags (face, rating, physiology) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 297,841 | 652 | 695 | 220 s | 238,163.39 | 0.023 | 2.96, -7.32, 1.39 |
| R warm | 248,663 | 695 | 753 | 235 s | 238,138.52 | 6.36 | 4.00 (bound), 2.45, 4.00 (bound) |

Unlike the synthetic case, the warm start did not shorten the search and did not reach the same optimum: the two runs end in different lag basins, the warm run at the upper lag bound for two streams, and neither meets the tolerance. The R fit's lags (-2.5, -1.5, 0.0) were translated directly; the GP objective moved them elsewhere. Two caveats: the response configuration here is ad hoc (fixed width 1 s, lag bounds -8 to 4 s, a broad lag prior), and the warm-start translation needed one fix on real data (the R latent grid extends one row past the last observation with a NaN that the scale translation must ignore). The synthetic gain came from a well-specified generator with unimodal lags; on this recording the lag landscape is multimodal, so a warm start would need to seed several lag basins rather than one point.

## 4. Posterior: blocked Gibbs against NUTS

Identity responses everywhere and a fixed timescale, so that the prototype and the package target the same density; both started from the same single-start MAP.

### Reduced problem (1,387 parameters, 403 nodes)

NUTS through the package's `sample`: two sequential chains, dense mass matrix, 300 warmup and 300 draws, tree depth capped at 8, target acceptance 0.9, on the PRO 6000. Gibbs: two chains on the CPU, 500 warmup sweeps and 10,000 sweeps thinned by 5, the second chain from a perturbed start.

| Quantity | NUTS min ESS (R-hat) | Gibbs min ESS (R-hat) | Gibbs with shift move, min ESS (R-hat) |
| --- | ---: | ---: | ---: |
| Log-likelihood | 4 (1.51) | 2,902 (1.000) | 2,765 (1.000) |
| Noise variances (19) | 61 (1.04) | 3,244 (1.001) | 3,181 (1.001) |
| Loading norms (342) | 2 (2.48) | 585 (1.006) | 647 (1.005) |
| Offsets (342) | 3 (2.17) | 13 (1.18) | 3,192 (1.002) |
| Sampling time | 2,813 s | 227 s | 235 s |

NUTS spent 153,000 leapfrog steps, 255 in every one of the 600 iterations, and its two chains had not met after 47 minutes: this is the flat rotation direction seen at full scale in the synthetic campaign, now on 1,387 real parameters with a dense metric. A cross-sampler comparison is therefore inconclusive for most quantities; on the noise variances, the one block NUTS did mix, the pooled four-chain R-hat is 1.02 and the means agree within 0.23 posterior sd.

### The offset defect and its fix

Without the shift move, offsets mixed poorly on real data (minimum ESS 13 here and 5 on the full problem, R-hat 1.18 and 1.32), worse than on synthetic data. An offset and the mean of the latent path are nearly confounded within a run. The fix is a Gibbs move along that likelihood-invariant direction: shift every factor's latent value by a constant and subtract the corresponding loading-weighted amount from every offset. The likelihood is unchanged; the path prior (a quadratic form in the path) and the Gaussian offset prior make the shift's conditional Gaussian, so it is drawn exactly, at about 4 percent extra cost per sweep. Validation: with and without the move, the pooled four-chain R-hat is 1.000 on log-likelihood, noise and loading norms and the means agree within 0.02 posterior sd; the remaining offset disagreement (0.21 sd) is the unconverged no-shift chains.

### Full problem (2,787 parameters, 1,346 nodes)

Gibbs with the shift move, two chains on the CPU, 500 warmup and 5,000 sweeps thinned by 5, 44 ms per sweep:

| Quantity | Count | Min ESS | Median ESS | Max R-hat |
| --- | ---: | ---: | ---: | ---: |
| Log-likelihood | 1 | 1,961 | 1,961 | 1.001 |
| Noise variances | 19 | 1,730 | 1,933 | 1.004 |
| Loading norms | 692 | 1,145 | 1,842 | 1.007 |
| Offsets | 692 | 1,521 | 1,970 | 1.005 |

Total sampling time 445 s. The NUTS run on this problem (two vectorized chains, diagonal metric, 100 warmup and 100 draws on the PRO 6000) was still running when this note was written and will be added; at 255 steps per iteration and 0.13 s per gradient it is expected to take several hours and to remain unconverged at that budget.

## What this changes in the recommended order

1. **Blocked Gibbs is the item to build.** It is the only candidate that turned a posterior on this recording from unattainable into routine, it runs on the CPU, and its one real-data defect has an exact, cheap fix. The production version needs the shift move from the start, then learned response and timescale parameters (a marginal Metropolis block on the grouped Kalman likelihood), the K=1 anchor, masks (already exercised here through censored volumes), multi-run sharing and the package's diagnostics, orientation and persistence contracts.
2. **The grouped-node state-space backend holds on real clocks** and stays the route for long single runs; its CPU gain with Gaussian responses depends on the per-factor block propagation still to be written.
3. **The warm start needs a different design than a single translated point on data like this.** Seeding the lag coordinates across several basins, or fixing lags from the R fit for a first GP pass and releasing them afterwards, are the obvious variants to test next; the synthetic 6.5x should not be quoted for real data.
4. **Device choice is settled for short runs:** the PRO 6000 for dense grouped MAP, the CPU with 8 to 16 threads for the Gibbs sampler and the state-space likelihood.

## Evidence and limits

Scripts, result rows (`*.jsonl`) and draws (`*.npz`) are retained under ignored `local_data/empirical-eval-2026-09-22/`. No participant data or derived arrays are committed. All fits use ad hoc priors and response settings chosen for computational comparison, not for scientific inference; no claim is made about the recovered lags, loadings or latent structure. NUTS budgets were limited by wall time, so its results establish cost and convergence failure at those budgets, not the posterior it would eventually reach. The prototype sampler fixes response parameters and the timescale and lacks the anchor and multi-run contracts.
