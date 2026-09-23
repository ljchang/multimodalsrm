# Gaussian responses: smaller states, parallel filtering, and alternative families

Follow-up to [the GPU optimization investigation](2026-09-22-gp-next-optimizations.md),
against `main` at `d066357`. These are research findings; production defaults and
supported response orders have not changed.

The user's subsequent modality choices are implemented in the
[response-candidate experiment](2026-09-22-gp-response-candidates.md): double-gamma
brain, Bateman EDA, and gamma candidates for face/ratings, with a Gaussian
comparison arm. That experiment supersedes the BOLD-Identity reference setup
described below as historical context.

The later [held-out response-family experiment](2026-09-22-gp-response-holdout.md)
obtains a Gaussian parallel warm-runtime result for a smaller one-factor,
60-state model: approximately 2x faster gradients on an RTX 3090, with agreement
at the fitted point. That result does not resolve the 150-state compilation
timeout documented below.

Prioritize a cheaper Gaussian likelihood and parallel filtering before choosing
between NUTS, Gibbs, and structured VI. The realistic NUTS chains reviewed so far
have not converged, so they cannot establish a reliable sampler speed ratio or
serve as a posterior reference. Gibbs remains a candidate, including at higher
observed feature counts, but its mixing must be measured there.

## What actually grows

With K latent factors and B distinct Gaussian response banks of order r, the
current state dimension is approximately K(2 + Br). The two extra states are
for the Matérn-3/2 latent process. Fixed identical widths already share banks;
different lags do not require separate banks. Independently learned widths
generally do.

More observed features increase data reduction and loading-update work, but
do not enlarge this state when they share a response. Conditional on a sampled
latent path, each feature's loading/offset update is a small K+1 dimensional
problem. This avoids a feature-by-feature dense covariance factorization;
it does not guarantee good Gibbs mixing as information increases. More latent
factors and more response banks directly enlarge the expensive filter matrices.

## Measured opportunity: reduce Gaussian order

[The reproducible order probe](../../scripts/qualify_gp_gaussian_orders.py) fits
the existing rational template at additional orders, without enabling those
orders in the backend. [Results and environment](2026-09-22-gp-gaussian-orders.json)
cover widths 0.5, 1, 2 seconds and Matérn length scales 1, 3, 10 seconds.

The probe weights response-transfer errors by the latent Matérn spectrum. This
measures the error after convolution with a GP, which can be much smaller than
a worst-case response-only error. All values below are maxima across those
nine combinations:

| Response order | Estimated normalized latent/response covariance error bound | Estimated relative response autocovariance error bound | K=3 state dimension, one bank |
| --- | ---: | ---: | ---: |
| 8 | 7.80e-3 | 7.01e-3 | 30 |
| 12 | 1.01e-4 | 9.05e-5 | 42 |
| 16 | 6.45e-7 | 5.80e-7 | 54 |
| 20 | 2.82e-9 | 3.10e-9 | 66 |
| 24 | 1.07e-9 | 1.41e-9 | 78 |

For one bank, replacing order 24 by 16 changes dimension 78 to 54, reducing
generic cubic matrix arithmetic by about 3.0 times and quadratic storage by
2.1 times. Order 12 would reduce cubic arithmetic by 6.4 times. These are
dimension-based estimates, not measured fitting speedups. The two-bank case
changes from dimension 150 to 102 at order 16.

Qualification limits:

- These are finite-band numerical estimates of spectral bounds, not rigorous
  certificates. The integrals stop at dimensionless frequency 10,000.
- The spectral reference is the full Gaussian with the production finite-support
  L2 normalizer. Production truncates at six standard deviations. That distinction
  particularly matters when interpreting errors near 1e-9.
- The sweep does not cover a continuous prior box, cross-covariances between
  differently filtered modalities, parameter derivatives, or posterior error.
- The existing conservative response-only L1 bound is 5.99e-6 at order 16
  and 7.33e-4 at order 12. These candidates cannot silently inherit the existing
  covariance-tolerance guarantee.
- Many features and low observation noise can amplify small covariance errors.
  Qualification should include the joint observation covariance relative to
  noise, objective values and gradients, response recovery, and predictions.

The spectral approach is related to the general idea of approximating GP
spectra for state-space inference in
[Karvonen and Särkkä (2016)](https://tskarvone.github.io/pdf/KarvonenSarkka2016-MLSP.pdf).
The particular response-order measurements above are our own experiment.

## Parallel filtering: compilation is a concrete target

[The staged compilation probe](../../scripts/benchmark_gp_gaussian_compile.py)
passes fixed transition arrays as explicit JIT operands. On the RTX PRO 6000,
the synthetic fixed-Gaussian case had 918 nodes, two different widths, K=3,
and state dimension 150. Preparation took 9.51 seconds and tracing/lowering
0.943 seconds. The process reached its 180-second budget **inside
`lowered.compile()`**, before any timed execution. Exit status was 124.

This identifies compilation as a substantial cost in this new probe. It does
not retrospectively identify the phase of every previous timeout, and it
provides no Gaussian parallel warm-runtime result. Passing arrays explicitly
alone did not make this case compile within the budget.

A separate float64 microbenchmark of 512 well-conditioned 150-by-150 batched
solves and their reverse-mode gradient compiled in 4.12 seconds and ran warm
in about 11.5 milliseconds. This excludes the model's conditioning and full
scan graph, but shows that the elementary solve alone is not intrinsically
a minutes-long operation. Raw stage outputs are retained under
`local_data/gp-gaussian-followup-2026-09-22/`.

Next implementation experiments:

1. Chunked temporal filtering: perform short sequential blocks on the GPU,
   combine block messages with a parallel prefix, then recover within-block
   results. Benchmark block size, total work, compilation, and memory together.
2. Reduce repeated full-state factorizations, reuse factors for transposed
   solves, and use rank-K observation structure. Preserve the existing
   high-SNR stability; an inexpensive subtractive covariance update needs
   independent numerical qualification.
3. Propagate cross-factor covariance in factor-sized blocks. Cross-factor
   posterior covariance must remain; it cannot be discarded as independent.
4. Specialize Gaussian **lag-only** learning. Currently a free Gaussian lag
   enters the general changing-response transition path even with fixed width.
   A fixed generator plus elapsed-time derivative can avoid that generic work.
   Preserve the internal shift `lag - 6*width` and exact derivatives at ties;
   simply switching the existing boolean flag would not be sufficient.

The underlying parallel filtering/smoothing construction is from
[Särkkä and García-Fernández](https://arxiv.org/abs/1905.13002). It can support
multiple inference algorithms, but the current public state-space backend
still does not expose posterior sampling. Integration and qualification remain.

A further research direction is a common fixed-pole response basis with
width-dependent output coefficients. Several widths might then share states
and fixed dynamics. We have not established the basis size needed across the
allowed width range; this is a proposal, not a demonstrated reduction.

## Other response families are well established

Response filters should be distinguished from the covariance kernel of the
latent GP. We can change a response family while retaining the Matérn latent GP.

| Application | Established response family | Computational implication here |
| --- | --- | --- |
| BOLD | Canonical double-gamma HRF, often with temporal/dispersion derivatives | Integer-gamma components have finite-state realizations; size depends on shapes and shared scales |
| Pupil dilation | Gamma/Erlang-shaped impulse response | Integer shape n requires n response states; published fitted noninteger shapes are not automatically supported by this finite-state path |
| Skin conductance | Biexponential/Bateman response | Two response states for each distinct rise/decay pair |
| Flexible response estimation | FIR, Fourier, or a small basis expansion | Flexibility does not itself ensure a small state or inexpensive GP inference |

Sources: [SPM's model specification documentation](https://www.fil.ion.ucl.ac.uk/spm/docs/manual/fmri_spec/fmri_spec/)
and [HRF implementation](https://github.com/spm/spm/blob/main/spm_hrf.m);
[Hoeks and Levelt (1993)](https://doi.org/10.3758/BF03204445) and
[McCloy et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5392052/) for pupil responses;
[Benedek and Kaernbach (2010)](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1469-8986.2009.00972.x)
for the SCR compartment model.

`Gamma`, `DoubleGamma`, and `BatemanSCR` already exist in the repository. The
untruncated integer-gamma/exponential responses have finite-state realizations;
the backend separately checks the error from restoring their truncated tails.
Gaussian responses require approximation. A Gaussian's two-sided symmetric
shape also differs from the onset and asymmetric recovery of these physiological
responses. Replacing the family requires a refit, and raw lag parameters are
not comparable: Gaussian lag locates its peak, whereas gamma/Bateman lag locates
onset. Compare peak or centroid timing and the full effective response instead.

The current empirical setup uses **BOLD as an Identity reference**, with Gaussian
responses for face, ratings, and a combined physiology modality. They therefore
describe timing and smoothing relative to a BOLD-referenced latent, rather than
direct neural-to-sensor impulse responses. A canonical physiological kernel
cannot simply be substituted with that interpretation unchanged. EDA, pulse
rate, and respiration rate also need separate response choices if modeled
physiologically; the current common physiology response does not distinguish them.

There are also cheap two-sided recursive approximations that retain a
Gaussian-like shape. The [Deriche implementation documented by Getreuer](https://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/group__deriche__gaussian.html)
uses causal and anticausal recursions of order 2–4. Those results concern
convolution of sampled signals, not this irregular-time, multi-output GP
likelihood. Adapting the idea would require a joint covariance construction,
correct boundaries and cross-modality covariances, and error qualification.
It is an interesting separate prototype, not a ready-made backend replacement.

## Decision experiment

Keep Gaussian as the baseline. First compare current order, qualified lower
order, and parallel/chunked implementations at identical parameters and
precision. Then compare Gaussian against a small, scientifically justified
gamma/exponential family using held-out prediction and response recovery.
Match priors on interpretable timing/width and preserve continuous L2 scaling.

Obtain a converged NUTS reference on a controlled small problem, then increase
observed feature count separately from latent factor count and response-bank
count. Compare posterior quantities and time to the same ESS/R-hat targets,
including warmup and compilation. The current results justify pursuing the
computational improvements; they do not yet select a general-purpose sampler.
