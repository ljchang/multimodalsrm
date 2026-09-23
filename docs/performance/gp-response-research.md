# GP response and NVIDIA optimization research

This collection records the response-family and computational experiments
through September 23, 2026. It adds reproducible research scripts and aggregate
results without changing production inference defaults or supported Gaussian
orders. Empirical observations, loadings, and local data loaders are excluded.

Start with these four reports:

- [Three-factor, full-parcellation fits](2026-09-23-gp-expanded-response-fits.md):
  all six MAP fits converge and pass selected-fit numerical checks. Gamma
  predicts brain and ratings better than Gaussian, but both lose to the
  training-mean baseline on those modalities in this fold.
- [Block filtering, transition reuse, and smaller Gaussian banks](2026-09-23-gp-block-filter.md):
  substantial compilation/memory reductions, modest warm gains, and unresolved
  low-noise accuracy limits. Order 16 is promising at the saved MAP but not
  qualified across the parameter box.
- [Parallel filtering at higher factor counts](2026-09-22-gp-parallel-scaling.md):
  gains shrink from about 2.6x at three gamma factors to 1.15x at six factors.
  The prefix prototype fails the extreme low-noise derivative gate.
- [Held-out response-family comparison](2026-09-22-gp-response-holdout.md):
  all 12 small MAP fits converge, but prediction is generally weak relative to
  a training-mean baseline. The comparison does not select a response family.

The scientific candidates retain fixed DoubleGamma brain responses and learned
Bateman EDA responses, and compare gamma versus Gaussian face/rating responses.
Pulse and respiration are excluded. Physical FWHM/peak coordinates, common
eligible observations, training-only preprocessing, and independent numerical
checks make the comparisons explicit. The three-factor, 100-parcel original
fold is complete; two further holdout schedules are prepared but remain unrun.

Earlier reports document how the final protocol was reached:

- [Initial optimization experiments](2026-09-22-gp-next-optimizations.md)
- [Gaussian states and alternative response families](2026-09-22-gp-gaussian-followup.md)
- [Modality-specific candidate configurations](2026-09-22-gp-response-candidates.md)
- [Small MAP convergence baseline](2026-09-22-gp-map-baseline.md)
- [Response-bound sensitivity](2026-09-22-gp-response-bound-sensitivity.md)

Each report links its runner and machine-readable results. Later protocols
supersede earlier exploratory priors and support choices; raw objectives across
those protocols are not model rankings. Timing results describe the specified
devices, dimensions, and parameter points. They do not establish posterior
convergence or end-to-end sampler speedups.
