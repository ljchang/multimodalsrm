# GP response and NVIDIA optimization research

This collection records the response-family and computational experiments
through September 23, 2026. It adds reproducible research scripts and aggregate
results without changing production inference defaults or supported Gaussian
orders. Empirical observations, loadings, and caches are excluded. The reviewed
movie-clock adapter requires the separately available historical local loader.

Start with the [clock and calibration audit](2026-09-23-gp-clock-calibration.md)
and the [repeat on the original movie clock](2026-09-23-gp-clock-preserved-fits.md).
The historical loader removed splice-marker rows and compressed the brain clock,
creating shifts up to 16 seconds. Independent pause markers support original row
times. The working preprocessing rule now retains the first 252 brain volumes,
trims the last eight, and keeps uncensored splice-marker observations. Earlier
empirical response estimates and predictive rankings use the historical clock
and are provisional; their numerical and timing checks still describe those inputs.
Use `--loader scripts/clock_preserving_emo_data.py` for new empirical comparisons.
Historical runners retain their recorded/default loader for reproducing older
measurements; saved-fit diagnostics restore the loader from the fit metadata.

The revised six-start comparison is complete. Five starts qualify; both selected
fits pass independent objective, gradient, and prediction checks. Brain and
ratings still lose to simple baselines. Longer latent timescales reduce brain
error, and private temporal residuals help rating predictions, but these are
fixed-parameter probes rather than refitted model rankings. The next step is a
training-selected timescale profile with full correlated-noise refits, together
with checks of response history around splice boundaries.

The earlier reports establish the computational baseline:

- [Three-factor, full-parcellation fits](2026-09-23-gp-expanded-response-fits.md):
  all six MAP fits converge and pass selected-fit numerical checks, but poor
  held-out brain/ratings predictions motivate the subsequent clock audit.
  These predictive results use the historical compressed clock.
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
fold is repeated with the revised loader; two further holdout schedules are
prepared but remain unrun.

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
