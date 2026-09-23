# GP response and NVIDIA optimization research

This collection records the response-family and computational experiments
through September 23, 2026, and the resulting grouped-prediction optimization.
The prediction path reuses covariance and factorizations across output features;
inference defaults and supported Gaussian orders are unchanged. Empirical
observations, loadings, and caches are excluded. The reviewed
movie-clock adapter requires the separately available historical local loader.

Start with the [production grouped-prediction optimization](2026-09-23-gp-grouped-prediction.md)
for the implemented speedup and matched before/after checks, and the
[temporal-noise refits and matched NVIDIA benchmarks](2026-09-23-gp-temporal-refits.md)
for the response/noise model comparison.
The Gaussian grouped gradient is 3.32 times faster than state-space on the same
RTX 3090 at the checked point. A separate complete gamma/OU MAP fit takes
40.0 minutes on the tested CPU configuration, 3.6 minutes on the RTX PRO 6000,
and 5.4 minutes on the RTX 3090, reaching the same objective. The report
separates those timing results from the unsuccessful predictive comparison.

The [clock and calibration audit](2026-09-23-gp-clock-calibration.md)
and [repeat on the original movie clock](2026-09-23-gp-clock-preserved-fits.md)
explain the input correction.
The historical loader removed splice-marker rows and compressed the brain clock,
creating shifts up to 16 seconds. Independent pause markers support original row
times. The working preprocessing rule now retains the first 252 brain volumes,
trims the last eight, and keeps uncensored splice-marker observations. Earlier
empirical response estimates and predictive rankings use the historical clock
and are provisional; their numerical and timing checks still describe those inputs.
Use `--loader scripts/clock_preserving_emo_data.py` for new empirical comparisons.
Historical runners retain their recorded/default loader for reproducing older
measurements; saved-fit diagnostics restore the loader from the fit metadata.

The subsequent 36-start refit comparison is complete. It reserves all three
block schedules from a common training set, profiles GP timescales 3/10/30
seconds, and compares independent versus fixed OU noise for both response
families. Twenty-eight starts qualify; all four training-selected models qualify
and use the three-second timescale. Full prediction refinement and production
API checks pass. Every model loses to the training mean for brain and face;
OU refits improve ratings but worsen brain predictions substantially. Rating
gains largely reflect private residual interpolation. Post-splice target flags
do not account for all the predictive difficulties.

These are additional blocked validations in the same previously explored
cohort, not an independent test cohort. The original six-start comparison and
its fixed-parameter probes remain available as earlier steps; their training
masks and standardization differ from the new refits. The next model experiment
should address private/shared variation, independent measurement noise, and
response history. The cached research scorer also identifies a concrete
production prediction optimization: reuse grouped covariance and factorizations
across features. That optimization is now implemented and benchmarked separately
in the grouped-prediction report linked above; it does not change these fits.

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
fold was repeated with the revised loader; all three schedules have now been
scored under the common reserved-block training mask.

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
