# Modality-specific response candidates

The selected empirical investigation focuses on **brain, face, ratings, and EDA**:
double-gamma for brain, Bateman for EDA, and gamma candidates for face and ratings.
The implementation is in
[the research configuration](../../scripts/gp_response_candidates.py) and
[its preparation/evaluation/fitting runner](../../scripts/benchmark_gp_response_candidates.py).
These are exploratory settings, not new package-wide defaults.

| Stream | Response | Parameters in the initial comparison |
| --- | --- | --- |
| Brain | `DoubleGamma()` | Fixed initial HRF, including lag zero, as the model's timing reference |
| EDA | `BatemanSCR(0.7, 3.0, 0.0)` | Learn rise, decay, and lag; separate rise/decay bounds |
| Face | `Gamma(shape=3, scale=1, lag=0)` | Learn scale and lag; compare integer shapes between fits |
| Ratings | `Gamma(shape=3, scale=1, lag=0)` | Learn scale and lag independently of face |

EDA is extracted from the first column of the existing physiology stream.
Its values, timestamps, and feature-specific masks are preserved. The loader's
column names/order are checked before the split. Subjects without physiology
remain supported. Pulse and respiration are excluded by default in both
comparison arms; `--other-physio` explicitly restores the earlier six-modality
exploratory setup, with separate Gaussian pulse/respiration responses.

## Timing and parameter interpretation

The previous experiment used Identity brain and allowed other signals to lead
a BOLD-referenced latent. This configuration uses `reference_modality="brain"`
with fixed double-gamma lag zero. The latent clock is now defined before that
brain response. Previous fitted lag values and loading/noise parameters are
not reused. This remains a modeling convention, not evidence that absolute
neural or physiological timing has been identified.

Gamma lag specifies onset. Its peak occurs at `lag + (shape - 1)*scale`.
Candidate shapes begin at the same two-second peak by setting
`scale = 2/(shape - 1)`. The initial grid is shapes 2, 3, and 6, selectable
independently for face and ratings. Integer shapes remain fixed within a fit,
as required by the current state-space backend. Scale priors are lognormal;
lag priors are normal in seconds, with explicit finite response bounds.

Brain response shape is fixed for the first comparison to limit confounding
with latent dynamics and other responses. The GP length scale is also fixed
at three seconds. Learning brain scales/undershoot or the GP timescale belongs
in a later sensitivity analysis with suitable backend support.

## Reproduce

With a Bayesian installation and float64 enabled:

```sh
PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_gp_response_candidates.py \
  --source synthetic --features 1 --action evaluate --output /tmp/response-check.json

PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_gp_response_candidates.py \
  --source empirical --action prepare --output local_data/response-prepare.json

PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_gp_response_candidates.py \
  --source empirical --subjects s001,s002 --parcels 5 --window 180 --features 1 \
  --action fit --starts 2 --maxiter 500 --face-shape 3 --rating-shape 3 \
  --output local_data/response-fit.json
```

Empirical runs reuse the ignored `emo_data.py` loader, its cached standardization,
splice-volume assumption, masks, and existing physiology binning. Its pandas
and nibabel dependencies and local dataset must be available. The runner does
not redistribute participant data. Metadata and progress go to JSON, and
completed parameter vectors go to an adjacent NPZ file.

Use `--family gaussian` for the comparison arm. It changes only face and rating
to learned Gaussian width/lag; brain remains double-gamma and EDA remains
Bateman. Both arms use the same four modalities and noise grouping.

## Numerical checks and limits

The exploratory state-space tolerance is explicitly `1e-6`, compared with the
package default of `1e-7`. The response-tail bound across the declared parameter
box for the default shape-three candidates is approximately `2.22e-7`, so this configuration would fail the stricter
default. No failure automatically relaxes a tolerance. The grouped comparison
uses response quadrature order 96; convergence over the full parameter box still
needs checking.

Completed checks:

- Full four-modality empirical preparation: 2,762 parameters, 169,708 scalar
  observations, 1,172 modality/time nodes, K=3, and state dimension 78.
- Small synthetic fixture: two subjects, asynchronous streams, feature-specific
  missingness, 43 parameters, and 42 included observations. State-space and
  grouped quadrature use exactly the same initial parameter vector.
- Absolute objective difference: `2.09e-8`; maximum absolute gradient difference:
  `4.30e-7`; relative gradient L2 difference: `3.44e-8`.
- The default physiology split reproduces EDA's original values, mask, and
  timestamps exactly, excludes pulse/respiration, and leaves the source dictionary intact.

A subsequent [four-modality MAP baseline](2026-09-22-gp-map-baseline.md) now
passes the physical-gradient check from three starts on the small empirical
subset. Two starts reach the same best constrained solution; active face/rating
bounds still require sensitivity analysis. Its fitted-point response derivatives
and grouped/state-space comparisons also agree numerically.

The [bound-sensitivity follow-up](2026-09-22-gp-response-bound-sensitivity.md)
finds an interior response solution with wider gamma bounds on identical
observations. Nearby alternative timing modes and sensitivity to edge-row
selection remain reasons to require longer-window held-out comparisons.

The [completed held-out comparison](2026-09-22-gp-response-holdout.md) uses
matched FWHM/peak priors and training-only preprocessing. It finds little
predictive difference between the screened families, weak improvement over
simple baselines, and useful NVIDIA parallel-filter speedups.

The earlier six-modality run is retained as historical evidence in the
[results file](2026-09-22-gp-response-candidates-results.json):

- That real-data optimization check used two subjects, five brain parcels, the
  first 180 seconds, and K=1: 188 parameters and 7,320 observations. It completed
  100 iterations in 142 seconds. It reached the iteration limit, with physical
  projected gradient 22.97, so it **did not converge**. Face/rating scale estimates
  approached opposite bounds; these are provisional optimization outputs,
  not recovered physiological parameters or a selected family.

With four modalities, shapes 2/3/6 give 24/26/32 states per
factor. The Gaussian face/rating comparator gives 60. All these configurations
prepared successfully under the explicit tolerance. These state counts compare
the new configurations, which both include double-gamma brain and Bateman EDA;
they are not observed end-to-end speedup ratios against the historical model.

Raw research outputs are under `local_data/gp-response-candidates-2026-09-22/`.
These checks establish that the configuration is usable and agrees numerically
at the tested point. They do not establish posterior convergence or predictive
superiority of the gamma family.

For family selection, use common support-eligible training and test observations:
different response support envelopes can otherwise select different edge rows.
Redo preprocessing using training observations alone for held-out evaluation;
the historical loader standardizes the complete clip. Compare prediction and
peak/width recovery, and assess sensitivity to priors on interpretable peak and
width. The current Gaussian and gamma parameter priors are exploratory, not
identical induced priors on response shape. A difference in raw MAP objective
or fitted onset lag alone is not a fair family comparison.

The [Gaussian efficiency work](2026-09-22-gp-gaussian-followup.md) remains relevant
for the face/rating comparator. Parallel filtering
and a converged NUTS/Gibbs comparison remain separate implementation and inference
qualification steps.
