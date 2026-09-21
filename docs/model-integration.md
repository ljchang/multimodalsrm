# Model handoff and extraction

The source development repository remains authoritative until a coordinated handoff. Do not copy a moving Bayesian implementation into this repository and maintain two versions in parallel.

## Required handoff

Supply one verified merged source commit, its supported feature/backend matrix, dependency versions, test results, archive schemas and relevant numerical qualification reports. Record that full commit in `model-integration.json`. Preserve source and contributor provenance in this document when the transfer occurs. The current null commit explicitly means no source transfer has occurred.

## Extraction map

All source paths below are relative to the development repository. This is a mapping plan, not an automated recursive copy rule.

| Source | Destination / disposition |
| --- | --- |
| `src/personalized_srm/multimodal/estimator.py` and its actual dependency closure | `src/multimodalsrm/` |
| Shared data, response kernels/operators, fitting, inference, calibration, diagnostics, selection and evaluation modules | Preserve their internal module relationships under `src/multimodalsrm/` |
| `src/personalized_srm/multimodal/bayesian/` | `src/multimodalsrm/bayesian/`, selecting supported production modules |
| Relevant `tests/test_multimodal_*.py` | `tests/core/`, reviewed individually to exclude unrelated estimator tests |
| Relevant `tests/test_bayesian_*.py` and shared numerical fixtures | `tests/bayesian/`, with synthetic provenance checked |
| Source quickstarts, API guides and capability reports | Focused `examples/` and `docs/` after updating imports and claims |
| Top-level `personalized_srm` initializer/catalog and legacy model exports | Do not copy; use focused package exports |
| Neighborhood/graph/mixture/private-factor estimators, finite-grid and continuous references | Remain in the research repository |
| Empirical campaigns, local observations, fitted archives and Git LFS payloads | Remain local or in their existing authorized research location |

Inspect transitive imports before excluding historical helpers. Rewrite internal package paths without changing objectives, kernels, priors, bounds, pooling, preprocessing, held-out target exclusion or participant isolation. Model imports should be explicit and Bayesian dependencies optional. The intended APIs are `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM`; they are not available in the scaffold.

## Compatibility acceptance

1. Compare old and new synthetic fit objectives, response operators, predictions and diagnostics with declared numerical tolerances.
2. Exercise the full supported train, infer, predict, calibrate, update and save/load workflow for each model; retain intentional differences between the model APIs.
3. Verify target exclusion before preprocessing and isolation of each participant's independent observations.
4. Test all supported GP archive schemas against the renamed package without refitting. Inventory R pickle/joblib consumers and document an explicit conversion or old-environment replay route.
5. Run every migrated core and Bayesian suite against an installed wheel, not a source-path shortcut. Run synthetic examples from the installed package.
6. Update the provisional dependency pins to the versions qualified by the handoff. Remove optional backends that are not part of the supported release surface.
7. Change the integration manifest to `integrated` and populate both suite lists with required test files only when they have actually been transferred. CI runs the entire corresponding directories; the manifest's named tests are mandatory anchors, not a replacement for test discovery.

A manifest flag cannot certify scientific recovery or empirical validity. Record those evidence boundaries in user-facing model documentation. Long research campaigns remain separate from routine CI. FIR remains deferred.

After integration, new core development belongs here. Research campaigns should consume a pinned released package or source commit. Avoid overlapping installed packages that both own `personalized_srm`.
