# Model integration and provenance

The production implementation is extracted from [shared-response-models](https://github.com/ljchang/shared-response-models) at the frozen merged commit [`67128c7a562cab67499bdc0604357731a628aa9e`](https://github.com/ljchang/shared-response-models/tree/67128c7a562cab67499bdc0604357731a628aa9e). The source commit, transferred suites and integration state are recorded in [`model-integration.json`](https://github.com/ljchang/multimodalsrm/blob/main/model-integration.json). This handoff is the implementation basis for version `0.1.0`. The transfer evidence below is distinct from release CI and registry publication.

## Scope

| Source | Package destination / disposition |
| --- | --- |
| `src/personalized_srm/multimodal/estimator.py` and its production dependency closure | `src/multimodalsrm/` |
| Data, response kernels/operators, fitting, inference, calibration, diagnostics, selection and evaluation | Internal module relationships retained under `src/multimodalsrm/` |
| Supported `src/personalized_srm/multimodal/bayesian/` production modules, including warmup checkpoints | `src/multimodalsrm/bayesian/` |
| Relevant synthetic `test_multimodal_*.py` and `test_bayesian_*.py` suites and fixtures | Reviewed core/Bayesian package suites |
| User-facing workflow guidance | Focused `examples/` and `docs/` with updated imports and bounded claims |
| Top-level `personalized_srm` catalog and legacy neighborhood/graph/mixture/private-factor/reference estimators | Not exported by the new package; retained in the research repository |
| Empirical campaigns, observations, fitted archives and Git LFS payloads | Not distributed |

Internal import paths change without intentionally changing objectives, kernels, priors, bounds, pooling, preprocessing, held-out target exclusion or participant isolation. Public entry points are `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM`. Bayesian numerical dependencies remain optional. Some internal support modules retain historical names and capabilities; their presence does not expand the public [support matrix](capabilities.md).

## Contributor provenance

The transferred implementation was developed in Luke J. Chang's `shared-response-models` repository. The frozen source history attributes commits under `src/personalized_srm/multimodal/` to `ljchang`; retain the source repository and full commit when citing or auditing the transfer. Existing source notices and the package MIT license remain applicable. This provenance does not invent a publication, DOI or independent scientific endorsement.

## Acceptance evidence

The extraction must be checked against the frozen source, including synthetic objectives, response operators, predictions and diagnostics; supported train/infer/predict/calibrate/update/persistence workflows; target exclusion; and participant isolation. Core and Bayesian suites must run against an installed artifact, and the examples must run from that installation. GP archive schemas require old-to-new replay coverage. R pickle/joblib and warmup compatibility have narrower contracts described in [migration](migration.md).

The [extraction validation record](extraction-validation.json) reports the executed macOS/Python 3.12 checks: 1,519 unique model tests passed (251 core and 1,268 Bayesian), with no failures or skips. The core suite also passed without JAX or NumPyro installed. Thirty infrastructure tests passed separately. Wheel and source-distribution quickstarts passed; source-generated Bayesian archives for schemas 1–4 replayed exactly, including calibrated predictions and seeded posterior trajectories. Fresh old/new numerical comparisons matched, excluding elapsed-time records and identities incorporating them. Hosted platform results belong to the pull request checks.

The [source manifest](https://github.com/ljchang/multimodalsrm/blob/main/model-source.json) records hashes for 79 transferred implementation modules; [test migration](test-migration.json) records every source test disposition and the test-only reference helpers. Independent review confirmed unchanged production ASTs and imports after permitted formatting and package-name messages. A populated manifest or a passing test suite alone does not certify convergence, uncertainty calibration, timing/filter recovery or empirical validity. Long research campaigns and their frozen environments remain separate from routine CI.

New production-model changes belong here after handoff. Research consumers should pin a package commit or released version and record their runtime. Preserve the frozen source environment when replaying historical experiments; do not silently substitute the package into an old research protocol.
