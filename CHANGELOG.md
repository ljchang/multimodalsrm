# Changelog

## Unreleased — 0.1.0.dev0

- Extract R-MSRM and GP-MSRM production modules from the frozen source handoff, retaining model objectives, response conventions, priors and numerical behavior.
- Expose `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM` with optional Bayesian dependencies.
- Transfer focused synthetic core/Bayesian tests and add small R fitting and GP MAP/archive-replay examples.
- Document supported posterior workflows, numerical backends, legacy archive compatibility and the source-bound warmup checkpoint contract.
- Retain Linux/macOS installed-wheel and installed-sdist CI, integration/version gates and release-driven Trusted Publishing infrastructure.

This remains an unreleased development version. No model implementation or PyPI package has been released from this repository. Test evidence belongs to the integration record and corresponding CI run; this changelog makes no calibration or recovery claim.
