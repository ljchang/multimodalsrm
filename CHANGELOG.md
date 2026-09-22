# Changelog

## Unreleased

- Group fixed-response state-space filtering and smoothing at exact modality/time nodes for strictly positive noise priors, preserving scalar updates for learned responses and noiseless support.
- Add BatemanSCR as the common SCR response option for R and GP workflows, including differentiable state-space MAP, finite normalization, and explicit tail bounds. Preserve BachSCR parameters and archive behavior; include a shared R/GP example and numerical qualification tests.

## 0.1.0

- Extract R-MSRM and GP-MSRM production modules from the frozen source handoff, retaining model objectives, response conventions, priors and numerical behavior.
- Expose `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM` with optional Bayesian dependencies.
- Transfer focused synthetic core/Bayesian tests and add small R fitting and GP MAP/archive-replay examples.
- Document supported posterior workflows, numerical backends, legacy archive compatibility and the source-bound warmup checkpoint contract.
- Retain Linux/macOS installed-wheel and installed-sdist CI, integration/version gates and release-driven Trusted Publishing infrastructure.

- Add a searchable Zensical documentation site with GitHub Pages deployment.
- Separate fast PR checks from the full Linux/macOS release suite.

This is the first package release of the extracted models. Test evidence belongs to the integration record and corresponding CI run; this changelog makes no calibration or recovery claim. Learned FIR responses remain deferred.
