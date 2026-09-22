# Changelog

## Unreleased — 0.1.0.dev0

- Extract R-MSRM and GP-MSRM production modules from the frozen source handoff, retaining model objectives, response conventions, priors and numerical behavior.
- Expose `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM` with optional Bayesian dependencies.
- Transfer focused synthetic core/Bayesian tests and add small R fitting and GP MAP/archive-replay examples.
- Document supported posterior workflows, numerical backends, legacy archive compatibility and the source-bound warmup checkpoint contract.
- Retain Linux/macOS installed-wheel and installed-sdist CI, integration/version gates and release-driven Trusted Publishing infrastructure.
- Warn once per process when the CPU symmetric factorization callback runs with a BLAS pool larger than half the physical cores, and document thread settings measured on a 64-core Linux workstation.
- Add the `bayesian-cuda` extra, which installs the pinned JAX CUDA plugin on Linux x86_64 and resolves to the CPU runtime elsewhere, and warn once per fit process when an NVIDIA driver is visible but JAX initialized its CPU backend, unless the platform was pinned through the environment.

This remains an unreleased development version. No model implementation or PyPI package has been released from this repository. Test evidence belongs to the integration record and corresponding CI run; this changelog makes no calibration or recovery claim.
