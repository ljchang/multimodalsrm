# Changelog

## 0.1.0

- Extract R-MSRM and GP-MSRM production modules from the frozen source handoff, retaining model objectives, response conventions, priors and numerical behavior.
- Expose `multimodalsrm.MultimodalSRM` and `multimodalsrm.bayesian.BayesianMultimodalSRM` with optional Bayesian dependencies.
- Transfer focused synthetic core/Bayesian tests and add small R fitting and GP MAP/archive-replay examples.
- Document supported posterior workflows, numerical backends, legacy archive compatibility and the source-bound warmup checkpoint contract.
- Retain Linux/macOS installed-wheel and installed-sdist CI, integration/version gates and release-driven Trusted Publishing infrastructure.
- Compute rank-normalized split R-hat, bulk and 5%/95% tail ESS, Monte Carlo errors and BFMI in batched NumPy over every quantity at once, reproducing ArviZ's definitions to rounding; the diagnostics phase of a 57,758-quantity posterior drops from 106 s to under 1 s at 15 draws and from about 160 s to under 40 s at 1,200 draws. The summary is a small array-backed table instead of a pandas DataFrame, so ArviZ, xarray and pandas all leave the `bayesian` extra and ArviZ joins the `test` extra as the reference implementation.
- Warn once per process when the CPU symmetric factorization callback runs with a BLAS pool larger than half the physical cores, and document thread settings measured on a 64-core Linux workstation.

- Add a searchable Zensical documentation site with GitHub Pages deployment.
- Separate fast PR checks from the full Linux/macOS release suite.

This is the first package release of the extracted models. Test evidence belongs to the integration record and corresponding CI run; this changelog makes no calibration or recovery claim. Learned FIR responses remain deferred.
