# Runnable synthetic tutorials

Install the package from this checkout as described in the [README](../README.md), then run the scripts below. They use fixed random seeds and small generated observations; no research data or external downloads are needed.

## R-MSRM: fit and held-out prediction

```sh
python examples/r_quickstart.py
```

The [R example](../examples/r_quickstart.py) generates two participants and two training runs with unequal brain/rating clocks. It fits one exact shared latent response per run, uses an Identity reference, and estimates Gaussian lag with width fixed via `estimate=True, fixed={"width": 0.3}`. A new run is used for target-excluded prediction and latent inference. It also calibrates a newcomer from a separate recording with fitted donors, then predicts that participant's independent next run.

The script prints shapes, valid observation counts, fitted response parameters and convergence status. Target observations are present in the synthetic input to demonstrate the API's explicit exclusion rule; they do not inform their predictions. R preprocessing is learned during fitting and reused. Inspect validity masks, especially near response-support boundaries. This is an API example, not a timing/filter recovery experiment or a general convergence guarantee.

## GP-MSRM: MAP and fitted archive replay

```sh
python -m pip install -e '.[bayesian]'
JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu python examples/gp_map_quickstart.py
```

The [GP example](../examples/gp_map_quickstart.py) creates two native-rate modalities for two participants, fits grouped MAP with a fixed-width/estimated-lag Gaussian response, and conditions a distinct new run with frozen training MAP parameters. It saves the model to a temporary archive, reloads it and checks that predictions and validity masks agree. The archive is removed automatically after the demonstration. Change the destination to a new persistent directory for a real analysis.

The script prints MAP diagnostics. Its conditional Gaussian prediction uncertainty excludes parameter-posterior uncertainty; no MCMC is run. It works in supplied observation units without a learned external standardizer. The Bayesian runtime must enable float64 before fitting. These small examples check usage and replay, not physiological recovery or interval coverage.

## Extending beyond the examples

Consult the [capability matrix](capabilities.md) for full posterior training, response quadrature, a learned shared GP timescale, joint donor updates, posterior participant calibration and joint latent trajectories. These methods need an appropriate sampling budget and per-fit diagnostic assessment; they are deliberately outside the short MAP demonstration. The [migration guide](migration.md) distinguishes fitted archives from source-bound warmup checkpoints and R pickle/joblib artifacts.
