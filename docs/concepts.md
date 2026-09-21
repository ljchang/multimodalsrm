# How the models work

## One shared response, several observation streams

Suppose several people observe the same movie. Brain measurements and behavioral ratings can reflect a shared time-varying signal while having different numbers of features, sampling rates, delays, and noise.

MultimodalSRM represents that signal using `K` latent factors. For each run, the default model shares exactly the same latent response across participants. Individual participant–modality mappings translate those factors into the measured feature space.

The observation model can be read schematically as:

```text
shared latent response for a run
    → modality response filter (smoothing and delay)
    → participant–modality feature mapping
    → predicted observations at the stream's native timestamps
```

Noise and the model's preprocessing or offset terms account for the remaining observation structure. This is a schematic description; the two model families use different objectives and uncertainty assumptions.

`features=K` selects the number of latent factors. It is an analysis choice, not an automatically inferred number of underlying biological processes. Factor coordinates also need conventions: rotations or sign changes can describe equivalent representations.

## R-MSRM: regularized estimation

`MultimodalSRM` represents the latent response on a grid with spacing `latent_dt`. It estimates latent values, participant–modality mappings, and any free response parameters through regularized optimization.

The regularization controls the fitted representation. Multiple initializations can explore different optimization outcomes, but a low objective alone does not establish convergence or correct response-filter recovery. Inspect `converged_` and the fitted diagnostics.

Once trained, the model can infer latent responses and predict excluded targets on new runs using learned mappings and response parameters. Separate calibration data can estimate a new participant's mapping. R-MSRM does not provide Bayesian posterior uncertainty.

Start with the [R tutorial](tutorials.md#r-msrm-fit-and-held-out-prediction).

## GP-MSRM: a continuous latent process

`BayesianMultimodalSRM` places a Gaussian-process prior on the shared latent response. Its `length_scale` controls the timescale of latent variation; it is distinct from a modality's response-filter width.

Observation covariances follow from the GP prior, response filters, feature mappings, and noise. The latent GP is marginalized when fitting the supported MAP and posterior parameter targets. Latent responses can then be inferred conditionally on observations and fitted or sampled parameters.

**MAP estimation** finds a parameter setting with high posterior density. Frozen conditioning holds those training parameters fixed when incorporating observations from a new run. Its predictive uncertainty is conditional on that MAP estimate and does not include parameter-posterior uncertainty.

**Posterior sampling** retains parameter draws for supported targets. Those draws can propagate parameter uncertainty into predictions and joint latent trajectories. Sampling needs its own convergence and diagnostic checks; it is not implied by using the Bayesian class or obtaining prediction intervals.

`linear_algebra` chooses the computational backend separately from MAP versus sampling. Dense and grouped algebra support the documented posterior workflows. State-space computation is an opt-in MAP backend with response-specific approximation constraints. Consult the [support matrix](capabilities.md) for exact combinations.

Start with the [GP MAP tutorial](tutorials.md#gp-msrm-map-and-fitted-archive-replay).

## Response filters and timing

An Identity response uses the latent signal directly. A Gaussian response can smooth and delay it. Other supported parametric families represent different response shapes. Parameters can be fixed or estimated subject to explicit bounds and supported pooling rules.

For example, fixing a Gaussian width while estimating lag asks a different question from estimating both. A broad latent GP timescale and a broad response filter can both smooth observations, making their separate interpretation difficult. A chosen reference or factor anchor supplies a coordinate convention; it does not establish physiological timing.

Learned FIR responses are deferred. The presence of an internal kernel type does not imply it supports every inference method.

## Evaluate the claim you want to make

Held-out prediction asks whether available observations predict excluded targets. Response recovery asks whether fitted filter parameters match a known generating mechanism. Convergence asks whether numerical optimization or sampling adequately explored the intended target. These require different evidence.

Use protected training/test splits, explicit target exclusion, appropriate fit diagnostics, and matched synthetic controls where recovery matters. The [capabilities guide](capabilities.md) describes interpretation boundaries for each workflow.
