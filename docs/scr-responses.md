# Skin-conductance responses

Use `BatemanSCR` as the common response option for new skin-conductance models. The same `Response` configuration works in R-MSRM, dense/grouped GP MAP and posterior inference, and state-space GP MAP. `BachSCR` remains available for reproducing existing analyses and fitted archives.

This addition is available from the development checkout; it is not included in the published `0.1.0` release. Install the checkout as described in the [installation guide](getting-started.md).

## Configure a response

```python
from multimodalsrm import BatemanSCR, Identity, Response

responses = {
    "reference": Response(Identity(), estimate=False, pooling="shared"),
    "scr": Response(
        BatemanSCR(rise=0.7, decay=3.0, lag=0.0),
        pooling="shared",
        bounds={"rise": (0.3, 1.4), "decay": (1.5, 4.0), "lag": (-1.0, 1.0)},
    ),
}
```

`rise` and `decay` are positive time constants in seconds, and `lag` shifts the onset. They do not directly specify the time of the peak. Swapping the two time constants leaves the response unchanged; separated bounds identify their labels when both are learned. Equal constants are supported numerically and reduce to a shape-two Erlang response.

The untruncated unit-mass transfer function is `1 / ((1 + s*rise) * (1 + s*decay))`. This is the difference-of-exponentials SCR model used by [Benedek and Kaernbach (2010)](https://doi.org/10.1016/j.jneumeth.2010.04.028). The toolbox retains the same support convention as Bach: 90 seconds after lag, with **continuous finite L2 normalization**. The initial constants are starting values, not validated physiological defaults or a fitted conversion of canonical Bach.

## Use with R and GP models

Pass `responses` above to `MultimodalSRM(..., responses=responses)` for R-MSRM. Cell integration is vectorized and includes complex-step derivatives of the time constants, lag, finite normalizer and moving support endpoints. This requires no optional Bayesian dependencies.

GP models additionally require priors for free response parameters:

```python
import numpy as np
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior

model = BayesianMultimodalSRM(
    features=1,
    responses=responses,
    reference_modality="reference",
    priors=BayesianPriors(
        noise=Prior.lognormal(np.log(0.02), 0.7),  # observation variance
        filters={
            "scr": {
                "rise": Prior.lognormal(np.log(0.7), 0.35),
                "decay": Prior.lognormal(np.log(3.0), 0.35),
                "lag": Prior.normal(0.0, 0.5),
            }
        },
    ),
    length_scale=3.0,
    inference="map",
    linear_algebra="grouped",
    response_quadrature_order=64,
)
```

Dense/grouped MAP and posterior inference require an explicit `response_quadrature_order`. Compare orders over the chosen parameter bounds: the analytic `covariance_tolerance` does not certify quadrature accuracy. Other posterior requirements, including shared response pooling and factor anchors, still apply.

State-space MAP uses two exact exponential response states, plus two Matérn states per factor. It can learn rise, decay and lag. A finite cutoff cannot be represented by this finite-state filter: state-space inference restores the tail beyond 90 seconds and checks a uniform covariance error bound over the full parameter box. Slow decay or broad bounds may fail that check; use dense/grouped integration or explicitly justify a larger tolerance. No tolerance is relaxed automatically. Mixed Bateman/Gaussian models require `state_space_gaussian="auto"` or `"rational"`. State-space posterior sampling and Bateman spectral inference are unsupported.

The [SCR example](https://github.com/ljchang/multimodalsrm/blob/main/examples/scr_quickstart.py) uses one response configuration for R and GP fitting:

```sh
python examples/scr_quickstart.py
JAX_ENABLE_X64=true python examples/scr_quickstart.py --gp grouped
JAX_ENABLE_X64=true python examples/scr_quickstart.py --gp state_space
```

The observations are synthetic. The example checks fitting, valid prediction and archive replay; it does not establish physiological recovery or uncertainty calibration.

## Migrating a Bach analysis

Replace `Response(BachSCR(...), ...)` with an explicit `Response(BatemanSCR(...), ...)`, choose time-constant bounds and GP priors, and **refit the model**. There is no automatic conversion of Bach parameters, fitted loadings or lag estimates. Their reference shapes differ, so a difference in fitted `lag` alone cannot establish a change in physiological timing.

`BachSCR` continues to use its original formula, parameter names, defaults, normalization and archive identifier. Loading a Bach archive never substitutes Bateman. The generic model default remains Identity when no responses are supplied; the toolbox does not infer which streams contain skin conductance.

Numerical tests cover independent convolution integrals, finite normalization, equal and near-equal poles, cell-integral derivatives, rational transfer functions, likelihood/gradient/prediction agreement, posterior execution and archive replay. Before changing an established scientific analysis, compare held-out prediction, inferred latents and response timing on representative recordings. The initial synthetic comparison supported Bateman as the simpler baseline; neither matching Bach's shape nor a synthetic fit alone establishes empirical superiority.
