"""Research configurations for the EmotionPictures response-family comparison.

The fixed double-gamma brain response defines a model clock before the HRF.
It does not establish absolute physiological timing. Priors and bounds below
are exploratory settings, not validated physiological defaults.
"""

import numpy as np

from multimodalsrm import BatemanSCR, DoubleGamma, Gamma, Gaussian, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig


def response_config(*, family="gamma", face_shape=3, rating_shape=3, other_physio=False):
    """Return matched candidate families with the same brain/EDA responses.

    Keep integer gamma shapes fixed within a fit so that state-space inference
    can learn scale and lag. Change shape between fits, not through a continuous
    optimizer. Shape candidates start with the same two-second peak latency.
    """
    if family not in ("gamma", "gaussian"):
        raise ValueError("family must be gamma or gaussian")
    for shape in (face_shape, rating_shape):
        if (
            isinstance(shape, bool)
            or not isinstance(shape, (int, np.integer))
            or not 2 <= shape <= 12
        ):
            raise ValueError("gamma shape candidates must be integers from 2 to 12")
    responses = {
        "brain": Response(DoubleGamma(), estimate=False, pooling="shared"),
        "eda": Response(
            BatemanSCR(rise=0.7, decay=3.0, lag=0.0),
            pooling="shared",
            bounds={"rise": (0.3, 1.4), "decay": (1.5, 4.0), "lag": (-2.0, 6.0)},
        ),
    }
    filters = {
        "eda": {
            "rise": Prior.lognormal(np.log(0.7), 0.35),
            "decay": Prior.lognormal(np.log(3.0), 0.35),
            "lag": Prior.normal(0.0, 2.0),
        }
    }
    for modality, shape in (("face", face_shape), ("rating", rating_shape)):
        if family == "gamma":
            scale = 2.0 / (shape - 1)
            responses[modality] = Response(
                Gamma(shape=shape, scale=scale, lag=0.0),
                fixed={"shape": shape},
                pooling="shared",
                bounds={"scale": (0.3 * scale, 2.0 * scale), "lag": (-2.0, 6.0)},
            )
            filters[modality] = {
                "scale": Prior.lognormal(np.log(scale), 0.4),
                "lag": Prior.normal(0.0, 2.0),
            }
        else:
            responses[modality] = Response(
                Gaussian(width=1.0, lag=2.0),
                pooling="shared",
                bounds={"width": (0.3, 2.0), "lag": (-2.0, 8.0)},
            )
            filters[modality] = {
                "width": Prior.lognormal(0.0, 0.4),
                "lag": Prior.normal(2.0, 2.0),
            }
    if other_physio:
        for modality in ("pulse", "respiration"):
            responses[modality] = Response(
                Gaussian(width=1.0, lag=2.0),
                fixed={"width": 1.0},
                pooling="shared",
                bounds={"lag": (-2.0, 8.0)},
            )
            filters[modality] = {"lag": Prior.normal(2.0, 2.0)}
    priors = BayesianPriors(
        loading_sd=1.5,
        offset_sd=1.0,
        noise=Prior.lognormal(np.log(0.5), 1.0),
        filters=filters,
    )
    return responses, priors


def split_physiology(data, *, other_physio=False):
    """Split the local loader's EDA/pulse/respiration columns without resampling.

    The original loader orders columns as EDA, pulse rate, respiration rate.
    Extract EDA by default. Preserve values, native binned times, and individual
    masks exactly. Pulse and respiration require other_physio=True.
    """
    result = {}
    for subject, runs in data.items():
        result[subject] = {}
        for run, modalities in runs.items():
            streams = dict(modalities)
            if "physio" in streams:
                physiology = streams.pop("physio")
                if physiology.values.shape[1] != 3:
                    raise ValueError("expected EDA, pulse rate, respiration rate columns")
                for i, name in enumerate(("eda", "pulse", "respiration")):
                    if i and not other_physio:
                        continue
                    if name in streams:
                        raise ValueError(f"physiology split would overwrite {name}")
                    streams[name] = TimeSeries(
                        physiology.values[:, i : i + 1],
                        physiology.times,
                        physiology.mask[:, i : i + 1],
                    )
            result[subject][run] = streams
    return result


def candidate_model(
    data,
    *,
    features=3,
    family="gamma",
    face_shape=3,
    rating_shape=3,
    other_physio=False,
    algebra="state_space",
    maxiter=300,
    starts=1,
):
    """Build a MAP research candidate, pruning responses absent from the data."""
    responses, priors = response_config(
        family=family, face_shape=face_shape, rating_shape=rating_shape, other_physio=other_physio
    )
    present = {m for runs in data.values() for streams in runs.values() for m in streams}
    unknown = present - responses.keys()
    if unknown:
        raise ValueError(f"unconfigured modalities: {sorted(unknown)}; split physiology first")
    responses = {m: response for m, response in responses.items() if m in present}
    priors = BayesianPriors(
        loading_sd=priors.loading_sd,
        offset_sd=priors.offset_sd,
        noise=priors.noise,
        filters={m: p for m, p in priors.filters.items() if m in present},
    )
    # Brain lag is fixed at zero; a filtered reference does not require Identity.
    return BayesianMultimodalSRM(
        features=features,
        responses=responses,
        priors=priors,
        reference_modality="brain",
        inference="map",
        linear_algebra=algebra,
        response_quadrature_order=None if algebra == "state_space" else 96,
        length_scale=3.0,
        # Explicit research tolerance: the full-box restored-tail bound for
        # this configuration is ~2.22e-7, above the package default of 1e-7.
        # Validate against grouped quadrature before interpreting fitted results.
        covariance_tolerance=1e-6,
        max_observations=100_000_000,
        search=SearchConfig(starts=starts, maxiter=maxiter),
        random_state=722,
    )
