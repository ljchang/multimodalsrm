"""Synthetic archive fixture extracted from the frozen source benchmark; no runner or evidence data."""

from dataclasses import replace

import numpy as np

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    Prior,
    SamplerConfig,
    SearchConfig,
)

from .continuous_covariance import response_covariance

DATA_SEED = 202609191

MAP_SEED = 719

ANCHORS = (("a", "brain", 0), ("a", "brain", 1))

SEARCH = SearchConfig(starts=4, maxiter=600, refine_maxiter=200, conditioning="diagonal", n_jobs=1)

SAMPLER = SamplerConfig(
    chains=4,
    warmup=600,
    draws=1000,
    target_accept=0.95,
    max_tree_depth=9,
    chain_method="sequential",
    mass_matrix="diagonal",
)


def data_from_arrays(arrays):
    return {
        s: {
            "train": {
                m: TimeSeries(
                    arrays[f"{s}_{m}_values"],
                    arrays[f"{s}_{m}_times"],
                    arrays[f"{s}_{m}_mask"],
                )
                for m in ("brain", "aux")
            }
        }
        for s in ("a", "b")
    }


def dataset():
    """Independent SciPy response covariance; no fitting covariance is used."""
    times = {"brain": np.arange(0.0, 32.0, 2.0), "aux": np.arange(0.0, 32.0, 1.5)}
    kernels = {"brain": Identity(), "aux": Gaussian(0.45, 0.6)}
    covariance = np.block(
        [
            [
                response_covariance(
                    times[a], times[b], kernels[a], kernels[b], 3.0, tolerance=1e-10
                )
                for b in times
            ]
            for a in times
        ]
    )
    jitter = 0.0
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        jitter = 1e-10
        chol = np.linalg.cholesky(covariance + jitter * np.eye(len(covariance)))
    rng = np.random.default_rng(DATA_SEED)
    factors = chol @ rng.normal(size=(len(covariance), 2))
    loadings = {
        ("a", "brain"): [[1.0, 0.2], [0.1, 0.8]],
        ("a", "aux"): [[0.7, -0.4]],
        ("b", "brain"): [[0.7, -0.5], [-0.4, 0.9]],
        ("b", "aux"): [[0.3, 0.8]],
    }
    offsets = {
        ("a", "brain"): [0.1, -0.1],
        ("a", "aux"): [0.05],
        ("b", "brain"): [-0.05, 0.12],
        ("b", "aux"): [0.03],
    }
    arrays = {"generator_covariance": covariance, "generator_factors": factors}
    for (s, m), weights in loadings.items():
        z = factors[: len(times["brain"])] if m == "brain" else factors[len(times["brain"]) :]
        clean = z @ np.asarray(weights).T + offsets[s, m]
        values = clean + rng.normal(0.0, 0.3, clean.shape)
        mask = np.ones_like(values, dtype=bool)
        if (s, m) == ("a", "brain"):
            mask[5, 1] = False
        if (s, m) == ("b", "aux"):
            mask[9, 0] = False
        for field, value in dict(
            values=values,
            clean=clean,
            mask=mask,
            times=times[m],
            true_loadings=weights,
            true_offsets=offsets[s, m],
        ).items():
            arrays[f"{s}_{m}_{field}"] = np.asarray(value)
    return (
        data_from_arrays(arrays),
        arrays,
        dict(
            data_seed=DATA_SEED,
            generator_jitter=jitter,
            covariance_reference="continuous.covariance.response_covariance",
            latent_kernel="Matern32",
            length_scale=3.0,
            variance=1.0,
            noise_sd=0.3,
            factors=2,
            fixed_truth=True,
            simulation_based_calibration=False,
        ),
    )


def estimator(arm):
    algebra, metric = arm.split("_")
    return BayesianMultimodalSRM(
        features=2,
        factor_anchors=ANCHORS,
        anchor=ANCHORS[0],
        reference_modality="brain",
        responses={
            "brain": Response(Identity(), pooling="shared", estimate=False),
            "aux": Response(
                Gaussian(0.45, 0.6),
                pooling="shared",
                estimate=True,
                bounds={"width": (0.25, 0.8), "lag": (-0.3, 1.2)},
            ),
        },
        priors=BayesianPriors(
            loading_sd=1.2,
            offset_sd=0.3,
            noise=Prior.lognormal(np.log(0.09), 0.6),
            filters={
                "aux": {
                    "width": Prior.normal(0.45, 0.25),
                    "lag": Prior.normal(0.6, 0.5),
                }
            },
        ),
        inference="map",
        linear_algebra=algebra,
        length_scale=3.0,
        random_state=MAP_SEED,
        search=SEARCH,
        sampler=replace(SAMPLER, mass_matrix=metric),
    )
