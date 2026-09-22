"""One Bateman SCR configuration for R-MSRM and optional GP MAP fitting.

Base install: python examples/scr_quickstart.py
Bayesian extra and JAX_ENABLE_X64=true: add --gp grouped or --gp state_space.
Synthetic observations demonstrate usage and replay, not physiological recovery.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from multimodalsrm import BatemanSCR, Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.operators import observation_operator


def responses():
    return {
        "reference": Response(Identity(), estimate=False, pooling="shared"),
        "scr": Response(
            BatemanSCR(rise=0.7, decay=2.0, lag=0.2),
            pooling="shared",
            bounds={"rise": (0.3, 1.4), "decay": (1.5, 3.0), "lag": (-0.5, 0.8)},
        ),
    }


def recording(seed, run):
    rng = np.random.default_rng(seed)
    grid = np.arange(0, 145.01, 0.5)
    phase = rng.uniform(-np.pi, np.pi)
    latent = np.sin(grid / 3 + phase) + 0.4 * np.cos(grid / 7)
    reference_times = np.arange(0, 145.01, 2.0)
    scr_times = np.arange(0, 145.01, 2.5)
    reference = np.interp(reference_times, grid, latent)
    h, valid = observation_operator(grid, scr_times, BatemanSCR(0.7, 2.0, 0.2))
    signal = h @ latent
    data = {}
    for i, subject in enumerate(("a", "b")):
        ref_values = (1 + 0.2 * i) * reference[:, None]
        scr_values = (0.8 + 0.2 * i) * signal[:, None]
        data[subject] = {
            run: {
                "reference": TimeSeries(
                    ref_values + rng.normal(0, 0.1, ref_values.shape), reference_times
                ),
                "scr": TimeSeries(
                    scr_values + rng.normal(0, 0.1, scr_values.shape), scr_times, valid[:, None]
                ),
            }
        }
    return data


def gp_fit(algebra, response_config, train, donors):
    from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model = BayesianMultimodalSRM(
        features=1,
        responses=response_config,
        reference_modality="reference",
        anchor=("a", "reference", 0),
        priors=BayesianPriors(
            noise=Prior.lognormal(np.log(0.02), 0.7),
            filters={
                "scr": {
                    "rise": Prior.lognormal(np.log(0.7), 0.35),
                    "decay": Prior.lognormal(np.log(2), 0.35),
                    "lag": Prior.normal(0.2, 0.4),
                }
            },
        ),
        length_scale=3,
        inference="map",
        linear_algebra=algebra,
        response_quadrature_order=None if algebra == "state_space" else 64,
        search=SearchConfig(starts=1, maxiter=600),
        random_state=17,
    ).fit(train)

    def predict(fitted):
        return fitted.condition(donors, targets={"b": ["scr"]}, mode="frozen").predict(
            times={"new": [100.0, 110.0, 120.0, 130.0]}
        )["b"]["new"]["scr"]

    expected = predict(model)
    assert expected.valid.all() and np.isfinite(expected.values).all()
    with TemporaryDirectory(prefix="multimodalsrm-bateman-") as directory:
        path = Path(directory) / "model"
        save_model(path, model)
        restored, _ = load_model(path)
        actual = predict(restored)
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(actual.variance, expected.variance, atol=1e-10, rtol=1e-10)
    print(
        "GP backend:",
        algebra,
        "gradient tolerance met:",
        model.map_diagnostics_["meets_gradient_tolerance"],
    )
    print("GP Bateman archive replay matched.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp", choices=("dense", "grouped", "state_space"))
    args = parser.parse_args()
    response_config = responses()
    train, donors = recording(31, "train"), recording(32, "new")
    model = MultimodalSRM(
        features=1,
        latent_dt=0.5,
        responses=response_config,
        init="hybrid",
        max_iter=150,
        kernel_max_iter=30,
        random_state=17,
    ).fit(train)
    predicted = model.predict(
        donors, targets={"b": ["scr"]}, times={"new": [100.0, 110.0, 120.0, 130.0]}
    )["b"]["new"]["scr"]
    assert predicted.valid.all() and np.isfinite(predicted.values).all()
    print("R training converged:", model.converged_)
    print("R SCR parameters:", model.subject_kernels_["a"]["scr"].parameters)
    if args.gp:
        gp_fit(args.gp, response_config, train, donors)


if __name__ == "__main__":
    main()
