"""Small synthetic GP MAP fit, frozen prediction and fitted-archive replay.

Install multimodalsrm[bayesian] and set JAX_ENABLE_X64=true before running.
The example does not run MCMC or claim uncertainty calibration.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.workflow import load_model, save_model


def recording(seed, run):
    """Smooth synthetic data in supplied units with unequal modality clocks."""
    reference_times = np.arange(0, 16.01, 0.8)
    signal_times = np.arange(0, 16.01, 1.0)
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    omega, width, lag = 2 * np.pi / 8, 0.3, 0.4
    reference = np.sin(omega * reference_times + phase)
    mass = np.sqrt(2) * np.pi**0.25 * np.sqrt(width)
    signal = mass * np.exp(-0.5 * (width * omega) ** 2)
    signal = signal * np.sin(omega * (signal_times - lag) + phase)
    data = {}
    for index, participant in enumerate(["a", "b"]):
        rng = np.random.default_rng(100 * seed + index)
        ref_values = (1 + 0.2 * index) * reference[:, None]
        ref_values += rng.normal(0, 0.1, ref_values.shape)
        signal_values = (0.8 + 0.2 * index) * signal[:, None]
        signal_values += rng.normal(0, 0.1, signal_values.shape)
        mask = np.ones_like(signal_values, dtype=bool)
        mask[7] = False
        data[participant] = {
            run: {
                "reference": TimeSeries(ref_values, reference_times),
                "signal": TimeSeries(signal_values, signal_times, mask),
            }
        }
    return data


def predict(model, donors):
    # Named target payloads are removed before validating/preparing donors.
    conditioned = model.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    return conditioned.predict(times={"new": [4.0, 8.0, 12.0]}, include_noise=True)["b"]["new"][
        "signal"
    ]


def main():
    model = BayesianMultimodalSRM(
        features=1,
        anchor=("a", "reference", 0),
        responses={
            "reference": Response(Identity(), estimate=False, pooling="shared"),
            "signal": Response(
                Gaussian(width=0.3, lag=0.4),
                estimate=True,
                fixed={"width": 0.3},
                bounds={"lag": (-0.2, 1.0)},
                pooling="shared",
            ),
        },
        priors=BayesianPriors(
            loading_sd=1.0,
            offset_sd=0.3,
            # This prior describes observation variance, not standard deviation.
            noise=Prior.lognormal(np.log(0.02), 0.7),
            filters={"signal": {"lag": Prior.normal(0.4, 0.3)}},
        ),
        length_scale=2.0,
        inference="map",
        linear_algebra="grouped",
        random_state=31,
        search=SearchConfig(starts=1, maxiter=500),
    ).fit(recording(31, "train"))
    donors = recording(32, "new")
    expected = predict(model, donors)
    assert expected.valid.all()
    assert np.isfinite(expected.values).all()
    with TemporaryDirectory(prefix="multimodalsrm-map-") as directory:
        destination = Path(directory) / "model"
        save_model(destination, model)
        restored, standardizer = load_model(destination)
        assert standardizer is None  # No external standardization in this example.
        actual = predict(restored, donors)
        np.testing.assert_array_equal(actual.valid, expected.valid)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.variance, expected.variance, rtol=1e-12, atol=1e-12)
    print("MAP diagnostics:", model.map_diagnostics_)
    print("Prediction shape:", expected.values.shape, "valid times:", expected.valid.sum())
    print("Archive replay matched; uncertainty is conditional on training MAP.")


if __name__ == "__main__":
    main()
