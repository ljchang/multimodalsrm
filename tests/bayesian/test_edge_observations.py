"""Scoring every in-run observation in the GP model."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.workflow import load_model, save_model


def recording(seed=0, duration=40.0, run="train"):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.03, 0.2, 8)
    amplitude, phase = rng.normal(size=8), rng.uniform(0, 6.28, 8)

    def latent(times):
        return (amplitude * np.sin(2 * np.pi * frequency * np.asarray(times)[:, None] + phase)).sum(
            -1
        )

    reference, signal = np.arange(0, duration, 1.0), np.arange(0.5, duration, 1.0)
    data = {}
    for index, subject in enumerate("ab"):
        maps = np.random.default_rng([7, index])
        local = np.random.default_rng([seed, index])
        data[subject] = {
            run: {
                "reference": TimeSeries(
                    latent(reference)[:, None] * maps.normal(size=(1, 2))
                    + local.normal(size=(len(reference), 2)) * 0.3,
                    reference,
                ),
                "signal": TimeSeries(
                    latent(signal - 1.0)[:, None] * maps.normal(size=(1, 2))
                    + local.normal(size=(len(signal), 2)) * 0.3,
                    signal,
                ),
            }
        }
    return data


def model(edge, algebra="grouped", **settings):
    options = dict(
        features=1,
        anchor=("a", "reference", 0),
        responses={
            "reference": Response(Identity(), estimate=False, pooling="shared"),
            "signal": Response(
                Gaussian(1.0, 1.0),
                pooling="shared",
                fixed={"width": 1.0},
                bounds={"lag": (-6.0, 8.0)},
            ),
        },
        priors=BayesianPriors(
            noise=Prior.lognormal(np.log(0.1), 1.0),
            loading_sd=2.0,
            filters={"signal": {"lag": Prior.normal(1.0, 3.0)}},
        ),
        length_scale=2.0,
        inference="map",
        linear_algebra=algebra,
        max_observations=5000,
        search=SearchConfig(starts=2, maxiter=400),
        random_state=0,
        edge_observations=edge,
    )
    options.update(settings)
    return BayesianMultimodalSRM(**options)


def test_wide_lag_bounds_discard_edge_observations_unless_included():
    data = recording()
    total = sum(ts.values.size for runs in data.values() for ts in runs["train"].values())
    _, excluded = _prepare(model("exclude"), data)
    _, included = _prepare(model("include"), data)
    assert len(included.systems["train"].times) == total
    # The signal envelope spans (-12, 14) s, so 26 of its 40 s are unusable.
    assert len(excluded.systems["train"].times) < 0.7 * total
    signal = [k for k in excluded.systems["train"].keys if k[1] == "signal"]
    assert len(signal) == 2 * 2 * 14


@pytest.fixture(scope="module")
def fits():
    data = recording()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {edge: model(edge).fit(data) for edge in ("exclude", "include")}


def test_including_edges_keeps_the_fitted_delay(fits):
    delays = {
        edge: fit.timing_report()["relative_peak_delay"]["signal"] for edge, fit in fits.items()
    }
    assert delays["include"] == pytest.approx(1.0, abs=0.25)
    assert delays["exclude"] == pytest.approx(delays["include"], abs=0.3)
    assert fits["include"].configuration_["edge_observations"] == "include"
    assert "edge_observations" not in fits["exclude"].configuration_
    assert "edge_observations" not in fits["exclude"].specification_


def test_included_edges_make_every_in_run_query_valid(fits):
    donors = recording(seed=3, run="new")
    times = donors["b"]["new"]["signal"].times
    valid = {}
    for edge, fit in fits.items():
        conditioned = fit.condition(donors, targets={"b": ["signal"]}, mode="frozen")
        result = conditioned.predict(times={"new": times})["b"]["new"]["signal"]
        valid[edge] = result.valid
        assert np.isfinite(result.values[result.valid]).all()
    assert valid["include"].all()
    assert valid["exclude"].sum() < 0.7 * len(times)


def test_archive_round_trip_preserves_the_edge_setting(fits, tmp_path: Path):
    donors = recording(seed=3, run="new")
    query = {"new": [5.0, 20.0, 35.0]}

    def predict(fit):
        conditioned = fit.condition(donors, targets={"b": ["signal"]}, mode="frozen")
        return conditioned.predict(times=query)["b"]["new"]["signal"]

    for edge, fit in fits.items():
        destination = tmp_path / edge
        save_model(destination, fit)
        restored, _ = load_model(destination)
        assert restored.edge_observations == edge
        assert restored.adapter_.edge_observations == edge
        expected, actual = predict(fit), predict(restored)
        np.testing.assert_array_equal(actual.valid, expected.valid)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-10, atol=1e-10)


def test_validation():
    with pytest.raises(ValueError, match="edge_observations"):
        model("pad").fit(recording())
    with pytest.raises(ValueError, match="dense/grouped"):
        model("include", algebra="state_space").fit(recording())
