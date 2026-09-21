"""Convolved filtering and prediction must agree with finite-response GP inference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import (
    DoubleGamma,
    Gamma,
    Identity,
    Response,
    TimeSeries,
)

from .test_bayesian_problem import api


def fixture(algebra="state_space", features=1, order=48, two_runs=False):
    from multimodalsrm.bayesian.persistence import _prepare

    b = api()
    responses = {
        m: Response(k, estimate=False, pooling="shared")
        for m, k in {
            "ref": Identity(),
            "brain": DoubleGamma(3, 0.7, 7, 1.1, 0.3, 1.25),
            "aux": Gamma(3, 0.7, -0.35),
        }.items()
    }
    model = b.BayesianMultimodalSRM(
        priors=b.BayesianPriors(noise=b.Prior.lognormal(-1.0, 0.5)),
        anchor=("a", "ref", 0),
        reference_modality="ref",
        features=features,
        factor_anchors=(("a", "ref", 0), ("a", "brain", 0)) if features == 2 else None,
        responses=responses,
        inference="map",
        linear_algebra=algebra,
        response_quadrature_order=order if algebra != "state_space" else None,
        length_scale=3.0,
        covariance_tolerance=1e-6,
        random_state=731,
        search=b.SearchConfig(starts=2, maxiter=500),
    )
    rng = np.random.default_rng(722)
    data = {}
    for subject in ("a", "b"):
        data[subject] = {}
        for run in ("train", "second") if two_runs else ("train",):
            data[subject][run] = {}
            for m in responses:
                if subject == "b" and run == "second" and m == "aux":
                    continue
                times = np.array(
                    [
                        0.0,
                        50.0,
                        50.00000001,
                        52.0,
                        54.0,
                        57.0,
                        61.0,
                        65.0,
                        70.0,
                        80.0,
                        90.0,
                    ]
                )
                if m == "aux":
                    times = times[::2] - 1.6  # Some effective times match brain.
                values = np.column_stack([np.sin(times / 5), np.cos(times / 7)]) + rng.normal(
                    0, 0.1, (len(times), 2)
                )
                mask = np.ones(values.shape, bool)
                mask[2::5, 1] = False
                data[subject][run][m] = TimeSeries(values, times, mask)
    _, problem = _prepare(model, data)
    x = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            x[i] = 0.8 if name[1] == "a" else -0.4 + 0.1 * rng.normal()
        elif name[0] == "noise":
            x[i] = 0.3
    return model, problem, x, data


@pytest.mark.parametrize("features", [1, 2])
def test_convolved_objective_gradients_and_dense_diagnostic(features):
    _, state, x, _ = fixture(features=features, two_runs=True)
    _, dense, _, _ = fixture("dense", features, two_runs=True)
    _, fine, _, _ = fixture("dense", features, order=96, two_runs=True)
    reference, reference_gradient = fine.value_gradient(x)
    coarse, coarse_gradient = dense.value_gradient(x)
    assert_allclose(coarse, reference, atol=2e-6, rtol=1e-8)
    assert_allclose(coarse_gradient, reference_gradient, atol=1e-5, rtol=1e-6)
    value, gradient = state.value_gradient(x)
    assert_allclose(value, reference, atol=2e-6, rtol=1e-8)
    assert_allclose(gradient, reference_gradient, atol=1e-5, rtol=1e-6)
    assert_allclose(state.covariance(x, "train"), fine.covariance(x, "train"), atol=2e-7)


@pytest.mark.parametrize("key", [None, ("b", "brain", 1), ("a", "aux", 0)])
def test_convolved_smoothing_matches_finite_gp_without_dense_fallback(key, monkeypatch):
    from multimodalsrm.bayesian.prediction import project

    _, state, x, _ = fixture(features=2)
    _, dense, _, _ = fixture("dense", features=2, order=96)
    query = np.array([-4.0, 0.0, 50.0, 50.00000001, 52.25, 58.4, 72.0, 95.0])
    kwargs = dict(key=key, include_noise=True, latent_loading=[0.6, -0.8] if key is None else None)
    expected = project(dense, x[None], "train", query, **kwargs)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense fallback in convolved state-space inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    assert np.isfinite(state.value_gradient(x)[0])
    actual = project(state, x[None], "train", query, **kwargs)
    assert_allclose(actual, expected, atol=2e-6, rtol=1e-6)


def test_public_fit_target_exclusion_independent_transform_and_archive(tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    state, _, _, data = fixture()
    dense, _, _, _ = fixture("grouped", order=64)
    state.fit(data)
    dense.fit(data)
    assert_allclose(state.objective_, dense.objective_, atol=2e-5)
    donors = {s: {"held": runs["train"].copy()} for s, runs in data.items()}
    query = np.array([51.0, 55.0, 60.0, 65.0, 70.0])
    actual = state.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(times=query)
    expected = dense.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(times=query)
    approximation = actual["b"]["held"]["brain"].metadata["covariance_approximation"]
    assert approximation["response_approximation"] == "restore_gamma_tails"
    assert approximation["temporal_covariance_tail_bound"] > 0
    assert_allclose(
        actual["b"]["held"]["brain"].values,
        expected["b"]["held"]["brain"].values,
        atol=2e-4,
    )
    before = state.transform(donors, times=query)
    ts = donors["b"]["held"]["ref"]
    donors["b"]["held"]["ref"] = TimeSeries(ts.values * -5, ts.times)
    after = state.transform(donors, times=query)
    assert_allclose(before["a"]["held"].values, after["a"]["held"].values, atol=0, rtol=0)
    assert not np.allclose(before["b"]["held"].values, after["b"]["held"].values)
    donors["b"]["held"]["brain"] = object()
    expected = state.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(times=query)
    save_model(tmp_path / "model", state)
    restored, _ = load_model(tmp_path / "model")
    actual = restored.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(
        times=query
    )
    assert_allclose(
        actual["b"]["held"]["brain"].values,
        expected["b"]["held"]["brain"].values,
        atol=0,
        rtol=0,
    )
    metadata = restored.configuration_["state_space"]
    assert metadata["approximation"]
    assert 0 < metadata["temporal_covariance_tail_bound"] < 1e-6


def test_convolved_noiseless_identity_observations_interpolate_at_tied_queries():
    from multimodalsrm.bayesian.persistence import _prepare
    from multimodalsrm.bayesian.prediction import project

    b = api()
    model = b.BayesianMultimodalSRM(
        priors=b.BayesianPriors(noise=b.Prior.uniform(0.0, 1.0)),
        anchor=("a", "ref", 0),
        reference_modality="ref",
        inference="map",
        linear_algebra="state_space",
        covariance_tolerance=1e-6,
        responses={
            "ref": Response(Identity(), pooling="shared", estimate=False),
            "signal": Response(Gamma(3, 0.7, 0.4), pooling="shared", estimate=False),
        },
    )
    times = np.array([0, 30, 32, 35, 40.0])
    values = np.sin(times / 4)[:, None]
    data = {"a": {"train": {m: TimeSeries(values, times) for m in ("ref", "signal")}}}
    _, state = _prepare(model, data)
    x = state.initial.copy()
    for i, name in enumerate(state.names):
        x[i] = (
            1.0
            if name[0] == "loading"
            else 0.0
            if name[0] == "offset" or name == ("noise", "a", "ref")
            else 0.2
        )
    actual = project(state, x[None], "train", times, key=("a", "ref", 0), include_noise=False)
    assert_allclose(actual[0][0], values[:, 0], atol=1e-10)
    assert_allclose(actual[1][0], 0, atol=1e-10)
