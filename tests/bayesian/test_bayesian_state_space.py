"""A dense Gaussian oracle checks the state-space model, not its recurrences."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from .test_bayesian_multifactor import fixture as multifactor_fixture
from .test_bayesian_problem import api, independent_parameters, problem_fixture


def state_problem(dense, **kwargs):
    return api().BayesianProblem(
        dense.adapter,
        dense.priors,
        anchor=dense.anchor,
        linear_algebra="state_space",
        **kwargs,
    )


@pytest.mark.parametrize("features", [1, 2, 3])
@pytest.mark.parametrize("zero", [False, True])
def test_innovations_match_dense_density_and_all_physical_gradients(features, zero):
    dense, x, _ = multifactor_fixture(features=features, gaussian=False)
    state = state_problem(dense)
    if zero:
        for i, name in enumerate(dense.names):
            if name[0] == "loading" and name[1] == "b":
                x[i] = 0.0
    value, gradient = state.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert_allclose(value, expected, atol=1e-8, rtol=1e-11)
    assert_allclose(gradient, expected_gradient, atol=1e-7, rtol=1e-8)


def test_runs_start_at_stationarity_and_match_independent_gaussian_density():
    dense, _, _ = problem_fixture(two_runs=True)
    state = state_problem(dense)
    x = independent_parameters(dense)
    parameters = dict(zip(dense.names, x))
    expected = 0.0
    for system in dense.systems.values():
        scaled = np.sqrt(3) * np.abs(system.times[:, None] - system.times) / 3.0
        weights = np.array([parameters[("loading", *k)] for k in system.keys])
        means = np.array([parameters[("offset", *k)] for k in system.keys])
        noise = np.array([parameters[("noise", *k[:2])] for k in system.keys])
        C = (1 + scaled) * np.exp(-scaled) * np.outer(weights, weights)
        C += np.diag(noise)
        expected -= multivariate_normal.logpdf(system.values, mean=means, cov=C)
    assert_allclose(state.nll(x), expected, atol=1e-10)


def test_high_signal_to_noise_preserves_gradients():
    dense, x, _ = multifactor_fixture(gaussian=False)
    state = state_problem(dense)
    for i, name in enumerate(dense.names):
        if name[0] == "noise":
            x[i] *= 1e-3
        elif name[0] == "loading":
            x[i] *= 5
    a, ga = state.value_gradient(x)
    b, gb = dense.value_gradient(x)
    assert_allclose(a, b, atol=1e-6, rtol=1e-8)
    assert_allclose(ga, gb, atol=1e-4, rtol=2e-6)


@pytest.mark.parametrize("option", ["quadrature", "baseline", "noise"])
def test_unsupported_observation_models_are_explicit(option):
    dense, _, _ = problem_fixture()
    kwargs = {
        "quadrature": {"response_quadrature_order": 16},
        "baseline": {"run_baseline_sd": {"ref": 0.2}},
        "noise": {"noise_timescales": {"ref": 0.5}},
    }[option]
    with pytest.raises(ValueError, match="state_space.*(fixed|Identity|quadrature|baseline|noise)"):
        state_problem(dense, **kwargs)


def test_likelihood_does_not_construct_dense_observation_covariance(monkeypatch):
    dense, _, _ = problem_fixture()
    state = state_problem(dense)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense covariance used by state-space inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    assert np.isfinite(state.value_gradient(independent_parameters(state))[0])


@pytest.mark.parametrize("features", [1, 2])
@pytest.mark.parametrize("observed,noisy", [(False, False), (True, False), (True, True)])
def test_smoothing_matches_dense_without_dense_fallback(features, observed, noisy, monkeypatch):
    from multimodalsrm.bayesian.prediction import project

    dense, x, _ = multifactor_fixture(features=features, gaussian=False)
    state = state_problem(dense)
    query = np.array([-8.0, 0.0, 1e-10, 1.2, 4.5, 16.0, 100.0])
    draws = np.stack([x, x.copy()])
    for i, name in enumerate(state.names):
        if name[0] == "loading" and name[1] == "b":
            draws[1, i] = 0.0
    kwargs = dict(key=("b", "aux", 1) if observed else None, include_noise=noisy)
    if features > 1 and not observed:
        kwargs["latent_loading"] = np.array([0.6, -0.8])
    expected = project(dense, draws, "train", query, **kwargs)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense covariance used by state-space smoothing")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    actual = project(state, draws, "train", query, **kwargs)
    assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_very_close_observations_and_large_gaps_preserve_inference():
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian.prediction import project

    original, adapter, data = problem_fixture()
    times = np.array([0.0, 1e-10, 1e-7, 0.01, 2.0, 2.00000001, 8.0, 20.0, 2000.0])
    for subject in data:
        for modality, ts in data[subject]["train"].items():
            data[subject]["train"][modality] = TimeSeries(ts.values, times)
    adapter._prepare(data)
    dense = api().BayesianProblem(adapter, original.priors, anchor=original.anchor)
    state = state_problem(dense)
    x = independent_parameters(state)
    assert_allclose(state.value_gradient(x)[1], dense.value_gradient(x)[1], atol=1e-8)
    query = np.array([0.0, 1e-8, 3.0, 100.0, 2001.0])
    actual = project(state, x[None], "train", query, key=None, include_noise=False)
    expected = project(dense, x[None], "train", query, key=None, include_noise=False)
    assert_allclose(actual, expected, atol=1e-9)


def identity_model(algebra, features=1):
    dense, _, data = problem_fixture()
    b = api()
    return b.BayesianMultimodalSRM(
        priors=dense.priors,
        anchor=dense.anchor,
        responses=dense.responses,
        features=features,
        factor_anchors=(dense.anchor, ("b", "signal", 0)) if features == 2 else None,
        inference="map",
        linear_algebra=algebra,
        random_state=541,
        search=b.SearchConfig(starts=2, maxiter=500),
    ), data


@pytest.mark.parametrize("features", [1, 2])
def test_public_fit_frozen_condition_transform_and_archive(features, tmp_path):
    from sklearn.base import clone

    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian.workflow import load_model, save_model

    state, data = identity_model("state_space", features)
    dense = clone(state).set_params(linear_algebra="dense")
    assert clone(state).linear_algebra == "state_space"
    state.fit(data)
    dense.fit(data)
    assert_allclose(state.objective_, dense.objective_, atol=1e-6, rtol=1e-8)
    query = np.array([0.0, 0.2, 4.0, 8.0, 19.9, 20.0])
    donors = {s: {"new": runs["train"].copy()} for s, runs in data.items()}
    # Predicted observations are independent of arbitrary latent factor rotation.
    a = state.condition(donors, targets={"b": ["signal"]}, mode="frozen").predict(times=query)
    b = dense.condition(donors, targets={"b": ["signal"]}, mode="frozen").predict(times=query)
    assert_allclose(a["b"]["new"]["signal"].values, b["b"]["new"]["signal"].values, atol=1e-4)
    assert_allclose(a["b"]["new"]["signal"].variance, b["b"]["new"]["signal"].variance, atol=1e-4)
    assert state.configuration_["state_space"]["state_dimension"] == 2 * features
    before = state.transform(donors, times=query)
    ts = donors["b"]["new"]["signal"]
    donors["b"]["new"]["signal"] = TimeSeries(ts.values * -8, ts.times)
    after = state.transform(donors, times=query)
    assert_allclose(before["a"]["new"].values, after["a"]["new"].values, atol=0, rtol=0)
    assert not np.allclose(before["b"]["new"].values, after["b"]["new"].values)
    # Poisoned targets must be removed before validation, including after reload.
    donors["b"]["new"]["signal"] = object()
    expected = state.condition(donors, targets={"b": ["signal"]}, mode="frozen").predict(
        times=query
    )
    save_model(tmp_path / "state", state)
    restored, _ = load_model(tmp_path / "state")
    actual = restored.condition(donors, targets={"b": ["signal"]}, mode="frozen").predict(
        times=query
    )
    assert_allclose(
        actual["b"]["new"]["signal"].values,
        expected["b"]["new"]["signal"].values,
        atol=0,
        rtol=0,
    )
    assert_allclose(
        actual["b"]["new"]["signal"].variance,
        expected["b"]["new"]["signal"].variance,
        atol=0,
        rtol=0,
    )


def test_state_space_parameter_sampling_is_explicitly_out_of_scope():
    model, data = identity_model("state_space")
    with pytest.raises(ValueError, match="state_space.*MAP"):
        model.set_params(inference="posterior").fit(data)


def test_noiseless_observations_allow_queries_at_identical_times():
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian.prediction import project

    _, adapter, _ = problem_fixture()
    adapter._prepare({"a": {"train": {"ref": TimeSeries([[0.0], [0.5], [1.0]], [0.0, 1.0, 2.0])}}})
    b = api()
    priors = b.BayesianPriors(noise=b.Prior.uniform(0.0, 1.0))
    dense = b.BayesianProblem(adapter, priors, anchor=("a", "ref", 0))
    state = state_problem(dense)
    x = np.array([1.0, 0.0, 0.0])
    assert_allclose(state.value_gradient(x)[1], dense.value_gradient(x)[1], atol=1e-8)
    q = np.array([0.0, 0.5, 1.0, 2.0])
    expected = project(dense, x[None], "train", q, key=None, include_noise=False)
    actual = project(state, x[None], "train", q, key=None, include_noise=False)
    assert_allclose(actual, expected, atol=1e-9)
    assert_allclose(actual[0][0, [0, 2, 3]], [0.0, 0.5, 1.0], atol=1e-12)
    assert_allclose(actual[1][0, [0, 2, 3]], 0.0, atol=1e-12)
