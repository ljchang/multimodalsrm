"""Learned delays must match a finite-response GP through event reordering."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Response

from .test_bayesian_problem import api
from .test_bayesian_state_space_responses import fixture as fixed_fixture


def fixture(algebra="state_space", features=1, two_runs=False, order=192):
    from multimodalsrm.bayesian.persistence import _prepare

    model, fixed, x, data = fixed_fixture(features=features, two_runs=two_runs)
    b = api()
    model.set_params(
        responses={
            m: r
            if m == "ref"
            else Response.lag_only(
                r.initial_kernel(), pooling="shared", bounds={"lag": (-2.0, 2.0)}
            )
            for m, r in model.responses.items()
        },
        priors=b.BayesianPriors(
            noise=b.Prior.lognormal(-1.0, 0.5),
            filters={m: {"lag": b.Prior.normal(0.0, 1.5)} for m in ("brain", "aux")},
        ),
        linear_algebra=algebra,
        response_quadrature_order=order if algebra != "state_space" else None,
    )
    _, problem = _prepare(model, data)
    parameters = dict(zip(fixed.names, x))
    parameters.update({("filter", "brain", "lag"): 1.25, ("filter", "aux", "lag"): -0.35})
    return model, problem, np.array([parameters[n] for n in problem.names]), data


@pytest.mark.parametrize("features", [1, 2])
def test_learned_lag_likelihood_gradients_and_diagnostic_cross_event_ties(features):
    _, state, x, _ = fixture(features=features, two_runs=True)
    _, dense, _, _ = fixture("dense", features=features, two_runs=True)
    lag = state.indices[("filter", "brain", "lag")]
    for value in (-0.8, 1.25 - 1e-7, 1.25, 1.25 + 1e-7):
        x[lag] = value
        actual, gradient = state.value_gradient(x)
        expected, expected_gradient = dense.value_gradient(x)
        assert_allclose(actual, expected, atol=3e-6, rtol=1e-8)
        assert_allclose(gradient, expected_gradient, atol=2e-5, rtol=2e-6)
    assert_allclose(state.covariance(x, "train"), dense.covariance(x, "train"), atol=2e-7)
    step = np.eye(len(x))[lag] * 1e-5
    finite = (float(state.objective(x + step)) - float(state.objective(x - step))) / 2e-5
    assert_allclose(gradient[lag], finite, atol=3e-6, rtol=1e-6)


@pytest.mark.parametrize("key", [None, ("b", "brain", 1), ("a", "aux", 0)])
def test_smoothing_uses_draw_specific_observation_and_query_delays(key, monkeypatch):
    from multimodalsrm.bayesian.prediction import project

    _, state, x, _ = fixture(features=2)
    _, dense, _, _ = fixture("dense", features=2)
    draws = np.stack([x, x.copy()])
    draws[1, state.indices[("filter", "brain", "lag")]] = -0.8
    draws[1, state.indices[("filter", "aux", "lag")]] = 1.4
    query = np.array([-4.0, 0.0, 50.0, 50.00000001, 52.25, 58.4, 72.0, 95.0])
    kwargs = dict(key=key, include_noise=True, latent_loading=[0.6, -0.8] if key is None else None)
    expected = project(dense, draws, "train", query, **kwargs)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense fallback in learned-delay state-space inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    assert np.isfinite(state.value_gradient(draws[1])[0])
    actual = project(state, draws, "train", query, **kwargs)
    assert_allclose(actual, expected, atol=2e-6, rtol=1e-6)


def test_public_learned_delay_fit_frozen_transform_and_archive(tmp_path):
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, _, _, data = fixture()
    model.fit(data)
    query = np.array([51.0, 55.0, 60.0, 65.0, 70.0])
    donors = {s: {"held": runs["train"].copy()} for s, runs in data.items()}
    before = model.transform(donors, times=query)
    ts = donors["b"]["held"]["ref"]
    donors["b"]["held"]["ref"] = TimeSeries(ts.values * -5, ts.times)
    after = model.transform(donors, times=query)
    assert_allclose(before["a"]["held"].values, after["a"]["held"].values, atol=0, rtol=0)
    assert not np.allclose(before["b"]["held"].values, after["b"]["held"].values)
    expected = model.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(times=query)
    donors["b"]["held"]["brain"] = object()
    poisoned = model.condition(donors, targets={"b": ["brain"]}, mode="frozen").predict(times=query)
    assert_allclose(
        expected["b"]["held"]["brain"].values,
        poisoned["b"]["held"]["brain"].values,
        atol=0,
        rtol=0,
    )
    save_model(tmp_path / "model", model)
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
    assert_allclose(
        actual["b"]["held"]["brain"].variance,
        expected["b"]["held"]["brain"].variance,
        atol=0,
        rtol=0,
    )
    metadata = actual["b"]["held"]["brain"].metadata["covariance_approximation"]
    assert metadata["delay_handling"] == "exact_learned_shift_of_event_times"
    assert metadata["learned_response_parameters"] == {"brain": ["lag"], "aux": ["lag"]}


@pytest.mark.parametrize("rotated_state", [False, True])
@pytest.mark.parametrize("noiseless", [False, True])
def test_smoother_lag_derivatives_at_parameter_dependent_ties(noiseless, rotated_state):
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.persistence import _prepare
    from multimodalsrm.bayesian.state_space import smoother

    jax, jnp, _, _ = runtime()
    model, problem, x, data = fixture()
    if noiseless:
        from dataclasses import replace

        b = api()
        model.set_params(priors=replace(model.priors, noise=b.Prior.uniform(0.0, 1.0)))
        # One exact measurement per timestamp; duplicate noiseless features
        # would make this an inconsistent/singular observation model.
        parameters = dict(zip(problem.names, x))
        ts = data["a"]["train"]["ref"]
        mask = ts.mask[:, :1].copy()
        mask[2] = False  # Avoid unresolved 1e-8 spacing in the dense oracle.
        data["a"]["train"]["ref"] = TimeSeries(ts.values[:, :1], ts.times, mask)
        _, problem = _prepare(model, data)
        x = np.array([parameters[n] for n in problem.names])
        x[problem.indices[("noise", "a", "ref")]] = 0.0
    if rotated_state:
        from dataclasses import replace

        from multimodalsrm.bayesian.state_space_delays import (
            transition_function,
        )

        # An orthogonal state-coordinate change preserves the physical model
        # and identity stationary covariance. It exposes roundoff at the nearly
        # coincident event after a noiseless observation on both CPU platforms.
        state = problem.response_state_space
        rotation = np.linalg.qr(
            np.random.default_rng(7).normal(size=(state.dimension, state.dimension))
        )[0]
        problem.response_state_space = replace(
            state,
            generator=rotation @ state.generator @ rotation.T,
            driving=rotation @ state.driving,
            outputs=state.outputs @ rotation.T,
        )
        problem.delay_transition = transition_function(problem.response_state_space)
    query = np.array([48.75, 50.0, 52.0])
    smooth = smoother(problem, "train", query)
    model.set_params(linear_algebra="dense", response_quadrature_order=192)
    _, dense = _prepare(model, data)
    lag = problem.indices[("filter", "brain", "lag")]

    def moments(value):
        return jnp.concatenate([a.ravel() for a in smooth(value)])

    derivative = np.asarray(jax.jit(jax.jacrev(moments))(jnp.asarray(x)))[:, lag]
    assert np.isfinite(derivative).all()

    def oracle(value):
        weights, offsets, _, _, _ = dense.arrays(value)
        ki, _, mi = dense._packed["train"]
        observations = dense.systems["train"]
        C = dense.covariance(value, "train")
        cross = (
            dense.temporal_covariance(value, query, np.full(3, -1), observations.times, mi)
            * weights[ki]
        )
        mean = cross @ jnp.linalg.solve(C, jnp.asarray(observations.values) - offsets[ki])
        variance = 1 - jnp.sum(cross * jnp.linalg.solve(C, cross.T).T, axis=1)
        return jnp.concatenate((mean, variance))

    expected = np.asarray(jax.jit(jax.jacrev(oracle))(jnp.asarray(x)))[:, lag]
    assert np.isfinite(expected).all()
    assert_allclose(moments(x), oracle(x), atol=2e-7, rtol=1e-6)
    assert_allclose(derivative, expected, atol=3e-6, rtol=1e-5)
    if not noiseless:
        step = np.eye(len(x))[lag] * 1e-5
        finite = (np.asarray(moments(x + step)) - np.asarray(moments(x - step))) / 2e-5
        assert np.isfinite(finite).all()
        assert_allclose(derivative, finite, atol=3e-6, rtol=1e-5)


def test_positive_noise_smoother_gain_preserves_small_resolved_variance():
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.state_space import _smoother_gain

    jax, jnp, _, _ = runtime()
    covariance = jnp.diag(jnp.array([1e-24, 1.0]))
    predicted = covariance + jnp.diag(jnp.array([1e-26, 0.01]))
    transition = jnp.eye(2)
    gain = jax.jit(_smoother_gain)(covariance, predicted, transition, False)
    assert_allclose(gain, np.eye(2) / 1.01, atol=0, rtol=1e-14)
    smoothed = covariance + gain @ (-0.5 * predicted) @ gain.T
    assert_allclose(
        np.diag(smoothed), np.array([1e-24, 1.0]) * (1 - 0.5 / 1.01), atol=0, rtol=1e-14
    )


def test_fixed_response_archive_from_previous_milestone_remains_loadable(tmp_path):
    from multimodalsrm.bayesian import _archive
    from multimodalsrm.bayesian.persistence import (
        _model_state,
        load_model,
    )

    model, _, _, data = fixed_fixture()
    model.fit(data)
    state = _model_state(model)
    metadata = state["fit"]["configuration"]["state_space"]
    metadata["response_support"] = "fixed_Identity_integer_Gamma_DoubleGamma"
    metadata.pop("learned_response_parameters", None)
    _archive.write(tmp_path / "legacy", state)
    restored, _ = load_model(tmp_path / "legacy")
    assert_allclose(restored.map_parameters_, model.map_parameters_, atol=0, rtol=0)
