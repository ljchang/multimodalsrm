"""Learned node clocks retain scalar/dense values and smoothing sensitivities."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import Response, TimeSeries
from multimodalsrm.bayesian._backend import runtime
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.state_space import smoother
from multimodalsrm.bayesian.state_space_grouped import event_system

from .test_bateman_scr import fixture as bateman_fixture


def fixture(lag_only=False):
    model, _, _, original = bateman_fixture()
    model.set_params(features=2)
    if lag_only:
        model.responses = dict(model.responses)
        model.responses["scr"] = Response.lag_only(
            model.responses["scr"].kernel, pooling="shared", bounds={"lag": (-0.5, 0.5)}
        )
        model.priors = replace(
            model.priors, filters={"scr": {"lag": model.priors.filters["scr"]["lag"]}}
        )
    rng = np.random.default_rng(76)
    data = {}
    for subject, runs in original.items():
        data[subject] = {}
        for run, shift in (("train", 0.0), ("second", 150.0)):
            data[subject][run] = {}
            for modality, series in runs["train"].items():
                if (subject, run, modality) == ("b", "second", "scr"):
                    continue
                times = series.times + shift
                times[2] = np.nextafter(times[1], np.inf)
                values = rng.normal(size=(len(times), 3))
                mask = rng.uniform(size=values.shape) > 0.2
                mask[0] = True
                values[~mask] = 1e20
                data[subject][run][modality] = TimeSeries(values, times, mask)
    _, state = _prepare(model, data)
    _, dense = _prepare(
        copy.copy(model).set_params(linear_algebra="dense", response_quadrature_order=96), data
    )
    scalar = copy.copy(state)
    scalar.grouped_state_space = False
    jax, _, _, _ = runtime()
    scalar._vg = jax.jit(jax.value_and_grad(scalar.objective))
    point = state.initial.copy()
    for i, name in enumerate(state.names):
        if name[0] == "loading":
            point[i] = rng.normal(scale=0.6)
        elif name[0] == "noise":
            point[i] = 0.3
    return state, scalar, dense, point


@pytest.mark.parametrize("lag_only", [False, True])
def test_learned_nodes_preserve_density_gradient_and_exact_clocks(lag_only, monkeypatch):
    from multimodalsrm.bayesian import state_space

    state, scalar, dense, point = fixture(lag_only)
    assert state.grouped_state_space and state.dynamic_state_space
    assert not state.state_space_systems  # Transitions are evaluated at current parameters.
    nodes = state.grouped_systems["train"]
    assert nodes.covariance_pairs is None
    native = nodes.times.copy()
    orderings = []
    lag = state.indices["filter", "scr", "lag"]
    for value in (0.13 - 1e-7, 0.13, 0.13 + 1e-7):
        point[lag] = value
        events, _ = event_system(state, point, "train")
        orderings.append(np.asarray(events.order))
        assert len(events.times) == len(nodes.times) < len(state.systems["train"].times)
        assert events.transition.shape[0] == len(nodes.times)
        assert_array_equal(nodes.times, native)
        actual = state.value_gradient(point)
        for reference in (scalar, dense):
            expected = reference.value_gradient(point)
            assert_allclose(actual[0], expected[0], atol=4e-6, rtol=1e-8)
            assert_allclose(actual[1], expected[1], atol=3e-5, rtol=3e-6)
    assert not np.array_equal(orderings[0], orderings[-1])
    for run, observations in state.systems.items():
        n = state.grouped_systems[run]
        assert len(n.times) == len(set(zip(state._packed[run][2], observations.times)))
        assert_array_equal(n.times[n.observation_nodes], observations.times)
    # Deficient information matrices must not require inverses or rank cutoffs.
    for loading in (0.3, 0.0):
        for i, name in enumerate(state.names):
            if name[0] == "loading":
                point[i] = loading
        expected = scalar.value_gradient(point)
        actual = state.value_gradient(point)
        assert_allclose(actual[0], expected[0], atol=2e-9, rtol=1e-10)
        assert_allclose(actual[1], expected[1], atol=2e-8, rtol=1e-8)

    def forbidden(*args, **kwargs):
        pytest.fail("grouped learned inference must not expand scalar events or dense matrices")

    monkeypatch.setattr(state_space, "operands", forbidden)
    monkeypatch.setattr(state, "covariance", forbidden)
    monkeypatch.setattr(state, "temporal_covariance", forbidden)
    # Retrace after patching, so the check exercises execution rather than a cached graph.
    jax, _, _, _ = runtime()
    assert np.isfinite(jax.jit(state.objective)(point))
    assert np.isfinite(jax.jit(smoother(state, "train", np.array([0.0, 94.0])))(point)[0]).all()


@pytest.mark.parametrize("lag_only", [False, True])
@pytest.mark.parametrize("modality", [-1, 1])
def test_joint_smoother_derivatives_across_observation_and_query_ties(lag_only, modality):
    state, scalar, dense, point = fixture(lag_only)
    jax, jnp, _, _ = runtime()
    query = np.array([119.0, 0.0, 94.0, 94.0, np.nextafter(94.0, np.inf), 98.2, -1.0])
    grouped_smooth = smoother(state, "train", query, modality)
    scalar_smooth = smoother(scalar, "train", query, modality)
    observations = dense.systems["train"]
    ki, _, mi = dense._packed["train"]
    qm = np.full(len(query), modality)

    def dense_smooth(x):
        weights, offsets, _, _, _ = dense.arrays(x)
        C = dense.covariance(x, "train")
        temporal = dense.temporal_covariance(x, query, qm, observations.times, mi)
        cross = temporal[:, None, :] * weights[ki].T[None]
        mean = cross @ jnp.linalg.solve(C, jnp.asarray(observations.values) - offsets[ki])
        solved = jnp.linalg.solve(C, cross.reshape(-1, len(ki)).T).reshape(len(ki), len(query), 2)
        prior = jnp.diag(dense.temporal_covariance(x, query, qm, query, qm))
        covariance = prior[:, None, None] * jnp.eye(2) - jnp.einsum("tkn,ntj->tkj", cross, solved)
        return mean, covariance

    def moments(fn):
        return lambda x: jnp.concatenate([a.ravel() for a in fn(x)])

    evaluate = [jax.jit(moments(fn)) for fn in (grouped_smooth, scalar_smooth, dense_smooth)]
    gradients = [jax.jit(jax.jacrev(moments(fn))) for fn in (grouped_smooth, scalar_smooth)]
    lag = state.indices["filter", "scr", "lag"]
    for value in (0.13 - 1e-7, 0.13, 0.13 + 1e-7):
        point[lag] = value
        actual = evaluate[0](point)
        assert_allclose(actual, evaluate[1](point), atol=2e-10, rtol=2e-8)
        assert_allclose(actual, evaluate[2](point), atol=3e-6, rtol=3e-6)
        assert_allclose(gradients[0](point), gradients[1](point), atol=2e-8, rtol=2e-6)
    point[lag] = 0.13
    derivative = gradients[0](point)
    for i, name in enumerate(state.names):
        if name[0] == "filter":
            dx = np.eye(len(point))[i] * 1e-5
            finite = (evaluate[2](point + dx) - evaluate[2](point - dx)) / 2e-5
            assert_allclose(derivative[:, i], finite, atol=3e-6, rtol=3e-5)


def test_scalar_learned_archive_replays_with_grouped_smoothing(tmp_path, monkeypatch):
    from multimodalsrm.bayesian import SearchConfig, state_space_grouped
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, _, _, data = bateman_fixture()
    model.set_params(search=SearchConfig(starts=1, maxiter=600, refine_maxiter=200))
    with monkeypatch.context() as patch:
        patch.setattr(state_space_grouped, "eligible", lambda problem: False)
        model.fit(data)
        assert model.map_diagnostics_["meets_gradient_tolerance"]
        expected = model.infer_latent(times=[95.0, 103.0, 110.0])["train"]
        save_model(tmp_path / "scalar", model)
    restored, _ = load_model(tmp_path / "scalar")
    assert restored.problem_.grouped_state_space and restored.problem_.dynamic_state_space
    assert restored.configuration_ == model.configuration_
    assert restored.map_diagnostics_ == model.map_diagnostics_
    actual = restored.infer_latent(times=[95.0, 103.0, 110.0])["train"]
    assert_allclose(actual.values, expected.values, atol=1e-9, rtol=1e-9)
    assert_allclose(actual.variance, expected.variance, atol=1e-9, rtol=1e-9)
    save_model(tmp_path / "resaved", restored)
