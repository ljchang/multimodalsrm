"""Grouped updates preserve Gaussian densities, gradients and joint factor moments."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior
from multimodalsrm.bayesian._backend import runtime
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.state_space import StateSpaceSystem, smoother


def prepared(features=2, noise=0.2):
    times = np.array([0.0, 1.0, np.nextafter(1.0, 2.0), 2.7, 5.0, 1000.0, 1001.0])
    rng = np.random.default_rng(310)
    data = {}
    for subject in ("a", "b"):
        data[subject] = {}
        for run, shift in (("first", 0.0), ("second", 100.0)):
            data[subject][run] = {}
            for modality in ("ref", "aux"):
                if (subject, run, modality) == ("b", "second", "aux"):
                    continue
                shifted = times + shift
                shifted[2] = np.nextafter(shifted[1], np.inf)
                t = shifted if modality == "ref" else shifted[::2]
                values = rng.normal(size=(len(t), 3))
                mask = rng.uniform(size=values.shape) > 0.2
                mask[0] = True
                values[~mask] = 1e20
                data[subject][run][modality] = TimeSeries(values, t, mask)
    model = BayesianMultimodalSRM(
        features=features,
        priors=BayesianPriors(noise=Prior.lognormal(-1.0, 0.8)),
        anchor=("a", "ref", 0),
        responses={
            m: Response(Identity(), estimate=False, pooling="shared") for m in ("ref", "aux")
        },
        inference="map",
        linear_algebra="state_space",
    )
    _, state = _prepare(model, data)
    _, dense = _prepare(copy.copy(model).set_params(linear_algebra="dense"), data)
    x = state.initial.copy()
    for index, name in enumerate(state.names):
        if name[0] == "loading":
            x[index] = rng.normal(scale=0.8)
            if features == 1 and name[1:] == state.anchor:
                x[index] = abs(x[index])
        elif name[0] == "offset":
            x[index] = rng.normal(scale=0.3)
        elif name[0] == "noise":
            x[index] = noise
    return state, dense, x


def scalar_reference(state):
    """Keep the pre-grouping recurrence as an independent computational oracle."""
    jax, _, _, _ = runtime()
    scalar = copy.copy(state)
    scalar.grouped_state_space = False
    scalar.state_space_systems = {
        run: StateSpaceSystem.prepare(
            system.times - state.response_state_space.lags[state._packed[run][2]],
            state.response_state_space,
            state.features,
        )
        for run, system in state.systems.items()
    }
    scalar._vg = jax.jit(jax.value_and_grad(scalar.objective))
    return scalar


@pytest.mark.parametrize("features", [1, 2, 3])
@pytest.mark.parametrize("rank", ["full", "one", "zero"])
def test_masked_multirun_density_and_gradient_match_dense_and_scalar(features, rank):
    state, dense, x = prepared(features)
    if rank != "full":
        for index, name in enumerate(state.names):
            if name[0] == "loading":
                x[index] = 0.3 if rank == "one" else 0.0
        if features == 1 and rank == "zero":
            x[state.indices["loading", *state.anchor]] = 0.3
    expected, gradient = dense.value_gradient(x)
    for problem in (state, scalar_reference(state)):
        actual, actual_gradient = problem.value_gradient(x)
        assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
        assert_allclose(actual_gradient, gradient, atol=2e-8, rtol=2e-9)


def test_low_noise_score_avoids_subtraction_of_large_quadratics():
    state, _, x = prepared(noise=1e-7)
    scalar = scalar_reference(state)
    # Consistent high-SNR observations make subtractive sufficient-statistic
    # quadratics especially fragile. All masks and near-coincident clocks stay.
    for run, system in state.systems.items():
        ki, _, _ = state._packed[run]
        weights, offsets, _, _, _ = map(np.asarray, state.arrays(x))
        z = np.column_stack((np.sin(system.times / 4), np.cos(system.times / 5)))
        system.values[:] = np.sum(weights[ki] * z, axis=1) + offsets[ki]
    expected, gradient = scalar.value_gradient(x)
    actual, actual_gradient = state.value_gradient(x)
    assert_allclose(actual, expected, atol=1e-7, rtol=1e-9)
    assert_allclose(actual_gradient, gradient, atol=0.02, rtol=2e-7)


def test_exact_grouping_reduces_transition_storage_without_merging_clocks():
    state, _, _ = prepared()
    assert state.grouped_state_space
    for run, system in state.systems.items():
        nodes = state.grouped_systems[run]
        events = state.state_space_systems[run]
        expected = {(key[1], float(t)) for key, t in zip(system.keys, system.times)}
        assert len(nodes.times) == len(expected) < len(system.times)
        assert events.transition.shape[0] == len(expected)
        assert events.process_covariance.shape == events.transition.shape
        assert nodes.covariance_pairs is None
        assert_array_equal(nodes.times[nodes.observation_nodes], system.times)
    nodes = state.grouped_systems["first"]
    assert 1.0 in nodes.times and np.nextafter(1.0, 2.0) in nodes.times
    assert len(np.flatnonzero(nodes.times == 0.0)) == 2  # Different modalities.


def test_smoother_joint_factor_covariance_and_derivatives_match_dense(monkeypatch):
    from multimodalsrm.bayesian import state_space

    state, dense, x = prepared()
    query = np.array([1002.0, 1.0, 0.0, 1.0, np.nextafter(1.0, 2.0), 4.0, -2.0])
    system = dense.systems["first"]
    ki, _, _ = dense._packed["first"]

    def reference(point):
        weights, offsets, _, _, _ = map(np.asarray, dense.arrays(point))
        dt = np.sqrt(3) * np.abs(query[:, None] - system.times) / dense.length_scale
        cross = ((1 + dt) * np.exp(-dt))[:, None, :] * weights[ki].T[None]
        C = np.asarray(dense.covariance(point, "first"))
        mean = cross @ np.linalg.solve(C, system.values - offsets[ki])
        solved = np.linalg.solve(C, cross.reshape(-1, len(ki)).T).reshape(len(ki), len(query), 2)
        covariance = np.eye(2) - np.einsum("tkn,ntj->tkj", cross, solved)
        return mean, covariance

    def forbidden(*args, **kwargs):
        pytest.fail("grouped inference must not expand scalar events or use dense covariance")

    monkeypatch.setattr(state_space, "operands", forbidden)
    monkeypatch.setattr(state, "covariance", forbidden)
    monkeypatch.setattr(state, "temporal_covariance", forbidden)
    jax, jnp, _, _ = runtime()
    smooth = jax.jit(smoother(state, "first", query))
    actual = smooth(x)
    expected = reference(x)
    for a, b in zip(actual, expected):
        assert_allclose(a, b, atol=2e-10, rtol=2e-9)
    assert np.max(np.abs(np.asarray(actual[1])[:, 0, 1])) > 1e-4
    assert np.linalg.eigvalsh(actual[1]).min() > 0
    gradient = np.asarray(jax.jit(jax.grad(lambda p: sum(jnp.sum(v) for v in smooth(p))))(x))
    for index in (0, state.indices["offset", "a", "ref", 0], state.indices["noise", "a", "ref"]):
        delta = np.eye(len(x))[index] * 1e-5
        finite = (
            sum(np.sum(v) for v in reference(x + delta))
            - sum(np.sum(v) for v in reference(x - delta))
        ) / 2e-5
        assert_allclose(gradient[index], finite, atol=2e-7, rtol=2e-6)


def test_learned_responses_and_zero_noise_support_keep_scalar_path():
    from .test_bayesian_problem import problem_fixture
    from .test_bayesian_state_space import state_problem

    dense, _, _ = problem_fixture(gaussian=True)
    assert not state_problem(dense).grouped_state_space
    dense, _, _ = problem_fixture()
    dense.priors = BayesianPriors(noise=Prior.uniform(0.0, 1.0))
    assert not state_problem(dense).grouped_state_space
