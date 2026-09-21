"""Repeated separations reuse covariance entries without binning time."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .test_bayesian_problem import api


@pytest.mark.parametrize("jitter", [0.0, 1e-7])
@pytest.mark.parametrize("identity", [False, True])
def test_pairs_match_entries_and_all_width_lag_derivatives(jitter, identity):
    api()
    import jax
    import jax.numpy as jnp

    from multimodalsrm.bayesian._backend import temporal
    from multimodalsrm.bayesian.covariance_pairs import CovariancePairs

    t = np.tile(np.arange(32) * 0.125, 2)
    t[5] += jitter
    mi = np.repeat([0, 1], 32)
    pairs = CovariancePairs.prepare(t, mi)
    assert pairs is not None
    assert len(pairs.deltas) < len(t) ** 2 / 2
    x = jnp.array([0.0 if identity else 0.3, 0.7, 0.0, 0.8])
    weights = jnp.asarray(np.random.default_rng(75).normal(size=(len(t), len(t))))

    def direct(x):
        return temporal(t, t, x[:2][mi], x[:2][mi], x[2:][mi], x[2:][mi], 3.0)

    def compact(x):
        return pairs.evaluate(x[:2], x[2:], 3.0)

    assert_allclose(compact(x), direct(x), atol=2e-15, rtol=2e-15)
    expected = jax.jit(jax.value_and_grad(lambda x: (direct(x) * weights).sum()))(x)
    actual = jax.jit(jax.value_and_grad(lambda x: (compact(x) * weights).sum()))(x)
    assert_allclose(actual[0], expected[0], atol=1e-11, rtol=1e-11)
    assert_allclose(actual[1], expected[1], atol=1e-10, rtol=1e-10)


def test_irregular_clocks_keep_direct_evaluation():
    api()
    from multimodalsrm.bayesian.covariance_pairs import CovariancePairs

    times = np.sort(np.random.default_rng(6).uniform(0, 50, 32))
    assert CovariancePairs.prepare(times, np.zeros(len(times), dtype=int)) is None


def test_grouped_backend_uses_pairs_for_repeated_native_separations():
    api()
    from multimodalsrm.bayesian.grouped import GroupedSystem

    times = np.tile(np.arange(40) * 0.125, 4)
    modalities = np.tile(np.repeat([0, 1], 40), 2)
    nodes = GroupedSystem.prepare(times, modalities)
    assert len(nodes.times) == 80
    assert nodes.covariance_pairs is not None
    assert len(nodes.covariance_pairs.deltas) < len(nodes.times) ** 2 / 2


@pytest.mark.parametrize("zero_loading", [False, True])
def test_compact_grouped_high_snr_zero_loadings_and_projection(zero_loading):
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian.prediction import project

    from .test_bayesian_problem import independent_parameters, problem_fixture

    b = api()
    old, adapter, data = problem_fixture(gaussian=True)
    times = np.arange(0.0, 20.0, 0.25)
    for subject, runs in data.items():
        for modalities in runs.values():
            for modality in modalities:
                y = np.sin(times / 3) if subject == "a" else np.cos(times / 3)
                modalities[modality] = TimeSeries(y[:, None], times)
    adapter._prepare(data)
    dense = b.BayesianProblem(adapter, old.priors, anchor=old.anchor)
    grouped = b.BayesianProblem(adapter, old.priors, anchor=old.anchor, linear_algebra="grouped")
    assert all(n.covariance_pairs is not None for n in grouped.grouped_systems.values())
    x = independent_parameters(dense)
    for i, name in enumerate(dense.names):
        if name[0] == "loading":
            x[i] = 0.0 if zero_loading and name[2] == "signal" else x[i] * 10
        elif name[0] == "noise":
            x[i] *= 0.001
    actual, expected = grouped.value_gradient(x), dense.value_gradient(x)
    assert_allclose(actual[0], expected[0], rtol=1e-8, atol=1e-7)
    assert_allclose(actual[1], expected[1], rtol=2e-6, atol=1e-5)
    for key in (None, ("b", "signal", 0)):
        kw = dict(key=key, include_noise=key is not None)
        actual = project(grouped, x[None], "train", np.array([6.0, 9.0, 13.0]), **kw)
        expected = project(dense, x[None], "train", np.array([6.0, 9.0, 13.0]), **kw)
        assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
