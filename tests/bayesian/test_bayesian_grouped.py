"""An unreduced covariance independently checks exact functional grouping."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import TimeSeries

from .test_bayesian_problem import api, problem_fixture


def grouped_fixture():
    b = api()
    p, adapter, data = problem_fixture(gaussian=True, two_runs=True)
    rng = np.random.default_rng(471)
    for subject in data:
        for run in data[subject]:
            for modality, ts in list(data[subject][run].items()):
                values = rng.normal(size=(len(ts.times), 3))
                mask = np.ones_like(values, dtype=bool)
                mask[2, 1] = False
                data[subject][run][modality] = TimeSeries(values, ts.times, mask)
    # Shared modality across participants; one distinct timestamp must stay distinct.
    for run, modalities in data["a"].items():
        ts = modalities["ref"]
        times = ts.times.copy()
        times[1] += 1e-7
        data["b"][run]["ref"] = TimeSeries(ts.values * 0.6, times, ts.mask)
    adapter._prepare(data)
    dense = b.BayesianProblem(adapter, p.priors, anchor=p.anchor)
    grouped = b.BayesianProblem(adapter, p.priors, anchor=p.anchor, linear_algebra="grouped")
    x = dense.initial.copy()
    for i, name in enumerate(dense.names):
        if name[0] == "loading":
            x[i] = 0.9 if name == ("loading", *p.anchor) else rng.normal()
        elif name[0] == "offset":
            x[i] = rng.normal(scale=0.3)
        elif name[0] == "noise":
            x[i] = 0.1 + rng.uniform()
    return dense, grouped, x


@pytest.mark.parametrize("zero", [False, True])
def test_grouped_density_and_all_physical_gradients_match_dense(zero):
    dense, grouped, x = grouped_fixture()
    if zero:
        # A whole modality can have zero loadings without breaking derivatives.
        for i, name in enumerate(dense.names):
            if name[0] == "loading" and name[2] == "signal":
                x[i] = 0.0
    assert_allclose(grouped.nll(x), dense.nll(x), rtol=1e-11, atol=1e-10)
    a, ga = grouped.value_gradient(x)
    b, gb = dense.value_gradient(x)
    assert_allclose(a, b, rtol=1e-11, atol=1e-10)
    assert_allclose(ga, gb, rtol=1e-8, atol=1e-8)
    for run, system in dense.systems.items():
        nodes = grouped.grouped_systems[run]
        assert len(nodes.times) < len(system.times) / 2
        expected = set(zip(dense._packed[run][2], system.times))
        assert set(zip(nodes.modalities, nodes.times)) == expected


@pytest.mark.parametrize(
    "key,include_noise",
    [(None, False), (("b", "signal", 1), False), (("b", "signal", 1), True)],
)
def test_grouped_conditional_mean_and_variance_match_dense(key, include_noise):
    from multimodalsrm.bayesian.prediction import project

    dense, grouped, x = grouped_fixture()
    draws = np.stack([x, x.copy()])
    for i, name in enumerate(dense.names):
        if name[0] == "loading" and name[2] == "signal":
            draws[1, i] = 0.0
    query = np.array([2.7, 7.1, 15.9])
    expected = project(dense, draws, "train", query, key=key, include_noise=include_noise)
    actual = project(grouped, draws, "train", query, key=key, include_noise=include_noise)
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_estimator_preserves_explicit_algebra_through_clone_and_condition():
    from sklearn.base import clone

    from .test_bayesian_model import make_model

    b = api()
    model, data = make_model(linear_algebra="grouped")
    model.search = b.SearchConfig(starts=1, maxiter=30)
    assert clone(model).linear_algebra == "grouped"
    model.fit(data)
    donors = {
        s: {"test": {m: TimeSeries(ts.values, ts.times) for m, ts in runs["train"].items()}}
        for s, runs in data.items()
    }
    conditioned = model.condition(donors, targets={"b": ["signal"]})
    assert conditioned.problem_.linear_algebra == "grouped"
    assert conditioned.configuration_["linear_algebra"] == "grouped"
    assert all(
        v["functionals"] <= v["observations"]
        for v in conditioned.configuration_["linear_system_sizes"].values()
    )
    with pytest.raises(ValueError, match="linear_algebra"):
        model.set_params(linear_algebra="invalid").fit(data)


def test_grouped_high_signal_to_noise_preserves_density_and_gradients():
    dense, grouped, x = grouped_fixture()
    for i, name in enumerate(dense.names):
        if name[0] == "loading":
            x[i] *= 10
        elif name[0] == "noise":
            x[i] *= 0.001
    value, gradient = grouped.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert_allclose(value, expected, rtol=1e-8, atol=1e-7)
    assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=1e-5)


def test_grouped_handles_singular_latent_covariance_across_identity_modalities():
    from .test_bayesian_problem import independent_parameters

    dense, adapter, _ = problem_fixture(gaussian=False, two_runs=True)
    grouped = api().BayesianProblem(
        adapter, dense.priors, anchor=dense.anchor, linear_algebra="grouped"
    )
    x = independent_parameters(dense)
    # Both modalities use Identity at the same times, so K contains duplicate
    # rows. Observation noise still makes the original covariance invertible.
    assert_allclose(grouped.value_gradient(x)[0], dense.value_gradient(x)[0], atol=1e-10)
    assert_allclose(grouped.value_gradient(x)[1], dense.value_gradient(x)[1], atol=1e-9)
