"""Independent Gaussian checks for marginalized participant/run baselines."""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from multimodalsrm import Identity
from multimodalsrm.bayesian import BayesianProblem
from multimodalsrm.bayesian.prediction import project

from ..reference.continuous_covariance import response_covariance
from .test_bayesian_problem import independent_parameters, problem_fixture


def baseline_problem(mode="dense", config=None, gaussian=True, two_runs=True, systems=None):
    assert "run_baseline_sd" in inspect.signature(BayesianProblem).parameters, (
        "run baseline model is missing"
    )
    base, adapter, _ = problem_fixture(gaussian, two_runs)
    return BayesianProblem(
        adapter,
        base.priors,
        anchor=base.anchor,
        systems=systems,
        linear_algebra=mode,
        run_baseline_sd={"signal": 2.0} if config is None else config,
    )


@pytest.mark.parametrize("mode", ["dense", "grouped"])
def test_density_and_gradient_include_independent_run_baselines(mode):
    p = baseline_problem(mode)
    x = independent_parameters(p)
    base, _, _ = problem_fixture(True, True)
    expected = 0.0
    for run, system in p.systems.items():
        U = np.array([2.0 if k[1] == "signal" else 0.0 for k in system.keys])
        C = np.asarray(base.covariance(x, run)) + np.outer(U, U)
        offsets = [0.12 if k[0] == "a" else -0.2 for k in system.keys]
        assert_allclose(p.covariance(x, run), C, atol=1e-12)
        expected -= multivariate_normal.logpdf(system.values, mean=offsets, cov=C)
    assert_allclose(p.nll(x), expected, atol=1e-9)
    # Check the actual marginal density derivative, including Gaussian lag/width.
    _, gradient = p.value_gradient(x)
    h = 1e-5
    finite = [
        (float(p.objective(x + h * e)) - float(p.objective(x - h * e))) / (2 * h)
        for e in np.eye(len(x))
    ]
    assert_allclose(gradient, finite, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["dense", "grouped"])
@pytest.mark.parametrize("key", [("b", "signal", 0), None])
def test_prediction_matches_joint_gaussian_baseline_cross_covariance(mode, key):
    p = baseline_problem(mode, gaussian=False, two_runs=False)
    x = independent_parameters(p)
    system = p.systems["train"]
    t = np.array([5.0, 8.0])
    weights = np.array([1.2 if k[0] == "a" else -0.7 for k in system.keys])
    offsets = np.array([0.12 if k[0] == "a" else -0.2 for k in system.keys])
    measurement = np.array([0.15 if k[0] == "a" else 0.3 for k in system.keys])
    T = response_covariance(system.times, system.times, Identity(), Identity(), 3.0)
    u = np.array([2.0 if k[1] == "signal" else 0.0 for k in system.keys])
    C = T * np.outer(weights, weights) + np.diag(measurement) + np.outer(u, u)
    loading, offset, extra = (-0.7, -0.2, 4.0) if key else (1.0, 0.0, 0.0)
    cross = response_covariance(t, system.times, Identity(), Identity(), 3.0) * loading * weights
    if key:
        cross += 2 * u[None, :]
    expect_mean = cross @ np.linalg.solve(C, system.values - offsets) + offset
    expect_var = loading**2 + extra - np.sum(cross * np.linalg.solve(C, cross.T).T, axis=1)
    mu, var = project(p, x[None], "train", t, key=key, include_noise=False)
    assert_allclose(mu[0], expect_mean, atol=1e-10)
    assert_allclose(var[0], expect_var, atol=1e-10)


def test_unobserved_target_baseline_keeps_prior_uncertainty():
    from multimodalsrm._observation_preparation import ObservationSystem

    system = ObservationSystem(np.array([1.0, 4.0]), [("a", "ref", 0)] * 2, np.array([2.0, -1.0]))
    p = baseline_problem("grouped", gaussian=False, two_runs=False, systems={"test": system})
    base = baseline_problem(
        "grouped", config={}, gaussian=False, two_runs=False, systems={"test": system}
    )
    x = independent_parameters(p)
    t = np.array([2.0, 3.0])
    mu, v = project(p, x[None], "test", t, key=("b", "signal", 0), include_noise=True)
    m0, v0 = project(base, x[None], "test", t, key=("b", "signal", 0), include_noise=True)
    assert_allclose(mu, m0, atol=1e-12)
    assert_allclose(v, v0 + 4.0, atol=1e-12)


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": 1.0},
        {"signal": -1.0},
        {"signal": True},
        {"signal": np.inf},
        {"signal": 1e200},
    ],
)
def test_invalid_baseline_specification_is_rejected(config):
    with pytest.raises(ValueError, match="baseline"):
        baseline_problem(config=config)


def test_zero_baseline_preserves_existing_model_exactly():
    base, _, _ = problem_fixture(True, True)
    p = baseline_problem(config={"signal": 0.0})
    x = independent_parameters(p)
    assert p.names == base.names
    assert float(p.objective(x)) == float(base.objective(x))
    assert_allclose(p.value_gradient(x)[1], base.value_gradient(x)[1], atol=0, rtol=0)


def test_estimator_preserves_baseline_specification_and_target_exclusion():
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian import BayesianMultimodalSRM, SearchConfig

    assert "run_baseline_sd" in inspect.signature(BayesianMultimodalSRM).parameters, (
        "estimator baseline option is missing"
    )
    base, adapter, data = problem_fixture(False, False)
    model = BayesianMultimodalSRM(
        priors=base.priors,
        anchor=base.anchor,
        responses=adapter.responses_,
        inference="map",
        search=SearchConfig(starts=1, maxiter=100, refine_maxiter=100),
        linear_algebra="grouped",
        run_baseline_sd={"signal": 2.0},
    )
    model.fit(data)
    from multimodalsrm.bayesian.prediction import result as series_result

    ref = series_result(
        model,
        "train",
        np.array([8.0]),
        model.map_parameters_[None],
        [("a", "ref", 0)],
        include_noise=False,
    )
    assert ref.metadata["quantity"] == "filtered_signal"
    t = np.arange(0.0, 22.0, 2.0)
    donors = {
        "a": {"test": {"ref": TimeSeries(np.sin(t / 3)[:, None], t + 100)}},
        "b": {"test": {"signal": object()}},
    }
    fitted = model.condition(donors, targets={"b": ["signal"]})
    result = fitted.predict(times=t + 100, include_noise=True)["b"]["test"]["signal"]
    assert fitted.problem_.run_baseline_sd == {"signal": 2.0}
    assert fitted.configuration_["run_baseline"]["marginalization"] == "analytic"
    assert model.problem_.systems.keys() == {"train"}
    assert result.metadata["run_baseline"]["sd"] == {"signal": 2.0}
    assert not any(k[:2] == ("b", "signal") for k in fitted.problem_.systems["test"].keys)
    assert np.isfinite(result.values).all()
    assert np.min(result.component_variances) >= 4.0


def test_baselines_are_independent_between_participants_and_features():
    from multimodalsrm._observation_preparation import ObservationSystem

    # Design independence is a key contract, independent of fitted feature maps.
    from multimodalsrm.bayesian.baselines import design

    keys = [("a", "eda", 0), ("a", "eda", 0), ("a", "eda", 1), ("b", "eda", 0)]
    U, _ = design(ObservationSystem(np.arange(4.0), keys, np.zeros(4)), {"eda": 2.0})
    assert_allclose(U @ U.T, [[4, 4, 0, 0], [4, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])


def test_donor_deletion_preserves_remaining_baselines_and_empty_prior():
    from multimodalsrm._observation_preparation import ObservationSystem
    from multimodalsrm.bayesian.diagnostics import fixed_prediction

    system = ObservationSystem(np.array([1.0, 4.0]), [("a", "ref", 0)] * 2, np.array([2.0, -1.0]))
    p = baseline_problem(
        "grouped",
        config={"signal": 2.0, "ref": 1.0},
        gaussian=False,
        two_runs=False,
        systems={"test": system},
    )
    x = independent_parameters(p)
    got = fixed_prediction(
        p,
        x,
        "test",
        np.array([2.0, 3.0]),
        target=("b", "signal"),
        drop_modalities=["ref"],
    )
    assert_allclose(got["mean"], -0.2)
    assert_allclose(got["variance"], 4.79)


def test_spectral_baselines_are_explicitly_rejected():
    from multimodalsrm.bayesian import SpectralConfig

    base, adapter, _ = problem_fixture(False, False)
    with pytest.raises(ValueError, match="baseline"):
        BayesianProblem(
            adapter,
            base.priors,
            anchor=base.anchor,
            linear_algebra="spectral",
            spectral=SpectralConfig(rank=16, padding=20.0),
            run_baseline_sd={"signal": 2.0},
        )


@pytest.mark.parametrize("sd,noise", [(3.0, 1e-6), (1000.0, 0.3)])
def test_baseline_only_posterior_keeps_small_positive_variance(sd, noise):
    p = baseline_problem("grouped", config={"signal": sd}, gaussian=False, two_runs=False)
    x = independent_parameters(p)
    x[p.indices["loading", "b", "signal", 0]] = 0.0
    x[p.indices["noise", "b", "signal"]] = noise
    _, variance = project(
        p,
        x[None],
        "train",
        np.array([5.0, 8.0]),
        key=("b", "signal", 0),
        include_noise=False,
    )
    expected = 1 / (1 / sd**2 + 9 / noise)
    assert_allclose(variance, expected, rtol=1e-7, atol=1e-12)


def test_grouped_high_signal_to_noise_baselines_match_dense():
    from .test_bayesian_grouped import grouped_fixture

    base, _, x = grouped_fixture()
    problems = [
        BayesianProblem(
            base.adapter,
            base.priors,
            anchor=base.anchor,
            linear_algebra=mode,
            run_baseline_sd={"ref": 3.0, "signal": 3.0},
        )
        for mode in ("dense", "grouped")
    ]
    for i, name in enumerate(base.names):
        if name[0] == "loading":
            x[i] *= 10
        elif name[0] == "noise":
            x[i] *= 0.001
    predictions = [
        project(
            p,
            x[None],
            "train",
            np.array([2.7, 7.1, 15.9]),
            key=("b", "signal", 2),
            include_noise=False,
        )
        for p in problems
    ]
    assert_allclose(predictions[1], predictions[0], rtol=1e-7, atol=2e-9)
    value, gradient = problems[1].value_gradient(x)
    expected, expected_gradient = problems[0].value_gradient(x)
    assert_allclose(value, expected, rtol=1e-8, atol=1e-7)
    assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=1e-5)
