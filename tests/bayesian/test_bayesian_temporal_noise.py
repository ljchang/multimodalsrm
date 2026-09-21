"""Independent dense Gaussian checks for fixed OU observation noise."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.problem import BayesianProblem

from .test_bayesian_multifactor import fixture


def problems(features=2):
    original, x, _ = fixture(features=features)
    kwargs = dict(anchor=original.anchor, noise_timescales={"brain": 3.0, "aux": 1.2})
    dense = BayesianProblem(original.adapter, original.priors, **kwargs)
    grouped = BayesianProblem(original.adapter, original.priors, linear_algebra="grouped", **kwargs)
    return original, dense, grouped, x


def independent_noise(problem, x, run):
    params = dict(zip(problem.names, x))
    system = problem.systems[run]
    expected = np.zeros((len(system.times), len(system.times)))
    for i, (t, key) in enumerate(zip(system.times, system.keys)):
        for j, (u, other) in enumerate(zip(system.times, system.keys)):
            if key == other:
                tau = problem.noise_timescales.get(key[1])
                expected[i, j] = params["noise", *key[:2]] * (
                    np.exp(-abs(t - u) / tau) if tau else float(i == j)
                )
    return expected


@pytest.mark.parametrize("features", [1, 2])
@pytest.mark.parametrize("zero", [False, True])
def test_ou_covariance_density_gradients_and_precision(features, zero):
    original, dense, grouped, x = problems(features)
    if zero:
        for i, name in enumerate(dense.names):
            if name[0] == "loading" and name[1] == "b":
                x[i] = 0
    for run, system in dense.systems.items():
        noise = independent_noise(dense, x, run)
        assert_allclose(
            dense.covariance(x, run),
            original.covariance(x, run, include_noise=False) + noise,
            atol=1e-12,
        )
        variance = np.asarray(dense.arrays(x)[2])[dense._packed[run][1]]
        rhs = np.random.default_rng(71).normal(size=(len(system.times), 3))
        assert_allclose(
            grouped.noise_systems[run].precision(rhs, variance),
            np.linalg.solve(noise, rhs),
            atol=1e-11,
        )
    a, ga = dense.value_gradient(x)
    b, gb = grouped.value_gradient(x)
    assert_allclose(a, b, atol=1e-8, rtol=1e-10)
    assert_allclose(ga, gb, atol=1e-7, rtol=1e-8)
    # Physical-coordinate finite differences independently check loading,
    # offset, variance and free response derivatives.
    for family in ("loading", "offset", "noise", "filter"):
        i = next(i for i, n in enumerate(dense.names) if n[0] == family)
        step = np.zeros(len(x))
        step[i] = 1e-5
        expected = (float(dense.objective(x + step)) - float(dense.objective(x - step))) / 2e-5
        assert_allclose(gb[i], expected, atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("features", [1, 2])
@pytest.mark.parametrize(
    "key,noisy", [(None, False), (("b", "brain", 1), False), (("b", "brain", 1), True)]
)
def test_ou_conditional_moments_match_dense(features, key, noisy):
    _, dense, grouped, x = problems(features)
    query = np.array([2.5, 4.0, 7.1])
    a = project(dense, x[None], "train", query, key=key, include_noise=noisy)
    b = project(grouped, x[None], "train", query, key=key, include_noise=noisy)
    assert_allclose(a, b, rtol=1e-8, atol=1e-9)
    if noisy:
        system = dense.systems["train"]
        indices = [i for i, k in enumerate(system.keys) if k == key]
        mean, variance = project(
            grouped,
            x[None],
            "train",
            system.times[indices],
            key=key,
            include_noise=True,
        )
        assert_allclose(mean[0], system.values[indices], atol=1e-8)
        assert_allclose(variance, 0, atol=1e-8)


def test_omitted_modalities_stay_white_and_variance_profile_fails_closed():
    from multimodalsrm.bayesian.noise_profile import variance_profile

    original, x, _ = fixture()
    p = BayesianProblem(
        original.adapter,
        original.priors,
        anchor=original.anchor,
        linear_algebra="grouped",
        noise_timescales={"brain": 3},
    )
    noise = independent_noise(p, x, "train")
    signal = original.covariance(x, "train", include_noise=False)
    assert_allclose(p.covariance(x, "train"), signal + noise, atol=1e-12)
    i = next(i for i, n in enumerate(p.names) if n[0] == "noise")
    result = variance_profile(p, x, i)
    assert not result.available and "correlated" in result.reason


@pytest.mark.parametrize(
    "setting",
    [
        {"brain": 0},
        {"brain": -1},
        {"brain": True},
        {"brain": np.inf},
        {"brain": np.nan},
        {"unknown": 2},
        [2],
    ],
)
def test_invalid_fixed_noise_config_rejected(setting):
    original, _, _ = fixture()
    with pytest.raises(ValueError, match="noise_timescales"):
        BayesianProblem(
            original.adapter,
            original.priors,
            anchor=original.anchor,
            noise_timescales=setting,
        )


def test_scalar_multirun_noise_keeps_run_boundaries_and_native_masks():
    from .test_bayesian_grouped import grouped_fixture

    old, _, x = grouped_fixture()
    kwargs = dict(anchor=old.anchor, noise_timescales={"ref": 2.5, "signal": 1.1})
    dense = BayesianProblem(old.adapter, old.priors, **kwargs)
    grouped = BayesianProblem(old.adapter, old.priors, linear_algebra="grouped", **kwargs)
    assert len(grouped.noise_systems) == 2
    for run in dense.systems:
        assert_allclose(
            dense.covariance(x, run),
            old.covariance(x, run, include_noise=False) + independent_noise(dense, x, run),
            atol=1e-12,
        )
    assert_allclose(grouped.value_gradient(x)[0], dense.value_gradient(x)[0], atol=1e-8)
    assert_allclose(grouped.value_gradient(x)[1], dense.value_gradient(x)[1], atol=1e-7)


def test_noise_config_clone_validation_and_unchanged_default():
    from sklearn.base import clone

    from .test_bayesian_model import make_model

    model, data = make_model()
    model.set_params(noise_timescales={"ref": 2.5})
    assert clone(model).noise_timescales == {"ref": 2.5}
    with pytest.raises(ValueError, match="noise_timescales"):
        model.set_params(inference="posterior").fit(data)
    with pytest.raises(ValueError, match="noise_timescales"):
        model.set_params(inference="map", run_baseline_sd={"ref": 0.1}).fit(data)
    original, x, _ = fixture()
    empty = BayesianProblem(
        original.adapter, original.priors, anchor=original.anchor, noise_timescales={}
    )
    assert_allclose(empty.value_gradient(x)[0], original.value_gradient(x)[0], atol=0, rtol=0)
    assert_allclose(empty.value_gradient(x)[1], original.value_gradient(x)[1], atol=0, rtol=0)


def test_precision_rejects_duplicate_times_without_jitter():
    from multimodalsrm.bayesian.temporal_noise import NoiseSystem

    keys = [("s", "m", 0)]
    with pytest.raises(ValueError, match="distinct"):
        NoiseSystem.prepare([0, 0], keys * 2, keys, {"m": 1})
    system = NoiseSystem.prepare([0, 1e-10], keys * 2, keys, {"m": 1})
    expected = np.log(-np.expm1(-2e-10))
    assert_allclose(system.logdet, expected, atol=1e-14)


def test_noisy_ou_interpolates_with_zero_loadings_while_white_is_a_replica():
    original, x, _ = fixture(features=1)
    p = BayesianProblem(
        original.adapter,
        original.priors,
        anchor=original.anchor,
        linear_algebra="grouped",
        noise_timescales={"brain": 3},
    )
    for i, name in enumerate(p.names):
        if name[0] == "loading":
            x[i] = 0
    params = dict(zip(p.names, x))
    for key, colored in [(("b", "brain", 1), True), (("b", "aux", 1), False)]:
        system = p.systems["train"]
        indices = [i for i, k in enumerate(system.keys) if k == key]
        query = system.times[indices]
        mean, variance = project(p, x[None], "train", query, key=key, include_noise=True)
        assert_allclose(
            mean[0],
            system.values[indices] if colored else params["offset", *key],
            atol=1e-9,
        )
        assert_allclose(variance, 0 if colored else params["noise", *key[:2]], atol=1e-9)
        mean, variance = project(p, x[None], "train", query, key=key, include_noise=False)
        assert_allclose(mean, params["offset", *key], atol=1e-10)
        assert_allclose(variance, 0, atol=1e-10)
