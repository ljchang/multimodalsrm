"""Joint latent paths checked against independent observation-space conditioning."""

import copy
from functools import lru_cache
from math import exp, pi, sqrt

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import erf

from .test_bayesian_posterior_updates import prepared


def trajectory_model(features=3, algebra="grouped"):
    """Physical parameter examples, deliberately not a convergence claim."""
    from multimodalsrm.bayesian.posterior_coordinates import metadata

    model = prepared(features, algebra)
    p = model.problem_
    first = model.map_parameters_.copy()
    second = first.copy()
    second[p.indices[("noise", "a", "brain")]] *= 2
    second[p.indices[("offset", "a", "brain", 0)]] += 0.8
    second[p.indices[("filter", "aux", "width")]] = 0.7
    rotation, _ = np.linalg.qr(np.random.default_rng(185).normal(size=(features, features)))
    rotated = second.copy()
    nw = len(p.keys) * features
    rotated[:nw] = (second[:nw].reshape(-1, features) @ rotation).ravel()
    model.parameter_draws_ = np.array([[first, second], [rotated, first]])
    model.configuration_.update(
        uncertainty="parameter_posterior_mixture",
        reference_convention=p.reference_convention,
        factor_orientation=metadata(model._factor_anchor_keys_),
    )
    model.sampling_diagnostics_ = {"passes": False}
    return model


@lru_cache(maxsize=8192)
def integrated_temporal(distance, width_a, width_b, length_scale):
    """Quadrature for the fitted analytic target, independent of its formula.

    Production uses full Gaussian responses with finite-L2 normalization,
    within its declared finite-support error bound. Integrating finite support
    instead would test a slightly different target. The Gaussian difference
    variable is normal; truncating it at 12 SD omits <4e-33 probability mass.
    """
    rate = sqrt(3) / length_scale
    sigma = np.hypot(width_a, width_b)
    if sigma == 0:
        return (1 + rate * distance) * exp(-rate * distance)
    mass = np.prod([sqrt(2) * pi**0.25 * sqrt(w) / sqrt(erf(6)) for w in (width_a, width_b) if w])

    def integrand(z):
        d = rate * abs(distance - sigma * z)
        return (1 + d) * exp(-d - z * z / 2) / sqrt(2 * pi)

    points = [-6, -3, 0, 3, 6]
    if -12 < distance / sigma < 12:
        points.append(distance / sigma)
    value, error = quad(integrand, -12, 12, points=sorted(set(points)), epsabs=2e-13, epsrel=2e-13)
    assert mass * error < 1e-11
    return mass * value


def reference_joint(model, x, query, run="train"):
    """Numerical Gaussian integrals and a direct observation covariance solve."""
    p = model.problem_
    pars = dict(zip(p.names, x, strict=True))
    W = np.array([[pars[("loading", *key, f)] for f in range(p.features)] for key in p.keys])
    offsets = np.array([pars[("offset", *key)] for key in p.keys])
    parameters = {}
    for modality, response in p.responses.items():
        kernel = response.initial_kernel()
        parameters[modality] = tuple(
            pars.get(("filter", modality, name), getattr(kernel, name, 0.0))
            for name in ("width", "lag")
        )
    parameters[None] = (0.0, 0.0)

    def temporal(a, ma, b, mb):
        wa, la = parameters[ma]
        wb, lb = parameters[mb]
        return integrated_temporal(
            float(abs(a - b - la + lb)), float(wa), float(wb), p.length_scale
        )

    system = p.systems[run]
    idx = np.array([p.keys.index(key) for key in system.keys])
    T = np.array(
        [
            [temporal(ta, ka[1], tb, kb[1]) for tb, kb in zip(system.times, system.keys)]
            for ta, ka in zip(system.times, system.keys)
        ]
    )
    C = T * (W[idx] @ W[idx].T)
    C += np.diag([pars[("noise", *key[:2])] for key in system.keys])
    cross = np.stack(
        [
            np.array([temporal(q, None, t, key[1]) for q in query])[:, None] * W[i]
            for t, key, i in zip(system.times, system.keys, idx)
        ],
        axis=-1,
    ).reshape(len(query) * p.features, -1)
    distance = np.sqrt(3) * np.abs(query[:, None] - query) / p.length_scale
    prior = np.kron((1 + distance) * np.exp(-distance), np.eye(p.features))
    mean = cross @ np.linalg.solve(C, system.values - offsets[idx])
    covariance = prior - cross @ np.linalg.solve(C, cross.T)
    anchor = W[[p.keys.index(key) for key in model._factor_anchor_keys_]]
    Q, R = np.linalg.qr(anchor.T)
    Q *= np.where(np.diag(R) >= 0, 1.0, -1.0)[None, :]
    rotation = np.kron(np.eye(len(query)), Q)
    return mean.reshape(-1, p.features) @ Q, rotation.T @ covariance @ rotation


@pytest.mark.parametrize("features", [3, 5])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_joint_moments_preserve_cross_time_factor_covariance(features, algebra):
    model = trajectory_model(features, algebra)
    assert hasattr(model, "sample_latent"), "joint trajectory sampling missing"
    from multimodalsrm.bayesian.posterior_coordinates import rotations
    from multimodalsrm.bayesian.trajectories import joint_moments

    query = np.array([4.0, 4.6, 7.1])
    draws = model.parameter_draws_.reshape(-1, len(model.parameter_names_))
    ids = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Q = rotations(model.problem_, draws, model._factor_anchor_keys_, ids)
    evaluate = joint_moments(model.problem_, "train", query)
    for x, rotation in zip(draws, Q):
        mean, covariance = evaluate(x, rotation)
        expected_mean, expected_covariance = reference_joint(model, x, query)
        assert_allclose(mean, expected_mean, atol=1e-9)
        assert_allclose(covariance, expected_covariance, atol=1e-9)
        # These examples would expose independent-time or diagonal-factor draws.
        assert np.max(np.abs(expected_covariance[:features, features:])) > 1e-3
        assert (
            np.max(
                np.abs(
                    expected_covariance[:features, :features]
                    - np.diag(np.diag(expected_covariance[:features, :features]))
                )
            )
            > 1e-4
        )


def test_samples_reproduce_joint_mixture_and_keep_parameter_draw_identity():
    model = trajectory_model()
    assert hasattr(model, "sample_latent"), "joint trajectory sampling missing"
    query = np.array([4.0, 4.6, 7.1])
    before = model.parameter_draws_.copy()
    result = model.sample_latent(
        times=query,
        max_draws=None,
        draws_per_parameter=12000,
        random_state=913,
    )["train"]
    assert result.samples.shape == (4, 12000, 3, 3)
    expected_means, expected_covariances = [], []
    for samples, x in zip(result.samples, before.reshape(4, -1)):
        mean, covariance = reference_joint(model, x, query)
        flat = samples.reshape(len(samples), -1)
        mean_se = np.sqrt(np.diag(covariance) / len(samples))
        assert np.max(np.abs(flat.mean(0) - mean.ravel()) / mean_se) < 6
        cov_se = np.sqrt(
            (np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2)
            / (len(samples) - 1)
        )
        assert np.max(np.abs(np.cov(flat, rowvar=False) - covariance) / cov_se) < 6
        expected_means.append(mean.ravel())
        expected_covariances.append(covariance)
    total = np.mean(expected_covariances, axis=0) + np.cov(expected_means, rowvar=False, ddof=0)
    assert_allclose(np.cov(result.samples.reshape(-1, 9), rowvar=False), total, atol=0.004)
    assert result.metadata["parameter_draw_indices"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert result.metadata["sampling_diagnostics_passed"] is False
    assert result.metadata["calibration_established"] is False
    assert result.metadata["include_noise"] is False
    assert not result.samples.flags.writeable
    assert not result.times.flags.writeable
    assert_array_equal(model.parameter_draws_, before)


def test_seed_support_subsampling_and_fitted_orientation_are_preserved():
    model = trajectory_model(5)
    assert hasattr(model, "sample_latent"), "joint trajectory sampling missing"
    args = dict(
        times=[-1.0, 4.0, 7.1, 17.0],
        max_draws=2,
        draws_per_parameter=3,
        random_state=68,
    )
    result = model.sample_latent(**args)["train"]
    assert result.valid.tolist() == [False, True, True, False]
    assert np.isnan(result.samples[:, :, ~result.valid]).all()
    assert np.isfinite(result.samples[:, :, result.valid]).all()
    assert result.metadata["parameter_draw_indices"] == [[0, 0], [1, 0]]
    assert_array_equal(model.sample_latent(**args)["train"].samples, result.samples)
    model.features, model.factor_anchors, model.inference, model.linear_algebra = (
        1,
        (("bad", "key", 0),),
        "map",
        "state_space",
    )
    assert_array_equal(model.sample_latent(**args)["train"].samples, result.samples)
    args["random_state"] = 69
    assert not np.array_equal(
        model.sample_latent(**args)["train"].samples[:, :, 1:3],
        result.samples[:, :, 1:3],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"draws_per_parameter": 0},
        {"draws_per_parameter": True},
        {"draws_per_parameter": 1.5},
        {"max_joint_size": 0},
        {"max_joint_size": True},
        {"max_joint_size": 2},
        {"max_draws": 1},
    ],
)
def test_invalid_requests_fail_before_drawing(kwargs):
    model = trajectory_model()
    assert hasattr(model, "sample_latent"), "joint trajectory sampling missing"
    with pytest.raises(ValueError):
        model.sample_latent(times=[4.0, 7.0], **kwargs)


def test_bad_anchor_keeps_original_chain_draw_location():
    model = trajectory_model()
    assert hasattr(model, "sample_latent"), "joint trajectory sampling missing"
    for key in model._factor_anchor_keys_:
        for f in range(3):
            model.parameter_draws_[1, 0, model.problem_.indices[("loading", *key, f)]] = 0
    with pytest.raises(ValueError, match="rank deficient.*chain 1, draw 0"):
        model.sample_latent(times=[4.0], max_draws=2)


def test_numerical_failure_identifies_run_and_original_parameter_draw():
    model = trajectory_model()
    model.parameter_draws_[1, 0, model.problem_.indices[("noise", "a", "brain")]] = np.nan
    with pytest.raises(FloatingPointError, match="run 'train'.*chain 1, draw 0"):
        model.sample_latent(times=[4.0, 7.0], max_draws=2, random_state=78)


def test_semidefinite_sampling_does_not_add_noise_or_hide_bad_covariance():
    from multimodalsrm.bayesian import results

    assert hasattr(results, "TrajectorySamples"), "trajectory result missing"
    from multimodalsrm.bayesian.trajectories import covariance_root

    covariance = np.ones((2, 2))
    root = covariance_root(covariance)
    assert_allclose(root @ root.T, covariance, atol=1e-14)
    assert_allclose(root[0], root[1], atol=1e-14)
    assert_allclose(
        covariance_root(np.diag([1.0, -1e-15])) @ covariance_root(np.diag([1.0, -1e-15])).T,
        np.diag([1.0, 0.0]),
        atol=1e-14,
    )
    for bad in (
        np.diag([1.0, -1e-5]),
        np.full((2, 2), np.nan),
        np.array([[1.0, 0.1], [0.2, 1.0]]),
    ):
        with pytest.raises(FloatingPointError):
            covariance_root(bad)


def test_trajectory_result_copies_arrays_and_rejects_nonfinite_valid_samples():
    from multimodalsrm.bayesian import results

    assert hasattr(results, "TrajectorySamples"), "trajectory result missing"
    samples = np.ones((2, 3, 2, 5))
    metadata = {"nested": [1]}
    result = results.TrajectorySamples(samples, [1.0, 2.0], [True, False], metadata)
    samples[:] = 2
    metadata["nested"][0] = 2
    assert_array_equal(result.samples[:, :, 0], 1)
    assert np.isnan(result.samples[:, :, 1]).all()
    assert result.metadata == {"nested": [1]}
    bad = copy.deepcopy(samples)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        results.TrajectorySamples(bad, [1.0, 2.0], [True, False], {})


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_scalar_runs_share_parameters_but_have_independent_conditional_paths(algebra):
    from .test_bayesian_problem import api, independent_parameters, problem_fixture

    b = api()
    initial, adapter, _ = problem_fixture(two_runs=True)
    p = b.BayesianProblem(adapter, initial.priors, anchor=initial.anchor, linear_algebra=algebra)
    x = independent_parameters(p)
    shifted = x.copy()
    for i, name in enumerate(p.names):
        if name[0] == "offset":
            shifted[i] += 2.0
    model = b.BayesianMultimodalSRM(priors=p.priors)
    model.problem_, model.adapter_ = p, adapter
    model.prediction_runs_ = dict(adapter.domains_)
    model.parameter_draws_ = np.array([[x], [shifted]])
    model.targets_, model.sampling_diagnostics_ = None, {"passes": False}
    model.configuration_ = {
        "inference": "posterior",
        "uncertainty": "parameter_posterior_mixture",
        "reference_convention": p.reference_convention,
    }
    query = {"train": [4.0, 4.6], "second": [104.0, 104.6]}
    paths = model.sample_latent(times=query, draws_per_parameter=10000, random_state=461)
    moments = model.infer_latent(times=query)
    for run in query:
        mean = paths[run].samples.mean(axis=1)
        variance = paths[run].samples.var(axis=1)
        assert_allclose(mean, moments[run].component_means, atol=0.015)
        assert_allclose(variance, moments[run].component_variances, atol=0.005)
        assert paths[run].metadata["parameter_draw_indices"] == [[0, 0], [1, 0]]
    for i in range(2):
        a = paths["train"].samples[i, :, :, 0]
        c = paths["second"].samples[i, :, :, 0]
        cross_correlation = np.corrcoef(np.column_stack((a, c)), rowvar=False)[:2, 2:]
        assert np.max(np.abs(cross_correlation)) < 0.06
    # Marginal dependence across runs comes from their shared parameter index.
    means_a = moments["train"].component_means[:, 0, 0]
    means_b = moments["second"].component_means[:, 0, 0]
    expected = np.mean((means_a - means_a.mean()) * (means_b - means_b.mean()))
    assert expected > 0.05
    actual = np.cov(
        paths["train"].samples[:, :, 0, 0].ravel(),
        paths["second"].samples[:, :, 0, 0].ravel(),
    )[0, 1]
    assert_allclose(actual, expected, atol=0.01)


def test_nearly_coincident_queries_and_all_invalid_support():
    model = trajectory_model(5)
    paths = model.sample_latent(times=[4.0, 4.0 + 1e-12], random_state=42)["train"]
    assert_allclose(paths.samples[:, :, 0], paths.samples[:, :, 1], atol=2e-7)
    outside = model.sample_latent(times=[-2.0, -1.0], random_state=42)["train"]
    assert not outside.valid.any()
    assert np.isnan(outside.samples).all()


@pytest.mark.parametrize("unsupported", ["map", "state_space", "noise", "baseline", "quadrature"])
def test_unsupported_fitted_targets_are_rejected(unsupported):
    model = trajectory_model()
    if unsupported == "map":
        model.configuration_["inference"] = "map"
    elif unsupported == "state_space":
        model.problem_.linear_algebra = "state_space"
    elif unsupported == "noise":
        model.problem_.noise_timescales = {"brain": 1.0}
    elif unsupported == "baseline":
        model.problem_.run_baseline_sd = {"brain": 1.0}
    else:
        model.problem_.response_quadrature = object()
    with pytest.raises(ValueError, match="posterior"):
        model.sample_latent(times=[4.0, 7.0])
