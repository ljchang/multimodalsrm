"""Independent densities catch normalization, coordinate and covariance errors."""

import importlib
import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import lognorm, multivariate_normal, norm, truncnorm

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

from ..reference.continuous_covariance import response_covariance


def api():
    assert importlib.util.find_spec("multimodalsrm.bayesian") is not None, (
        "Bayesian package missing"
    )
    pytest.importorskip("jax").config.update("jax_enable_x64", True)
    pytest.importorskip("numpyro")
    pytest.importorskip("arviz")
    return importlib.import_module("multimodalsrm.bayesian")


def problem_fixture(gaussian=False, two_runs=False):
    b = api()
    t = np.array([0.0, 1.3, 3.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0])
    data = {
        "a": {"train": {"ref": TimeSeries(np.sin(t / 3)[:, None], t)}},
        "b": {"train": {"signal": TimeSeries(np.cos(t / 3)[:, None], t)}},
    }
    responses = {"ref": Response(Identity(), pooling="shared", estimate=False)}
    filters = {}
    if gaussian:
        responses["signal"] = Response(
            Gaussian(0.4, 0.5),
            pooling="shared",
            bounds={"width": (0.2, 0.7), "lag": (0.0, 1.0)},
        )
        filters = {
            "signal": {
                "width": b.Prior.normal(0.4, 0.2),
                "lag": b.Prior.normal(0.5, 0.4),
            }
        }
    if two_runs:
        for s in data:
            data[s]["second"] = {
                m: TimeSeries(ts.values * 0.7, ts.times + 100.0)
                for m, ts in data[s]["train"].items()
            }
    adapter = BayesianObservationAdapter(
        features=1, latent_dt=1.0, responses=responses, standardize=False
    )
    adapter._prepare(data)
    priors = b.BayesianPriors(
        loading_sd=1.5,
        offset_sd=0.8,
        noise=b.Prior.lognormal(np.log(0.2), 0.7),
        filters=filters,
    )
    p = b.BayesianProblem(adapter, priors, anchor=("a", "ref", 0))
    return p, adapter, data


def independent_parameters(p):
    values = []
    for name in p.names:
        kind = name[0]
        values.append(
            {
                "loading": 1.2 if name[1] == "a" else -0.7,
                "offset": 0.12 if name[1] == "a" else -0.2,
                "noise": 0.15 if name[1] == "a" else 0.3,
                "filter": 0.45 if name[-1] == "width" else 0.6,
            }[kind]
        )
    return np.array(values)


@pytest.mark.parametrize("gaussian", [False, True])
def test_likelihood_matches_independent_continuous_integrator(gaussian):
    p, adapter, _ = problem_fixture(gaussian, two_runs=True)
    x = independent_parameters(p)
    expected = 0.0
    for run, system in p.systems.items():
        kernels = {
            "ref": Identity(),
            "signal": Gaussian(0.45, 0.6) if gaussian else Identity(),
        }
        means = np.array([0.12 if s == "a" else -0.2 for s, _, _ in system.keys])
        weights = np.array([1.2 if s == "a" else -0.7 for s, _, _ in system.keys])
        noises = np.array([0.15 if s == "a" else 0.3 for s, _, _ in system.keys])
        C = np.array(
            [
                [
                    response_covariance(
                        [t], [u], kernels[k[1]], kernels[j[1]], 3.0, tolerance=1e-9
                    )[0, 0]
                    for u, j in zip(system.times, system.keys)
                ]
                for t, k in zip(system.times, system.keys)
            ]
        )
        C = C * np.outer(weights, weights) + np.diag(noises)
        assert_allclose(p.covariance(x, run), C, atol=1e-8, rtol=1e-8)
        expected -= multivariate_normal.logpdf(system.values, mean=means, cov=C)
    assert_allclose(p.nll(x), expected, atol=1e-7)


def test_normalized_priors_are_on_physical_noise_variance():
    p, _, _ = problem_fixture(True)
    x = independent_parameters(p)
    expected = 0.0
    for name, v in zip(p.names, x):
        if name[0] == "loading":
            expected += norm.logpdf(v, scale=1.5) + (np.log(2) if name[1] == "a" else 0.0)
        elif name[0] == "offset":
            expected += norm.logpdf(v, scale=0.8)
        elif name[0] == "noise":
            expected += lognorm.logpdf(v, s=0.7, scale=0.2)
        elif name[-1] == "width":
            expected += truncnorm.logpdf(v, -1.0, 1.5, loc=0.4, scale=0.2)
        else:
            expected += truncnorm.logpdf(v, -1.25, 1.25, loc=0.5, scale=0.4)
    assert_allclose(p.log_prior(x), expected, atol=1e-11)
    assert_allclose(p.objective(x), p.nll(x) - expected, atol=1e-11)


def test_unconstrained_density_has_the_physical_jacobian():
    p, _, _ = problem_fixture(True)
    x = independent_parameters(p)
    z = p.to_unconstrained(x)
    recovered, jac = p.from_unconstrained(z)
    h = 1e-5
    J = np.column_stack(
        [
            (
                np.asarray(p.from_unconstrained(z + np.eye(len(z))[j] * h)[0])
                - np.asarray(p.from_unconstrained(z - np.eye(len(z))[j] * h)[0])
            )
            / (2 * h)
            for j in range(len(z))
        ]
    )
    assert_allclose(recovered, x, atol=1e-12)
    assert_allclose(jac, np.linalg.slogdet(J)[1], atol=1e-8)
    assert_allclose(p.potential(z), p.objective(x) - jac, atol=1e-11)
    assert not np.isclose(float(p.potential(z)), float(p.objective(x)))


def test_joint_gradient_matches_finite_difference():
    p, _, _ = problem_fixture(True)
    x = independent_parameters(p)
    _, actual = p.value_gradient(x)
    h = 1e-5
    expected = [
        (float(p.objective(x + e * h)) - float(p.objective(x - e * h))) / (2 * h)
        for e in np.eye(len(x))
    ]
    assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


def test_prior_draws_support_and_normalization():
    b = api()
    prior = b.Prior.lognormal(np.log(0.2), 0.7).bounded(0.05, 0.7)
    grid = np.linspace(0.05, 0.7, 20001)
    assert_allclose(
        np.trapezoid(np.exp(np.asarray(prior.distribution().log_prob(grid))), grid),
        1.0,
        atol=1e-7,
    )
    q = prior.ppf(np.array([0.1, 0.5, 0.9]))
    cdf = lognorm.cdf(q, s=0.7, scale=0.2)
    lo, hi = lognorm.cdf([0.05, 0.7], s=0.7, scale=0.2)
    assert_allclose((cdf - lo) / (hi - lo), [0.1, 0.5, 0.9], atol=1e-12)
    uniform = b.Prior.uniform(-2.0, 3.0)
    assert_allclose(uniform.distribution().log_prob(0.3), -np.log(5.0))


def test_filter_priors_remain_required_and_sign_anchor_can_be_filtered():
    p, adapter, _ = problem_fixture(True)
    b = api()
    with pytest.raises(ValueError, match="filter priors"):
        b.BayesianProblem(
            adapter,
            b.BayesianPriors(noise=b.Prior.lognormal(-2.0, 1.0)),
            anchor=("a", "ref", 0),
        )
    filtered = b.BayesianProblem(adapter, p.priors, anchor=("b", "signal", 0))
    assert filtered.anchor == ("b", "signal", 0)
    assert filtered.reference_convention["reference_modality"] == "ref"
    assert filtered.reference_convention["mode"] == "automatic_fixed_lag"


def test_accuracy_guard_does_not_silently_use_full_gaussian():
    p, adapter, _ = problem_fixture(True)
    adapter.covariance_tolerance = 1e-12
    with pytest.raises(ValueError, match="accuracy"):
        api().BayesianProblem(adapter, p.priors, anchor=("a", "ref", 0))


def test_bounded_lognormal_support_is_enforced_in_density_and_transform():
    p, adapter, _ = problem_fixture()
    b = api()
    bounded = b.BayesianPriors(noise=b.Prior.lognormal(-2.0, 1.0).bounded(0.05, 0.7))
    p = b.BayesianProblem(adapter, bounded, anchor=("a", "ref", 0))
    x = p.initial.copy()
    noise = p.indices[("noise", "a", "ref")]
    x[noise] = 3.0
    assert np.isneginf(float(p.log_prior(x)))
    for value in (-20.0, 20.0):
        z = np.zeros(len(p.names))
        z[noise] = value
        physical, _ = p.from_unconstrained(z)
        assert 0.05 < physical[noise] < 0.7
    with pytest.raises(ValueError, match="support"):
        p.to_unconstrained(x)


@pytest.mark.parametrize("width,ell", [(0.01, 1.3), (2.5, 1.3), (2.5, 30.0)])
def test_covariance_accuracy_near_supported_width_timescale_extremes(width, ell):
    api()
    import jax.numpy as jnp

    from multimodalsrm.bayesian._backend import temporal

    t = np.array([0.0, 0.01, 0.2, 1.0, 5.0, 25.0])
    observed = temporal(
        jnp.array(t),
        jnp.array(t),
        jnp.full(t.shape, width),
        jnp.full(t.shape, width),
        jnp.zeros_like(t),
        jnp.zeros_like(t),
        ell,
    )
    expected = response_covariance(t, t, Gaussian(width), Gaussian(width), ell, tolerance=1e-9)
    assert_allclose(observed, expected, atol=4e-8, rtol=1e-8)


def test_multiple_features_share_mapping_noise_but_have_separate_loadings():
    p, adapter, data = problem_fixture()
    ts = data["a"]["train"]["ref"]
    mask = np.ones((len(ts.times), 2), dtype=bool)
    mask[1, 1] = False
    data["a"]["train"]["ref"] = TimeSeries(
        np.c_[ts.values[:, 0], 2 * ts.values[:, 0]], ts.times, mask
    )
    adapter._prepare(data)
    p = api().BayesianProblem(adapter, p.priors, anchor=("a", "ref", 0))
    assert len([n for n in p.names if n[0] == "noise"]) == 2
    assert len([n for n in p.names if n[0] == "loading"]) == 3
    assert len(p.systems["train"].times) == 26
    x = p.initial.copy()
    x[p.indices[("loading", "a", "ref", 0)]] = 1.0
    x[p.indices[("loading", "a", "ref", 1)]] = 2.0
    C = np.asarray(p.covariance(x, "train", include_noise=False))
    assert_allclose(C[0, 9], 2.0)  # different features of the exact same latent at t=0


@pytest.mark.parametrize("lo,hi", [(8.0, 9.0), (9.0, 10.0), (-10.0, -9.0), (24.0, 34.0)])
def test_far_tail_truncation_has_a_normalized_density(lo, hi):
    b = api()
    prior = b.Prior.normal().bounded(lo, hi)
    x = prior.ppf(np.array([0.1, 0.5, 0.9]))
    assert_allclose(prior.distribution().log_prob(x), truncnorm.logpdf(x, lo, hi), atol=1e-10)


def test_unrepresentable_one_sided_prior_fails_before_search():
    b = api()
    with pytest.raises(ValueError, match="normalization"):
        b.Prior.normal().bounded(40.0, np.inf).distribution()
