"""Independent Gaussian calculations exercise multiple shared factors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import lognorm, multivariate_normal, norm

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

from ..reference.continuous_covariance import response_covariance
from .test_bayesian_problem import api


def fixture(features=2, algebra="dense", gaussian=True):
    b = api()
    t = np.arange(0.0, 18.0, 2.0)
    u = np.arange(0.0, 18.0, 1.5)
    rng = np.random.default_rng(401)
    data = {}
    for subject in ("a", "b", "c"):
        values = rng.normal(size=(len(t), 3))
        mask = np.ones_like(values, dtype=bool)
        mask[3, 1] = False
        data[subject] = {"train": {"brain": TimeSeries(values, t, mask)}}
        if subject != "c":
            data[subject]["train"]["aux"] = TimeSeries(rng.normal(size=(len(u), 2)), u)
    responses = {"brain": Response(Identity(), pooling="shared", estimate=False)}
    filters = {}
    if gaussian:
        responses["aux"] = Response(
            Gaussian(0.5, 0.3),
            pooling="shared",
            bounds={"width": (0.2, 0.8), "lag": (-0.5, 0.8)},
        )
        filters = {"aux": {"width": b.Prior.normal(0.5, 0.3), "lag": b.Prior.normal(0.3, 0.4)}}
    adapter = BayesianObservationAdapter(
        features=features,
        latent_dt=1.0,
        responses=responses,
        standardize=False,
        max_observations=400,
    )
    adapter._prepare(data)
    priors = b.BayesianPriors(
        loading_sd=1.3,
        offset_sd=0.8,
        noise=b.Prior.lognormal(np.log(0.2), 0.7),
        filters=filters,
    )
    problem = b.BayesianProblem(adapter, priors, anchor=("a", "brain", 0), linear_algebra=algebra)
    x = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            x[i] = rng.normal()
            if features == 1 and name[1:] == problem.anchor:
                x[i] = abs(x[i])
        elif name[0] == "offset":
            x[i] = rng.normal(scale=0.1)
        elif name[0] == "noise":
            x[i] = 0.1 + rng.uniform()
    return problem, x, data


def independent(problem, x, run="train"):
    """Extract physical coordinates by names; never call arrays/covariance."""
    params = dict(zip(problem.names, x))
    k = problem.adapter.features
    weights = np.array([[params[("loading", *key, f)] for f in range(k)] for key in problem.keys])
    offsets = np.array([params[("offset", *key)] for key in problem.keys])
    kernels = {}
    for m, response in problem.responses.items():
        initial = response.initial_kernel()
        kernels[m] = (
            Identity()
            if isinstance(initial, Identity)
            else Gaussian(
                params.get(("filter", m, "width"), initial.width),
                params.get(("filter", m, "lag"), initial.lag),
            )
        )
    system = problem.systems[run]
    indices = np.array([problem.keys.index(key) for key in system.keys])
    temporal = np.empty((len(indices), len(indices)))
    for a in problem.modalities:
        ia = np.flatnonzero([key[1] == a for key in system.keys])
        for b in problem.modalities:
            ib = np.flatnonzero([key[1] == b for key in system.keys])
            temporal[np.ix_(ia, ib)] = response_covariance(
                system.times[ia],
                system.times[ib],
                kernels[a],
                kernels[b],
                problem.length_scale,
                tolerance=1e-9,
            )
    covariance = temporal * (weights[indices] @ weights[indices].T)
    covariance += np.diag([params[("noise", *key[:2])] for key in system.keys])
    return covariance, weights, offsets, indices, kernels


def log_prior(problem, x):
    total = 0.0
    for p, value in zip(problem.parameter_priors, x):
        distribution = (
            lognorm(p.scale, scale=np.exp(p.loc))
            if p.family == "lognormal"
            else norm(p.loc, p.scale)
        )
        total += distribution.logpdf(value) - np.log(
            distribution.cdf(p.upper) - distribution.cdf(p.lower)
        )
    return total


def test_multifactor_density_and_derivatives_match_independent_gaussian():
    problem, x, _ = fixture()
    covariance, _, offsets, indices, _ = independent(problem, x)
    observed = problem.systems["train"].values
    expected = -multivariate_normal.logpdf(observed, mean=offsets[indices], cov=covariance)
    assert_allclose(problem.nll(x), expected, atol=1e-8)
    prior = log_prior(problem, x)
    assert_allclose(problem.log_prior(x), prior, atol=1e-9)
    _, gradient = problem.value_gradient(x)
    # Independently recompute the likelihood, including quadrature, at probes.
    for i in range(len(x)):
        h = 1e-5
        values = []
        for sign in (-1, 1):
            probe = x.copy()
            probe[i] += sign * h
            C, _, mu, idx, _ = independent(problem, probe)
            likelihood = -multivariate_normal.logpdf(observed, mean=mu[idx], cov=C)
            lp = log_prior(problem, probe)
            values.append(likelihood - lp)
        assert_allclose(gradient[i], (values[1] - values[0]) / (2 * h), atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("gaussian", [False, True])
def test_multifactor_grouped_agrees_including_zero_loadings(zero, gaussian):
    dense, x, _ = fixture(gaussian=gaussian)
    grouped, _, _ = fixture(algebra="grouped", gaussian=gaussian)
    if zero:
        for i, name in enumerate(dense.names):
            if name[0] == "loading" and name[1] == "b":
                x[i] = 0.0
    assert_allclose(grouped.value_gradient(x)[0], dense.value_gradient(x)[0], atol=1e-8)
    assert_allclose(grouped.value_gradient(x)[1], dense.value_gradient(x)[1], atol=1e-7)


def test_multifactor_initialization_has_distinct_loading_directions():
    from multimodalsrm.bayesian.fitting import initial_points

    problem, _, _ = fixture()
    x = initial_points(problem, 1, 21)[0]
    W = np.asarray(problem.arrays(x)[0])
    assert W.shape == (len(problem.keys), 2)
    assert np.linalg.matrix_rank(W) == 2
    assert_array_equal(x, initial_points(problem, 1, 21)[0])


def test_multifactor_grouped_variance_repair_is_available():
    from multimodalsrm.bayesian.noise_profile import variance_profile

    p, x, _ = fixture(algebra="grouped")
    profile = variance_profile(p, x, p.indices[("noise", "a", "brain")])
    assert profile.available, profile.reason
    assert_allclose(
        profile.gradient,
        p.value_gradient(x)[1][p.indices[("noise", "a", "brain")]],
        atol=1e-7,
    )


def test_training_orientation_is_invariant_and_rejects_missing_rank():
    from multimodalsrm.bayesian.multifactor import orientation

    p, x, _ = fixture()
    anchors = (("a", "brain", 0), ("a", "brain", 1))
    result = orientation(p, x, anchors)
    Q = np.asarray(result["rotation"])
    W = np.asarray(p.arrays(x)[0])
    assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
    block = W[[p.keys.index(a) for a in anchors]] @ Q
    assert_allclose(block[0, 1], 0.0, atol=1e-12)
    assert (np.diag(block) > 0).all()
    Z = np.random.default_rng(71).normal(size=(10, 2))
    assert_allclose((Z @ Q) @ (W @ Q).T, Z @ W.T, atol=1e-12)
    rotated = x.copy()
    rotated[: W.size] = (W @ Q).ravel()
    assert_allclose(p.objective(rotated), p.objective(x), atol=1e-8)
    for key in anchors:
        for factor in range(2):
            x[p.indices[("loading", *key, factor)]] = 1.0
    with pytest.raises(ValueError, match="rank"):
        orientation(p, x, anchors)


@pytest.mark.parametrize("key", [None, ("b", "aux", 1)])
def test_multifactor_conditional_moments_match_separate_solve(key):
    from multimodalsrm.bayesian.prediction import project

    p, x, _ = fixture()
    grouped, _, _ = fixture(algebra="grouped")
    C, W, offsets, idx, kernels = independent(p, x)
    query = np.array([4.0, 7.1, 10.0])
    if key is None:
        loading, offset, kernel, extra = np.array([0.6, 0.8]), 0.0, Identity(), 0.0
    else:
        index = p.keys.index(key)
        loading, offset, kernel = W[index], offsets[index], kernels[key[1]]
        extra = x[p.indices[("noise", *key[:2])]]
    system = p.systems["train"]
    cross = np.column_stack(
        [
            response_covariance(query, [t], kernel, kernels[k[1]], p.length_scale, tolerance=1e-10)[
                :, 0
            ]
            * np.dot(loading, W[i])
            for t, k, i in zip(system.times, system.keys, idx)
        ]
    )
    expected_mean = offset + cross @ np.linalg.solve(C, system.values - offsets[idx])
    prior = np.diag(
        response_covariance(query, query, kernel, kernel, p.length_scale, tolerance=1e-10)
    )
    expected_var = (
        prior * np.dot(loading, loading)
        + extra
        - np.sum(cross * np.linalg.solve(C, cross.T).T, axis=1)
    )
    for problem in (p, grouped):
        mean, variance = project(
            problem,
            x[None],
            "train",
            query,
            key=key,
            include_noise=key is not None,
            latent_loading=loading if key is None else None,
        )
        assert_allclose(mean[0], expected_mean, atol=1e-8)
        assert_allclose(variance[0], expected_var, atol=1e-8)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"features": 0}, "features"),
        ({"features": True}, "features"),
        ({"features": 1.5}, "features"),
        ({"inference": "posterior", "factor_anchors": None}, "factor_anchors"),
        ({"inference": "posterior", "linear_algebra": "state_space"}, "MAP"),
        ({"inference": "posterior", "linear_algebra": "spectral"}, "dense/grouped"),
        ({"inference": "posterior", "noise_timescales": {"aux": 1.0}}, "MAP"),
        ({"inference": "posterior", "run_baseline_sd": {"aux": 0.2}}, "one factor"),
        ({"linear_algebra": "spectral"}, "dense/grouped"),
        ({"factor_anchors": ()}, "factor_anchors"),
        ({"factor_anchors": (("a", "brain", 0), ("a", "brain", 0))}, "distinct"),
        ({"factor_anchors": (("a", "brain", 0), ("a", "brain", 9))}, "observed"),
        ({"run_baseline_sd": {"aux": 0.2}}, "one factor"),
    ],
)
def test_multifactor_unsupported_combinations_fail_before_search(options, message, monkeypatch):
    from multimodalsrm.bayesian import model as module

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid configuration reached optimization")

    monkeypatch.setattr(module, "search", forbidden)
    p, _, data = fixture()
    config = dict(
        features=2,
        inference="map",
        priors=p.priors,
        responses=p.responses,
        anchor=p.anchor,
        factor_anchors=(("a", "brain", 0), ("a", "brain", 1)),
    )
    config.update(options)
    with pytest.raises(ValueError, match=message):
        api().BayesianMultimodalSRM(**config).fit(data)


@pytest.fixture(scope="module")
def fitted():
    b = api()
    p, _, data = fixture(gaussian=False)
    model = b.BayesianMultimodalSRM(
        features=2,
        priors=p.priors,
        responses=p.responses,
        anchor=p.anchor,
        factor_anchors=(("a", "brain", 0), ("a", "brain", 1)),
        inference="map",
        linear_algebra="grouped",
        max_observations=400,
        random_state=781,
        search=b.SearchConfig(starts=2, maxiter=500, refine_maxiter=60),
    ).fit(data)
    return model, data


def test_multifactor_independent_transform_freezes_parameters_and_orientation(
    fitted, monkeypatch, tmp_path
):
    import copy

    from sklearn.base import clone

    from multimodalsrm import temporal_isc
    from multimodalsrm.bayesian import model as module
    from multimodalsrm.bayesian.prediction import project
    from multimodalsrm.bayesian.workflow import load_results, save_results

    model, data = fitted
    assert clone(model).features == 2
    assert model.configuration_["features"] == 2
    assert model.configuration_["reference_convention"]["positive_loading_anchor"] is None
    assert model.configuration_["linear_system_sizes"]["train"]["factor_functionals"] > 1
    original = model.map_parameters_.copy()
    Q = np.asarray(model.configuration_["factor_orientation"]["rotation"])

    def forbidden(*args, **kwargs):
        raise AssertionError("independent inference refitted parameters")

    monkeypatch.setattr(module, "search", forbidden)
    test = {s: {"test": {"brain": mods["train"]["brain"]}} for s, mods in data.items()}
    query = np.arange(2.0, 16.0, 2.0)
    results = model.transform(test, times=query, modalities=["brain"])
    for s, runs in results.items():
        result = runs["test"]
        assert result.values.shape == (len(query), 2)
        assert result.metadata["factor_orientation"] == model.configuration_["factor_orientation"]
        adapter = copy.deepcopy(model.adapter_)
        _, domains = adapter._grids({s: test[s]})
        systems, _ = adapter._systems({s: test[s]}, domains)
        p = api().BayesianProblem(adapter, model.priors, anchor=model.anchor, systems=systems)
        for f in range(2):
            mean, variance = project(
                p,
                original[None],
                "test",
                query,
                key=None,
                include_noise=False,
                latent_loading=Q[:, f],
            )
            assert_allclose(result.values[:, f], mean[0], atol=1e-8)
            assert_allclose(result.variance[:, f], variance[0], atol=1e-8)
    changed = copy.deepcopy(test)
    ts = changed["b"]["test"]["brain"]
    changed["b"]["test"]["brain"] = TimeSeries(ts.values * -100, ts.times, ts.mask)
    other = model.transform(changed, times=query, modalities=["brain"])
    assert_array_equal(results["a"]["test"].values, other["a"]["test"].values)
    assert_array_equal(model.map_parameters_, original)
    flat = {s: runs["test"] for s, runs in results.items()}
    assert temporal_isc(flat)["n_features"] == 2
    destination = tmp_path / "results"
    save_results(
        destination,
        {(s, "test", "latent"): runs["test"] for s, runs in results.items()},
    )
    restored, _ = load_results(destination)
    for s in results:
        assert_array_equal(restored[(s, "test", "latent")].values, results[s]["test"].values)
        from multimodalsrm.bayesian.workflow import _json

        assert restored[(s, "test", "latent")].metadata == _json(results[s]["test"].metadata)
    altered = copy.deepcopy(flat)
    altered["b"].metadata["reference_convention"]["factor_orientation"]["rotation"][0][0] += 0.1
    with pytest.raises(ValueError, match="reference convention"):
        temporal_isc(altered)
    with pytest.raises(ValueError, match="frozen"):
        model.condition(test, targets={"c": ["brain"]})
    conditioned = model.condition(test, targets={"c": ["brain"]}, mode="frozen")
    assert (
        conditioned.configuration_["factor_orientation"]
        == model.configuration_["factor_orientation"]
    )
    prediction = conditioned.predict(times=query)["c"]["test"]["brain"]
    assert prediction.values.shape == (len(query), 3)
