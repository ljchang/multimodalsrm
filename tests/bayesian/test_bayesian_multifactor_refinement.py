"""Independent multifactor variance calculations and bounded refinement work."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult

from .test_bayesian_multifactor import fixture, independent
from .test_bayesian_noise_profile import conditional_change, stiff_fixture


@pytest.mark.parametrize("factors", [2, 3])
@pytest.mark.parametrize("family", ["normal", "lognormal", "uniform"])
def test_multifactor_profiles_match_independent_observation_density(factors, family):
    from multimodalsrm.bayesian.noise_profile import variance_profile

    from .test_bayesian_problem import api

    b = api()
    p, x, data = fixture(features=factors, algebra="grouped")
    # A second run, missing participant/modality and masked observations.
    for subject, runs in data.items():
        runs["second"] = deepcopy(runs["train"])
    del data["a"]["second"]["aux"]
    p.adapter._prepare(data)
    prior = {
        "normal": b.Prior.normal(0.3, 0.5).bounded(0.01, 2),
        "lognormal": b.Prior.lognormal(-1.0, 0.7).bounded(0.01, 2),
        "uniform": b.Prior.uniform(0.01, 2),
    }[family]
    p = b.BayesianProblem(
        p.adapter,
        b.BayesianPriors(noise=prior, filters=p.priors.filters),
        anchor=p.anchor,
        linear_algebra="grouped",
    )
    j = p.indices[("noise", "a", "brain")]

    # The oracle extracts physical names and uses independent temporal quadrature.
    def oracle(point, run):
        C, _, mu, indices, _ = independent(p, point, run)
        return C, p.systems[run].values - mu[indices]

    gradient, curvature = 0.0, 0.0
    for run, system in p.systems.items():
        C, residual = oracle(x, run)
        selected = np.array([key[:2] == ("a", "brain") for key in system.keys])
        P = np.linalg.inv(C)
        a = (P @ residual)[selected]
        block = P[np.ix_(selected, selected)]
        gradient += 0.5 * np.trace(block) - 0.5 * a @ a
        curvature += -0.5 * np.sum(block**2) + a @ block @ a
    if family == "normal":
        gradient += (x[j] - prior.loc) / prior.scale**2
        curvature += 1 / prior.scale**2
    elif family == "lognormal":
        a = np.log(x[j]) - prior.loc
        gradient += (1 + a / prior.scale**2) / x[j]
        curvature += (1 / prior.scale**2 - 1 - a / prior.scale**2) / x[j] ** 2
    profile = variance_profile(p, x, j)
    assert profile.available, profile.reason
    assert_allclose(
        [profile.gradient, profile.curvature],
        [gradient, curvature],
        rtol=1e-9,
        atol=1e-8,
    )
    # Use independent covariances in the high-precision conditional-density oracle.
    oracle_problem = SimpleNamespace(
        names=p.names,
        systems=p.systems,
        parameter_priors=p.parameter_priors,
        arrays=lambda point: (None, np.zeros(len(p.keys)), None, None, None),
        covariance=lambda point, run: oracle(point, run)[0],
        _packed=p._packed,
    )
    oracle_problem.systems = {
        run: SimpleNamespace(keys=s.keys, values=oracle(x, run)[1]) for run, s in p.systems.items()
    }
    for delta in (-0.02, -1e-10, 1e-10, 0.01):
        assert_allclose(
            profile.change(delta),
            conditional_change(oracle_problem, x, j, delta),
            atol=1e-18,
            rtol=1e-9,
        )
    assert np.isinf(profile.change(2.0))


@pytest.mark.parametrize("factors", [2, 3])
def test_multifactor_solver_handles_singular_covariance_and_rotations(factors, monkeypatch):
    from multimodalsrm.bayesian import grouped, noise_profile

    rng = np.random.default_rng(617)
    nodes = np.tile(np.arange(5), 9)
    temporal = np.ones((5, 5))  # rank one, no inverse available
    weights = rng.normal(size=(len(nodes), factors))
    weights[:, -1] = 0.0  # unobserved factor direction
    weights[:5] = 0.0
    weights[10:15] = -weights[5:10]  # cancelling observations at the same nodes
    variance = np.geomspace(1e-5, 0.4, len(nodes))
    residual = rng.normal(size=len(nodes))
    rhs = rng.normal(size=(len(nodes), 7))
    C = temporal[np.ix_(nodes, nodes)] * (weights @ weights.T) + np.diag(variance)
    expected = np.linalg.solve(C, rhs)
    rotation = np.linalg.qr(rng.normal(size=(factors, factors)))[0]
    for W in (weights, weights @ rotation):
        monkeypatch.setattr(
            grouped, "operands", lambda *args: (temporal, W, residual, variance, nodes)
        )
        detail = {}
        solve, _ = noise_profile._grouped_solver(
            SimpleNamespace(features=factors, baseline_designs={}),
            None,
            "train",
            detail,
        )
        assert_allclose(solve(rhs), expected, atol=1e-5, rtol=1e-8)
        assert detail["factor_system_rows"] == 5 * factors
        assert detail["temporal_nodes"] == 5
        assert np.isfinite(detail["maximum_relative_solve_residual"])


def test_multifactor_profile_guards_expanded_allocation(monkeypatch):
    from multimodalsrm.bayesian import noise_profile

    p, x, _ = fixture(algebra="grouped")
    monkeypatch.setattr(noise_profile, "MAX_FACTOR_ROWS", 1)
    monkeypatch.setattr(
        noise_profile,
        "lu_factor",
        lambda *a, **k: pytest.fail("capacity must precede factorization"),
    )
    profile = noise_profile.variance_profile(p, x, p.indices[("noise", "a", "brain")])
    assert not profile.available and "factor-system capacity" in profile.reason


def test_two_stiff_variances_are_repaired_without_full_hessian():
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import polish

    from .test_bayesian_problem import api

    old, _, _ = stiff_fixture("grouped", v=1.5e-5)
    old.adapter.features = 2
    p = api().BayesianProblem(old.adapter, old.priors, anchor=old.anchor, linear_algebra="grouped")
    x = np.zeros(len(p.names))
    noise = [j for j, n in enumerate(p.names) if n[0] == "noise"]
    x[noise] = 1.5e-5
    x[noise[0]] += 1e-10
    x[noise[1]] += 2e-10
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    assert np.sum(abs(p.value_gradient(x)[1]) > 0.001) == 2
    report = polish(p, record, config, 3)
    assert record["meets_gradient_tolerance"], record
    assert report["gradient_evaluations"] <= 6
    assert report["accepted_steps"] <= 3
    for j in noise[:2]:
        assert conditional_change(p, x, j, record["parameters"][j] - x[j]) < 0


def test_multifactor_refinement_escapes_saturated_filter_transform():
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
        _refine,
    )

    jax, jnp, _, _ = runtime()

    # True constrained minimum at width=1. Logistic gradient is tiny at the
    # wrong boundary even while its physical gradient is large and inward.
    def objective(x):
        return 0.5 * ((x[0] - 2.0) ** 2 + (x[1] - 0.1) ** 2)

    def from_z(z):
        return jnp.array([jax.nn.sigmoid(z[0]), z[1]]), 0.0

    vg = jax.jit(jax.value_and_grad(objective))
    p = SimpleNamespace(
        features=2,
        names=[("filter", "aux", "width"), ("offset",)],
        bounds=[(0.0, 1.0), (-np.inf, np.inf)],
        objective=objective,
        value_gradient=lambda x: (float(vg(x)[0]), np.asarray(vg(x)[1])),
        to_unconstrained=lambda x: np.array([np.log(x[0] / (1 - x[0])), x[1]]),
        from_unconstrained=from_z,
    )
    transformed = jax.jit(jax.value_and_grad(lambda z: objective(from_z(z)[0])))

    def evaluate(z):
        return float(transformed(z)[0]), np.asarray(transformed(z)[1])

    x = np.array([1e-10, 0.1])
    config = SearchConfig(refine_maxiter=2)
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    record["elapsed_seconds"] = 0.0
    _refine(p, record, config, evaluate)
    assert record["meets_gradient_tolerance"], record
    assert_allclose(record["parameters"], [1.0, 0.1], atol=1e-8)
    assert record["refinement"]["coordinates"] == "physical_filter_boxes_other_unconstrained"
    assert record["refinement"]["iterations"] <= 2
    assert record["objective"] == float(objective(np.array(record["parameters"])))


@pytest.mark.parametrize("edge", [None, 0, 1])
def test_real_gaussian_filter_boxes_keep_density_and_chain_rule(edge):
    from multimodalsrm.bayesian.refinement_coordinates import (
        filter_coordinates,
    )

    p, x, _ = fixture(algebra="grouped")
    filters = [j for j, n in enumerate(p.names) if n[0] == "filter"]
    if edge is not None:
        for j in filters:
            x[j] = p.bounds[j][edge]
    initial, physical, evaluate, bounds, indices = filter_coordinates(p, x)
    assert indices == filters
    assert_allclose(physical(initial), x, atol=1e-15)
    value, gradient = evaluate(initial)
    physical_value, physical_gradient = p.value_gradient(x)
    assert_allclose(value, physical_value, atol=1e-9)
    assert np.isfinite(gradient).all()
    assert_allclose(gradient[filters], physical_gradient[filters], atol=1e-8)
    for j, name in enumerate(p.names):
        if name[0] == "filter":
            assert bounds[j] == p.bounds[j]
            continue
        assert bounds[j] == (None, None)
        plus, minus = initial.copy(), initial.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        derivative = (physical(plus)[j] - physical(minus)[j]) / 2e-5
        assert_allclose(gradient[j], physical_gradient[j] * derivative, rtol=1e-7, atol=1e-7)
    x[filters[0]] = p.bounds[filters[0]][1] + 0.1
    with pytest.raises(ValueError, match="within physical bounds"):
        filter_coordinates(p, x)


def test_multifactor_profile_chunks_rhs_without_observation_square(monkeypatch):
    from multimodalsrm import TimeSeries
    from multimodalsrm.bayesian import noise_profile

    from .test_bayesian_problem import api

    p, _, data = fixture(algebra="grouped")
    ts = data["a"]["train"]["brain"]
    data["a"]["train"]["brain"] = TimeSeries(
        np.tile(ts.values, (1, 3)), ts.times, np.tile(ts.mask, (1, 3))
    )
    p.adapter._prepare(data)
    p = api().BayesianProblem(p.adapter, p.priors, anchor=p.anchor, linear_algebra="grouped")
    x = p.initial
    j = p.indices[("noise", "a", "brain")]
    expected = conditional_change(p, x, j, -1e-7)
    monkeypatch.setattr(p, "covariance", lambda *args: pytest.fail("observation covariance"))
    count = len(p.systems["train"].values)

    def checked(original):
        def allocate(shape, *args, **kwargs):
            assert shape != (count, count)
            if isinstance(shape, tuple) and shape[0] == count:
                assert shape[1] <= 32
            return original(shape, *args, **kwargs)

        return allocate

    monkeypatch.setattr(noise_profile.np, "zeros", checked(np.zeros))
    monkeypatch.setattr(noise_profile.np, "empty", checked(np.empty))
    profile = noise_profile.variance_profile(p, x, j)
    assert profile.available, profile.reason
    assert_allclose(profile.change(-1e-7), expected, rtol=1e-9)
    detail = profile.diagnostics["runs"][0]
    assert detail["maximum_rhs_columns"] == 32
    assert detail["rhs_batches"] == 4  # 78 selected observations plus residual
    assert detail["relative_precision_asymmetry"] < 1e-10


def test_grouped_profile_never_broadcasts_observation_factor_squares():
    from multimodalsrm.bayesian.noise_profile import _multifactor_solver

    class BoundedWeights(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            arrays = [np.asarray(a) if isinstance(a, BoundedWeights) else a for a in inputs]
            if ufunc is np.multiply and method == "__call__":
                shape = np.broadcast_shapes(*(np.shape(a) for a in arrays))
                assert len(shape) < 3 or shape[0] <= 32, "unbounded observation factor square"
            return getattr(ufunc, method)(*arrays, **kwargs)

    weights = np.ones((96, 3)).view(BoundedWeights)
    solve, _ = _multifactor_solver(
        np.eye(2), weights, np.ones(96), np.ones(96), np.arange(96) % 2, {}
    )
    assert np.isfinite(solve(np.ones((96, 1)))).all()


def test_filter_refinement_reuses_compiled_physical_density(monkeypatch):
    from multimodalsrm.bayesian.refinement_coordinates import (
        filter_coordinates,
    )

    p, x, _ = fixture(algebra="grouped")
    expected = p.value_gradient(x)
    monkeypatch.setattr(
        p, "objective", lambda *args: pytest.fail("do not compile another full density")
    )
    initial, _, evaluate, _, indices = filter_coordinates(p, x)
    actual = evaluate(initial)
    assert_allclose(actual[0], expected[0], atol=1e-9)
    assert_allclose(actual[1][indices], expected[1][indices], atol=1e-8)
