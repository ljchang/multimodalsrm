"""Direct feature mappings preserve scalar prior math at large dimension."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import qmc

from multimodalsrm import Identity, Response, TimeSeries
from multimodalsrm.bayesian.fitting import initial_points
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

from .test_bayesian_problem import api, problem_fixture


def wide_problem(width):
    b = api()
    times = np.arange(4.0)
    values = np.sin(times[:, None] + np.arange(width)[None, :])
    adapter = BayesianObservationAdapter(
        features=3,
        latent_dt=1.0,
        responses={"brain": Response(Identity(), pooling="shared", estimate=False)},
        standardize=False,
        max_observations=4 * width,
    )
    adapter._prepare({"a": {"train": {"brain": TimeSeries(values, times)}}})
    return b.BayesianProblem(
        adapter,
        b.BayesianPriors(noise=b.Prior.lognormal(-2.0, 0.7)),
        anchor=("a", "brain", 0),
        linear_algebra="grouped",
    )


def scalar_initial_points(problem, count, seed):
    """Frozen pre-repair calculation, including observation order and Sobol."""
    u = qmc.Sobol(len(problem.names), scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(count)))
    )[:count]
    points = np.column_stack(
        [p.ppf(np.clip(u[:, j], 1e-5, 1 - 1e-5)) for j, p in enumerate(problem.parameter_priors)]
    )
    base = problem.initial.copy()
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(len(problem.keys), problem.features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for i, key in enumerate(problem.keys):
        values = np.concatenate(
            [s.values[np.array([k == key for k in s.keys])] for s in problem.systems.values()]
        )
        scale = np.sqrt(max(0.01, np.var(values)))
        begin = i * problem.features
        base[begin : begin + problem.features] = (
            scale if problem.features == 1 else scale * directions[i]
        )
        base[problem.indices[("offset", *key)]] = np.mean(values)
    for i, prior in enumerate(problem.parameter_priors):
        base[i] = np.clip(base[i], prior.ppf(1e-4), prior.ppf(1 - 1e-4))
    points[0] = base
    return points


@pytest.mark.parametrize("factors", [1, 3])
def test_initialization_preserves_scalar_sobol_and_masked_multi_run_statistics(factors):
    from .test_bayesian_multifactor import fixture

    problem, _, data = fixture(features=factors)
    for runs in data.values():
        runs["second"] = {
            m: TimeSeries(ts.values * 0.3 + 1e5, ts.times + 100, ts.mask)
            for m, ts in runs["train"].items()
        }
    adapter = problem.adapter
    adapter._prepare(data)
    problem = api().BayesianProblem(
        adapter, problem.priors, anchor=problem.anchor, linear_algebra="grouped"
    )
    assert_allclose(
        initial_points(problem, 4, 871),
        scalar_initial_points(problem, 4, 871),
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.parametrize("noise_family", ["normal", "lognormal", "uniform"])
def test_batched_prior_density_transforms_and_gradients_equal_scalar_math(noise_family):
    import jax
    import jax.numpy as jnp

    _, adapter, _ = problem_fixture(True)
    b = api()
    noise = {
        "normal": b.Prior.normal(0.2, 0.1).bounded(0.02, np.inf),
        "lognormal": b.Prior.lognormal(-2.0, 0.7).bounded(0.05, 0.7),
        "uniform": b.Prior.uniform(0.05, 0.7),
    }[noise_family]
    problem = b.BayesianProblem(
        adapter,
        b.BayesianPriors(
            noise=noise,
            filters={
                "signal": {
                    "width": b.Prior.uniform(0.1, 0.9),
                    "lag": b.Prior.normal(0.5, 0.4),
                }
            },
        ),
        anchor=("a", "ref", 0),
    )
    z = jnp.linspace(-1.3, 1.7, len(problem.names))

    def scalar_from(z):
        x = jnp.stack([t(z[i]) for i, t in enumerate(problem.transforms)])
        jac = sum(t.log_abs_det_jacobian(z[i], x[i]) for i, t in enumerate(problem.transforms))
        return x, jac

    def scalar_density(x):
        return sum(d.log_prob(x[i]) for i, d in enumerate(problem.distributions))

    expected, expected_jac = scalar_from(z)
    actual, actual_jac = problem.from_unconstrained(z)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert_allclose(actual_jac, expected_jac, atol=1e-13)
    assert_allclose(problem.to_unconstrained(actual), z, atol=1e-13)
    assert_allclose(problem.log_prior(actual), scalar_density(expected), atol=1e-13)

    def scalar_potential(z):
        x, jac = scalar_from(z)
        return -scalar_density(x) - jac

    def batched_potential(z):
        x, jac = problem.from_unconstrained(z)
        return -problem.log_prior(x) - jac

    gradient = jax.grad(batched_potential)(z)
    assert np.isfinite(gradient).all()
    assert_allclose(gradient, jax.grad(scalar_potential)(z), atol=1e-12)
    for invalid in (0.0, -0.1, np.inf, np.nan):
        bad = np.asarray(actual).copy()
        bad[problem.indices[("noise", "a", "ref")]] = invalid
        assert np.isneginf(float(problem.log_prior(bad)))
        with pytest.raises(ValueError, match="support|finite"):
            problem.to_unconstrained(bad)


def test_parameter_graph_size_does_not_grow_with_repeated_feature_priors():
    import jax

    small, larger = wide_problem(2), wide_problem(20)
    for method in ("log_prior", "from_unconstrained"):
        small_graph = jax.make_jaxpr(getattr(small, method))(small.initial)
        larger_graph = jax.make_jaxpr(getattr(larger, method))(larger.initial)
        assert len(larger_graph.jaxpr.eqns) < 3 * len(small_graph.jaxpr.eqns)


@pytest.mark.parametrize("as_list", [False, True])
def test_integer_unconstrained_inputs_do_not_truncate_physical_parameters(as_list):
    import jax.numpy as jnp

    problem, _, _ = problem_fixture()
    z = np.ones(len(problem.names), dtype=int)
    if as_list:
        z = z.tolist()
    expected = jnp.stack([t(z[i]) for i, t in enumerate(problem.transforms)])
    actual, _ = problem.from_unconstrained(z)
    assert_allclose(actual, expected, atol=1e-14)


def test_initialization_above_sobol_limit_is_finite_reproducible_and_supported():
    b = api()
    width, factors = 18139, 3
    keys = [("a", "brain", i) for i in range(width)]
    names = [("loading", *k, f) for k in keys for f in range(factors)]
    names += [("offset", *k) for k in keys] + [("noise", "a", "brain")]
    normal, noise = b.Prior.normal(), b.Prior.lognormal(-2.0, 0.7)
    priors = [normal] * (len(names) - 1) + [noise]
    problem = SimpleNamespace(
        names=names,
        parameter_priors=priors,
        features=factors,
        keys=keys,
        indices={name: i for i, name in enumerate(names)},
        initial=np.r_[np.zeros(len(names) - 1), np.exp(-2.0)],
        systems={
            "train": SimpleNamespace(
                keys=keys + keys,
                values=np.r_[np.ones(width), -np.ones(width)],
            )
        },
    )
    assert len(names) > qmc.Sobol.MAXDIM
    points = initial_points(problem, 3, 2026)
    assert points.shape == (3, len(names))
    assert np.isfinite(points).all()
    assert (points[:, -1] > 0).all()
    assert_array_equal(points, initial_points(problem, 3, 2026))
    assert not np.array_equal(points[1:], initial_points(problem, 3, 2027)[1:])
    assert_allclose(np.linalg.norm(points[0, : factors * width].reshape(width, factors), axis=1), 1)
    assert_array_equal(points[0, factors * width : -1], np.zeros(width))
