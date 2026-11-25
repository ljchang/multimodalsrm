"""Grouped solves must stay exact where LU of I + K D loses all accuracy.

With about 400 shared nodes on a regular grid, a wide Gaussian response and a
node precision near 10, partial-pivoting LU of the nonsymmetric I + K D has
element growth above 1e15: the grouped density disagreed with the unreduced
covariance by factors of 1e4 to 1e11 although the matrix condition number is
about 1e3. Every grouped consumer of that factor is checked against dense.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Gaussian, Identity, Response, TimeSeries

from .test_bayesian_problem import api

SAMPLES = 200
VARIANTS = {
    "single": dict(),
    "two_factors": dict(features=2),
    "correlated_noise": dict(noise_timescales={"rating": 4.0}),
    "run_baselines": dict(run_baseline_sd={"rating": 1.0, "au": 0.5}),
}


def problems(variant, width):
    from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

    b = api()
    settings = dict(VARIANTS[variant])
    features = settings.pop("features", 1)
    rng = np.random.default_rng(0)
    times = np.arange(SAMPLES, dtype=float)
    data = {
        subject: {
            "run": {
                modality: TimeSeries(rng.normal(size=(SAMPLES, features)), times)
                for modality in ("au", "rating")
            }
        }
        for subject in ("a", "b")
    }
    responses = {
        "au": Response(Identity(), estimate=False),
        "rating": Response(Gaussian(2.0, 0.0), bounds={"lag": (-15.0, 15.0), "width": (0.5, 10.0)}),
    }
    priors = b.BayesianPriors(
        noise=b.Prior.lognormal(np.log(0.7), 1.0),
        loading_sd=1.0,
        offset_sd=1.0,
        filters={
            "rating": {
                "lag": b.Prior.normal(0.0, 5.0),
                "width": b.Prior.lognormal(np.log(2.0), 0.7),
            }
        },
    )
    built = []
    for algebra in ("dense", "grouped"):
        adapter = BayesianObservationAdapter(
            features=features,
            latent_dt=1.0,
            responses=responses,
            length_scale=3.0,
            standardize=False,
            max_observations=10_000,
        )
        adapter.edge_observations = "include"
        adapter._prepare(data)
        built.append(
            b.BayesianProblem(
                adapter,
                priors,
                anchor=("a", "rating", 0),
                linear_algebra=algebra,
                response_quadrature_order=16,
                **settings,
            )
        )
    dense, grouped = built
    x = np.asarray(dense.initial, float).copy()
    for i, name in enumerate(dense.names):
        if name[0] == "loading":
            x[i] = 2.2 if features == 1 else 2.2 * rng.normal()
        elif name[0] == "noise":
            x[i] = 1.0
    x[dense.names.index(("filter", "rating", "width"))] = width
    return dense, grouped, x


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("width", [6.0, 8.0])
def test_grouped_density_matches_dense_for_smooth_covariance(variant, width):
    dense, grouped, x = problems(variant, width)
    assert len(next(iter(grouped.grouped_systems.values())).times) == 2 * SAMPLES
    assert_allclose(float(grouped.nll(x)), float(dense.nll(x)), rtol=1e-11)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_grouped_gradient_matches_dense_for_smooth_covariance(variant):
    dense, grouped, x = problems(variant, 8.0)
    value, gradient = grouped.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert_allclose(value, expected, rtol=1e-11)
    assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("key,include_noise", [(None, False), (("b", "rating", 0), True)])
def test_grouped_prediction_matches_dense_for_smooth_covariance(variant, key, include_noise):
    from multimodalsrm.bayesian.prediction import project

    dense, grouped, x = problems(variant, 8.0)
    query = np.array([20.3, 71.0, 150.5])
    expected = project(dense, x[None], "run", query, key=key, include_noise=include_noise)
    actual = project(grouped, x[None], "run", query, key=key, include_noise=include_noise)
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("variant", ["single", "two_factors", "correlated_noise"])
def test_grouped_joint_latent_moments_match_dense_for_smooth_covariance(variant):
    from multimodalsrm.bayesian.trajectories import joint_moments

    dense, grouped, x = problems(variant, 8.0)
    query = np.array([20.3, 71.0, 150.5])
    rotation = np.eye(dense.features)
    expected = joint_moments(dense, "run", query)(x, rotation)
    actual = joint_moments(grouped, "run", query)(x, rotation)
    for a, e in zip(actual, expected):
        assert_allclose(a, e, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("variant", ["single", "two_factors", "run_baselines"])
def test_grouped_variance_profile_matches_dense_for_smooth_covariance(variant):
    from multimodalsrm.bayesian.noise_profile import variance_profile

    dense, grouped, x = problems(variant, 8.0)
    coordinate = dense.names.index(("noise", "b", "rating"))
    expected = variance_profile(dense, x, coordinate)
    actual = variance_profile(grouped, x, coordinate)
    assert expected.available and actual.available, (expected.reason, actual.reason)
    assert_allclose(actual.gradient, expected.gradient, rtol=1e-8)
    assert_allclose(actual.curvature, expected.curvature, rtol=1e-8)


@pytest.mark.parametrize("factors,degenerate", [(1, True), (3, True)])
def test_precision_factor_derivatives_match_unfactored_algebra(factors, degenerate):
    """The Cholesky cache carries no derivative; the identities in K and D do.

    Zero-loading nodes and rank-one blocks (a repeated zero eigenvalue for
    three factors) are where a differentiated square root would fail.
    """
    api()
    import jax
    import jax.numpy as jnp

    from multimodalsrm.bayesian import precision_factor as pf

    rng = np.random.default_rng(3)
    nodes, observations = 9, 30
    times = np.sort(rng.uniform(0, 10, nodes))
    kernel = jnp.asarray(np.exp(-0.5 * (times[:, None] - times) ** 2 / 4.0))
    index = rng.integers(0, nodes, observations)
    loadings = rng.normal(size=(observations, factors))
    if degenerate:
        loadings[index == index[0]] = 0.0
        loadings[index == index[1], 1:] = 0.0
    residual = jnp.asarray(rng.normal(size=observations))
    size = nodes * factors
    rhs = jnp.asarray(rng.normal(size=(size, 2)))
    probe = jnp.asarray(rng.normal(size=size))
    x = jnp.concatenate(
        [
            jnp.asarray(loadings).ravel(),
            jnp.asarray(rng.uniform(0.5, 2, observations)),
            jnp.asarray([1.3]),
        ]
    )

    def operands(x):
        w = x[: observations * factors].reshape(observations, factors)
        variance = x[observations * factors : -1]
        blocks = jax.ops.segment_sum(
            w[:, :, None] * w[:, None, :] / variance[:, None, None], index, num_segments=nodes
        )
        score = jax.ops.segment_sum(w * (residual / variance)[:, None], index, num_segments=nodes)
        return kernel ** x[-1], blocks, score.reshape(-1)

    def stable(x):
        temporal, blocks, score = operands(x)
        factor = pf.factorize(temporal, blocks)
        return (
            pf.logdet(factor)
            + pf.score_solve(factor, score) @ probe
            + jnp.sum(pf.precision_solve(factor, rhs * x[-1]) ** 2)
        )

    def unfactored(x):
        temporal, blocks, score = operands(x)
        D = jnp.einsum("ij,iab->iajb", jnp.eye(nodes), blocks).reshape(size, size)
        B = jnp.eye(size) + jnp.kron(temporal, jnp.eye(factors)) @ D
        return (
            jnp.linalg.slogdet(B)[1]
            + jnp.linalg.solve(B.T, score) @ probe
            + jnp.sum((D @ jnp.linalg.solve(B, rhs * x[-1])) ** 2)
        )

    assert_allclose(stable(x), unfactored(x), rtol=1e-12)
    assert_allclose(jax.grad(stable)(x), jax.grad(unfactored)(x), rtol=1e-9, atol=1e-10)
    assert_allclose(jax.jacfwd(stable)(x), jax.grad(unfactored)(x), rtol=1e-9, atol=1e-10)
    # Second order, forward over reverse and reverse over reverse: a
    # derivative-free cache is only correct if both see the identities.
    direction = jnp.cos(jnp.arange(len(x)))
    expected = jax.jvp(jax.grad(unfactored), (x,), (direction,))[1]
    assert_allclose(
        jax.jvp(jax.grad(stable), (x,), (direction,))[1], expected, rtol=1e-8, atol=1e-9
    )
    assert_allclose(
        jax.grad(lambda x: jax.grad(stable)(x) @ direction)(x), expected, rtol=1e-8, atol=1e-9
    )
