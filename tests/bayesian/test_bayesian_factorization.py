"""Exact SPD factorization and differentiable CPU execution contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm.bayesian._backend import runtime


@pytest.mark.parametrize("size", [7, 144])
def test_symmetric_solve_inverse_logdet_and_derivatives(size):
    from multimodalsrm.bayesian.factorization import symmetric_solve

    jax, jnp, _, _ = runtime()
    rng = np.random.default_rng(192)
    a = rng.normal(size=(size, size)) / np.sqrt(size)
    matrix = a @ a.T + np.eye(size) * 0.3
    rhs = rng.normal(size=size)
    dm = rng.normal(size=matrix.shape) / size
    dm = (dm + dm.T) / 2
    db = rng.normal(size=size)
    function = symmetric_solve()
    inverse, solution, logdet = jax.jit(function)(matrix, rhs)
    assert_allclose(inverse, np.linalg.inv(matrix), rtol=1e-11, atol=1e-12)
    assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=1e-11, atol=1e-12)
    assert_allclose(
        logdet,
        2 * np.log(np.diag(np.linalg.cholesky(matrix))).sum(),
        rtol=1e-11,
        atol=1e-12,
    )

    # An independent dense solve, not the optimized inverse, is the oracle.
    def oracle(a, b):
        return 0.5 * (b @ jnp.linalg.solve(a, b) + jnp.linalg.slogdet(a)[1])

    def optimized(a, b):
        _, q, logdet = function(a, b)
        return 0.5 * (b @ q + logdet)

    inputs = (jnp.asarray(matrix), jnp.asarray(rhs))
    direction = (jnp.asarray(dm), jnp.asarray(db))
    for f in (optimized, oracle):
        grad = jax.grad(f, argnums=(0, 1))
        result = jax.jit(lambda a, b, da, db: jax.jvp(grad, (a, b), (da, db)))(*inputs, *direction)
        if f is optimized:
            actual = result
        else:
            for aa, bb in zip(jax.tree.leaves(actual), jax.tree.leaves(result)):
                assert_allclose(aa, bb, rtol=1e-9, atol=1e-10)


def test_cpu_symmetric_solve_vmap_and_failure_are_numerical():
    from multimodalsrm.bayesian.factorization import symmetric_solve

    jax, jnp, _, _ = runtime()
    a = jnp.stack([jnp.eye(144) * 2, jnp.eye(144) * 3])
    b = jnp.ones((2, 144))
    inverse, q, logdet = jax.jit(jax.vmap(symmetric_solve()))(a, b)
    assert_allclose(q, np.broadcast_to([[0.5], [1 / 3]], (2, 144)))
    assert_allclose(inverse, np.stack([np.eye(144) / 2, np.eye(144) / 3]))
    assert_allclose(logdet, 144 * np.log([2, 3]))
    # Invalid SPD inputs must return nonfinite results, not throw inside a
    # pure callback; the likelihood can then choose its exact LU fallback.
    bad = a[0].at[0, 0].set(-1)
    answer = jax.jit(symmetric_solve())(bad, b[0])
    assert all(np.isnan(x).all() for x in answer)


@pytest.mark.parametrize(
    "case",
    ["ordinary", "zero_factor", "all_zero", "ill_conditioned", "singular_temporal"],
)
def test_guarded_gaussian_terms_match_observation_space(case):
    from multimodalsrm.bayesian.multifactor_cholesky import gaussian_terms

    jax, _, _, _ = runtime()
    rng = np.random.default_rng(34)
    count, factors = 7, 3
    nodes = np.tile(np.arange(count), 6)
    raw = rng.normal(size=(count, count))
    temporal = raw @ raw.T / count
    weights = rng.normal(size=(len(nodes), factors))
    if case == "singular_temporal":
        temporal = np.ones_like(temporal)
    elif case == "all_zero":
        weights[:] = 0
    elif case == "zero_factor":
        weights[:, -1] = 0
    elif case == "ill_conditioned":
        weights[:, -1] = weights[:, 0] + 1e-7 * weights[:, -1]
    variance = rng.uniform(0.1, 0.5, len(nodes))
    residual = rng.normal(size=len(nodes))
    design = np.zeros((len(nodes), count * factors))
    for i, node in enumerate(nodes):
        design[i, node * factors : (node + 1) * factors] = weights[i]
    prior = np.kron(temporal, np.eye(factors))
    covariance = design @ prior @ design.T + np.diag(variance)
    precision = np.linalg.inv(covariance)
    posterior = prior - prior @ design.T @ precision @ design @ prior
    reduction = design.T @ precision @ design
    expected_trace = np.trace(reduction.reshape(count, factors, count, factors), axis1=1, axis2=3)
    diagonal = np.stack(
        [
            posterior[i * factors : (i + 1) * factors, i * factors : (i + 1) * factors]
            for i in range(count)
        ]
    )
    value, q, actual_diagonal, trace = jax.jit(gaussian_terms)(
        temporal, weights, residual, variance, nodes
    )
    assert_allclose(
        value,
        0.5
        * (
            residual @ precision @ residual
            + 2 * np.log(np.diag(np.linalg.cholesky(covariance))).sum()
            + len(nodes) * np.log(2 * np.pi)
        ),
        atol=1e-10,
    )
    assert_allclose(q, (design.T @ precision @ residual).reshape(count, factors), atol=1e-10)
    assert_allclose(actual_diagonal, diagonal, atol=1e-10)
    assert_allclose(trace, expected_trace, atol=1e-10)


@pytest.mark.parametrize("failed_symmetric", [False, True])
def test_large_likelihood_and_both_curvature_orders_match_dense(failed_symmetric, monkeypatch):
    from multimodalsrm.bayesian import factorization
    from multimodalsrm.bayesian.multifactor_score import gaussian_score

    jax, jnp, _, _ = runtime()
    count, factors = 48, 3  # 144 latent dimensions exercises the CPU callback.
    rng = np.random.default_rng(752)
    nodes = np.tile(np.arange(count), 5)
    times = np.arange(count) / 10
    temporal = np.exp(-np.abs(times[:, None] - times[None, :]))
    weights = rng.normal(size=(len(nodes), factors))
    residual = rng.normal(size=len(nodes))
    variance = rng.uniform(0.2, 0.8, len(nodes))
    # New custom-JVP objects prevent a previously traced original callback
    # from bypassing the forced numerical failure in this test.
    factorization.symmetric_solve.cache_clear()
    gaussian_score.cache_clear()
    calls = []
    if failed_symmetric:

        def numerical_failure(matrix, rhs):
            calls.append(True)
            return (
                np.full_like(matrix, np.nan),
                np.full_like(rhs, np.nan),
                np.asarray(np.nan, dtype=matrix.dtype),
            )

        monkeypatch.setattr(factorization, "_cpu_symmetric_solve", numerical_failure)

    def actual(z):
        return gaussian_score()(
            jnp.asarray(temporal),
            jnp.asarray(weights) * jnp.exp(z),
            jnp.asarray(residual),
            jnp.asarray(variance),
            jnp.asarray(nodes),
        )

    def oracle(z):
        w = jnp.asarray(weights) * jnp.exp(z)
        covariance = temporal[np.ix_(nodes, nodes)] * (w @ w.T) + jnp.diag(variance)
        chol = jnp.linalg.cholesky(covariance)
        whitened = jax.scipy.linalg.solve_triangular(chol, residual, lower=True)
        return (
            0.5 * (whitened @ whitened + len(nodes) * jnp.log(2 * jnp.pi))
            + jnp.log(jnp.diag(chol)).sum()
        )

    point = jnp.asarray(0.13)
    for f in (actual, oracle):
        value, gradient = jax.jit(jax.value_and_grad(f))(point)
        forward = jax.jit(lambda z: jax.jvp(jax.grad(f), (z,), (jnp.ones_like(z),))[1])(point)
        reverse = jax.jit(jax.grad(jax.grad(f)))(point)
        result = np.asarray([f(point), value, gradient, forward, reverse])
        if f is actual:
            answer = result
        else:
            assert_allclose(answer, result, rtol=1e-9, atol=1e-8)
    if failed_symmetric:
        assert calls, "The failed-SPD fallback was not exercised"
    factorization.symmetric_solve.cache_clear()
    gaussian_score.cache_clear()


def test_blas_oversubscription_thresholds_are_half_the_physical_cores():
    from multimodalsrm.bayesian.factorization import blas_oversubscription

    assert blas_oversubscription(blas_threads=1, physical_cores=64) is None
    assert blas_oversubscription(blas_threads=32, physical_cores=64) is None
    message = blas_oversubscription(blas_threads=33, physical_cores=64)
    assert "33 BLAS threads on 64 physical cores" in message
    assert "OPENBLAS_NUM_THREADS" in message
    # A single core still tolerates one thread; two threads oversubscribe it.
    assert blas_oversubscription(blas_threads=1, physical_cores=1) is None
    assert blas_oversubscription(blas_threads=2, physical_cores=1) is not None


def test_cpu_callback_warns_once_about_blas_oversubscription(monkeypatch):
    import warnings

    from multimodalsrm.bayesian import factorization

    monkeypatch.setattr(factorization, "_blas_threads", lambda: 64)
    monkeypatch.setattr(factorization, "_physical_cores", lambda: 64)
    monkeypatch.setattr(factorization, "_thread_warning_issued", False)
    matrix, rhs = np.eye(3) * 2.0, np.ones(3)
    with pytest.warns(RuntimeWarning, match="64 BLAS threads on 64 physical cores"):
        inverse, solution, logdet = factorization._cpu_symmetric_solve(matrix, rhs)
    assert_allclose(inverse, np.eye(3) / 2.0)
    assert_allclose(solution, np.full(3, 0.5))
    assert_allclose(logdet, 3 * np.log(2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factorization._cpu_symmetric_solve(matrix, rhs)
    # Within the limit, a fresh process would not warn at all.
    monkeypatch.setattr(factorization, "_blas_threads", lambda: 16)
    monkeypatch.setattr(factorization, "_thread_warning_issued", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factorization._cpu_symmetric_solve(matrix, rhs)
