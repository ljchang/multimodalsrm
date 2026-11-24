"""Covariance symmetry reuse against explicit finite weighted Matérn Grams."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import BatemanSCR, DoubleGamma, Gamma, Gaussian, Identity, Response
from multimodalsrm.bayesian.response_quadrature import ResponseQuadrature

from .test_bayesian_problem import api


def fixture():
    api()
    import jax.numpy as jnp

    kernels = dict(
        identity=Identity(),
        face=Gaussian(1.1, 0.7),
        rating=Gamma(2.7, 0.8, 1.2),
        brain=DoubleGamma(lag=0.4),
        eda=BatemanSCR(0.7, 3.0, 0.2),
    )
    responses, indices, values = {}, {}, []
    for modality, kernel in kernels.items():
        bounds = {}
        for name, value in kernel.parameters.items():
            indices["filter", modality, name] = len(values)
            values.append(value)
            bounds[name] = (
                (value - 0.5, value + 0.5) if name == "lag" else (0.8 * value, 1.2 * value)
            )
        if modality == "eda":
            bounds.update(rise=(0.3, 4.0), decay=(0.3, 4.0))
        responses[modality] = Response(
            kernel, bounds=bounds, estimate=bool(bounds), pooling="shared"
        )
    q = ResponseQuadrature(responses, indices, 3.0, 8, minimum_length_scale=1.0)
    return q, jnp.array([*values, 3.0])


def direct(q, x, ta, ma, tb, mb):
    """Independent O(Q**2) summation; no prefix moments or symmetry reuse."""
    import jax.numpy as jnp

    nodes = {int(m): q.nodes(x[:-1], int(m)) for m in np.union1d(ma, mb)}
    rows = []
    for t, m in zip(ta, ma):
        u, wa = nodes[int(m)]
        row = []
        for s, n in zip(tb, mb):
            v, wb = nodes[int(n)]
            distance = jnp.sqrt(3.0) * jnp.abs(t - s - u[:, None] + v[None, :]) / x[-1]
            row.append(wa @ ((1 + distance) * jnp.exp(-distance)) @ wb)
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


@pytest.mark.parametrize("length", [1.0, 3.0, 15.0])
def test_all_families_learned_shapes_and_timescale_match_direct_gram(length):
    q, x = fixture()
    import jax
    import jax.numpy as jnp

    x = x.at[-1].set(length)
    # Equal Bateman constants, signed DoubleGamma, latent/Identity entries,
    # repeated/unsorted times, and distinct modalities on the same clock.
    x = x.at[q.indices["filter", "eda", "rise"]].set(3.0)
    times = np.array([8.3, 2.0, 4.7, 2.0, 3.8, 6.1, 4.7, 6.1])
    modalities = np.array([3, -1, 1, 2, 0, 4, 1, 4])
    weights = jnp.asarray(np.random.default_rng(831).normal(size=(len(times), len(times))))

    def actual(point):
        return q.covariance(
            point[:-1], times, modalities, times.copy(), modalities.copy(), length_scale=point[-1]
        )

    def expected(point):
        return direct(q, point, times, modalities, times, modalities)

    matrix, reference = np.asarray(actual(x)), np.asarray(expected(x))
    assert np.array_equal(matrix, matrix.T)
    assert_allclose(matrix, reference, atol=2e-11, rtol=2e-12)
    assert np.linalg.eigvalsh(matrix).min() > -1e-10
    # Nonsymmetric cotangents expose dropped/doubled transpose contributions.
    gradients = [
        jax.jit(jax.grad(lambda p: jnp.sum(fn(p) * weights)))(x) for fn in (actual, expected)
    ]
    assert_allclose(*gradients, atol=2e-9, rtol=2e-10)


def test_default_gaussian_width_on_a_two_second_clock_has_the_direct_gradient():
    """Unit width and a 2 s clock put response nodes exactly on quadrature endpoints.

    The Gaussian panels start at -6 and -4 widths, so a 2 s time difference
    aligns a node of one panel with the first node of the rule. Two responses
    starting from identical parameters also share both endpoints at zero lag.
    """
    api()
    import jax
    import jax.numpy as jnp

    responses = {
        name: Response(
            Gaussian(1.0, 0.0), bounds=dict(width=(0.4, 3.0), lag=(-2.0, 2.0)), pooling="shared"
        )
        for name in ("face", "rating")
    }
    indices = {
        ("filter", name, parameter): 2 * i + j
        for i, name in enumerate(responses)
        for j, parameter in enumerate(("width", "lag"))
    }
    q = ResponseQuadrature(responses, indices, 3.0, 8)
    x = jnp.array([1.0, 0.0, 1.0, 0.0, 3.0])
    times = np.tile(2.0 * np.arange(6), 2)
    modalities = np.repeat([0, 1], 6)
    weights = jnp.asarray(np.random.default_rng(47).normal(size=(12, 12)))

    def actual(point):
        return q.covariance(
            point[:-1], times, modalities, times.copy(), modalities.copy(), length_scale=point[-1]
        )

    def expected(point):
        return direct(q, point, times, modalities, times, modalities)

    assert_allclose(actual(x), expected(x), atol=2e-11, rtol=2e-12)
    gradients = [jax.grad(lambda p: jnp.sum(fn(p) * weights))(x) for fn in (actual, expected)]
    assert_allclose(*gradients, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize(
    "query_kind", ["rectangular", "different_times", "different_modalities", "dynamic"]
)
def test_cross_queries_and_time_derivatives_keep_full_covariance(query_kind):
    q, x = fixture()
    import jax
    import jax.numpy as jnp

    ta = np.array([3.1, 8.4, 5.3])
    ma = np.array([1, 3, 4])
    tb, mb = ta.copy(), ma.copy()
    if query_kind == "rectangular":
        tb, mb = np.array([7.2, 1.4]), np.array([2, -1])
    elif query_kind == "different_times":
        tb += np.array([0.2, -0.1, 0.4])
    elif query_kind == "different_modalities":
        mb = np.array([3, 4, 1])
    else:
        ta, tb = jnp.asarray(ta), jnp.asarray(tb)

    def actual(times):
        return q.covariance(x[:-1], times, ma, tb, mb, length_scale=x[-1])

    reference = direct(q, x, ta, ma, tb, mb)
    assert_allclose(actual(ta), reference, atol=2e-11, rtol=2e-12)
    if query_kind == "dynamic":
        # Differentiating one query clock must not reflect its entries into the
        # separately supplied training clock, even if the values happen to match.
        derivatives = [
            jax.jacfwd(fn)(ta) for fn in (actual, lambda times: direct(q, x, times, ma, tb, mb))
        ]
        assert_allclose(*derivatives, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("left,right", [(0, 0), (0, 2), (2, 0)])
def test_empty_covariance_shapes(left, right):
    q, x = fixture()
    result = q.covariance(
        x[:-1], np.zeros(left), np.zeros(left, int), np.zeros(right), np.zeros(right, int)
    )
    assert result.shape == (left, right)
