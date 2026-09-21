"""Independent checks of the real orthonormal rational realization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import solve_sylvester

from multimodalsrm.bayesian._backend import runtime


def _example(width=1.0, frequency=1.3):
    from multimodalsrm.bayesian.state_space_rational import (
        conjugate_pole_block,
        real_pole_block,
    )

    _, jnp, _, _ = runtime()
    rate = 1.2
    latent = jnp.array([1.0, -1.0]) / jnp.sqrt(2.0)
    A = jnp.array([[-rate, 0.0], [-2 * rate, -rate]])
    b = jnp.full(2, jnp.sqrt(2 * rate))
    blocks = (
        (A, b),
        conjugate_pole_block(rate / width, frequency / width),
        real_pole_block(rate / width),
        real_pole_block(rate / width),
    )
    F = jnp.zeros((6, 6)).at[:2, :2].set(A)
    F = F.at[2:4, 2:4].set(jnp.array([[-rate, -frequency], [frequency, -rate]]) / width)
    F = F.at[2, :2].set(latent)
    # A downstream two-stage Gamma bank, with repeated real poles.
    F = F.at[4, 4].set(-rate / width).at[4, 2].set(1 / width)
    F = F.at[5, 5].set(-rate / width).at[5, 4].set(1 / width)
    B = jnp.zeros(6).at[:2].set(b)
    C = jnp.zeros((3, 6)).at[0, :2].set(latent)
    C = C.at[1, 2:4].set(jnp.array([0.7, -0.3]) / jnp.sqrt(width))
    C = C.at[2, 5].set(1.0)
    return F, B, C, blocks


@pytest.mark.parametrize("width,frequency", [(0.4, 1.3), (1.0, 1.3), (2.7, 1.3), (1.0, 0.0)])
def test_realization_matches_scipy_at_coincident_and_complex_poles(width, frequency):
    from multimodalsrm.bayesian.state_space_rational import (
        block_sylvester,
        rational_realization,
    )

    F, B, C, blocks = _example(width, frequency)
    G, b, outputs = map(np.asarray, rational_realization(F, B, C, blocks))
    rhs = -np.outer(B, b)
    expected = solve_sylvester(np.asarray(F), G.T, rhs)
    actual = np.asarray(block_sylvester(F, G, rhs, (2, 2, 1, 1)))
    assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)
    assert_allclose(F @ actual + actual @ G.T, rhs, atol=3e-13)
    assert_allclose(F @ actual, actual @ G, atol=3e-13)
    assert_allclose(actual @ b, B, atol=3e-13)
    assert_allclose(outputs, C @ expected, atol=3e-13)
    assert_allclose(G + G.T + np.outer(b, b), 0.0, atol=3e-14)
    assert np.max(np.linalg.eigvals(G).real) < 0
    assert_allclose(G[:2, :2], F[:2, :2], atol=0)
    assert_allclose(outputs[0], C[0], atol=3e-14)
    for omega in (0.0, 0.1, 1.7, 12.0):
        expected_transfer = C @ np.linalg.solve(1j * omega * np.eye(6) - F, B)
        actual_transfer = outputs @ np.linalg.solve(1j * omega * np.eye(6) - G, b)
        assert_allclose(actual_transfer, expected_transfer, atol=3e-13, rtol=3e-12)
        rate = 1.2
        exact_matern_transfer = 2 * rate**1.5 / (1j * omega + rate) ** 2
        assert_allclose(actual_transfer[0], exact_matern_transfer, atol=3e-13)


def test_general_block_sylvester_independent_rhs():
    from multimodalsrm.bayesian.state_space_rational import (
        block_sylvester,
    )

    rng = np.random.default_rng(13)
    sizes = (1, 2, 1, 2)
    F, G = [np.tril(rng.normal(size=(6, 6))) for _ in range(2)]
    for matrix in (F, G):
        matrix[np.diag_indices(6)] = -np.arange(1.0, 7.0)
        matrix[1, 2], matrix[4, 5] = 0.3, -0.2
    rhs = rng.normal(size=(6, 6))
    actual = block_sylvester(F, G, rhs, sizes)
    assert_allclose(actual, solve_sylvester(F, G.T, rhs), atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("width", [0.4, 1.0, 2.7])
def test_width_jit_and_reverse_derivatives_match_independent_scipy(width):
    from multimodalsrm.bayesian.state_space_rational import (
        rational_realization,
    )

    jax, jnp, _, _ = runtime()

    def output(w):
        F, B, C, blocks = _example(w)
        return rational_realization(F, B, C, blocks)[2]

    def oracle(w):
        F, B, C, blocks = _example(w)
        # Independently construct the stationary cascade for the oracle.
        G = np.zeros((6, 6))
        b = np.concatenate([np.asarray(v) for _, v in blocks])
        start = 0
        for A, drive in blocks:
            end = start + len(drive)
            G[start:end, start:end] = A
            G[start:end, :start] = -np.outer(drive, b[:start])
            start = end
        return np.asarray(C) @ solve_sylvester(np.asarray(F), G.T, -np.outer(B, b))

    value = jax.jit(output)(jnp.asarray(width))
    derivative = jax.jit(jax.jacrev(output))(jnp.asarray(width))
    step = 2e-5
    expected = (oracle(width + step) - oracle(width - step)) / (2 * step)
    assert_allclose(value, oracle(width), atol=3e-13, rtol=3e-12)
    assert_allclose(derivative, expected, atol=2e-9, rtol=3e-7)
    assert_allclose(derivative[0], 0.0, atol=3e-14)


def test_invalid_static_block_topology_rejected():
    from multimodalsrm.bayesian.state_space_rational import (
        block_sylvester,
    )

    for sizes in ((), (3,), (1, 1)):
        with pytest.raises(ValueError, match="block"):
            block_sylvester(-np.eye(3), -np.eye(3), np.eye(3), sizes)


def test_safe_cutoff_bounds_actual_transient_even_with_repeated_poles():
    from scipy.linalg import expm
    from scipy.special import gammaln, logsumexp

    from multimodalsrm.bayesian.state_space_rational import (
        rational_realization,
        safe_decay_cutoff,
    )

    for width, frequency in ((1.0, 0.0), (0.2, 18.0), (4.0, 2.0)):
        F, B, C, blocks = _example(width, frequency)
        G = np.asarray(rational_realization(F, B, C, blocks)[0])
        decay = min(1.2, 1.2 / width)
        norm = np.linalg.norm(G, "fro")
        cutoff = safe_decay_cutoff(6, decay, norm, tolerance=1e-14)
        assert cutoff >= 5 / decay
        for t in (cutoff, 1.2 * cutoff):
            k = np.arange(6)
            bound = -decay * t + logsumexp(k * np.log(2 * norm * t) - gammaln(k + 1))
            assert bound <= np.log(1e-14) + 2e-12
            assert np.linalg.norm(expm(G * t), 2) <= 1e-14


@pytest.mark.parametrize(
    "args",
    [(0, 1, 1), (2, 0, 1), (2, 1, 0), (2, 1, np.inf), (2, 1, 1, 0), (2, 1, 1, 1)],
)
def test_safe_cutoff_rejects_invalid_domain(args):
    from multimodalsrm.bayesian.state_space_rational import (
        safe_decay_cutoff,
    )

    with pytest.raises(ValueError):
        safe_decay_cutoff(*args)
