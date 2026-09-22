"""Independent integration detects wrong filter scales, signs, delays or normalization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from multimodalsrm import DoubleGamma, Gamma, Identity, Response


def realization(kernels, **kwargs):
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    return ResponseStateSpace.prepare(
        {m: Response(k, pooling="shared", estimate=False) for m, k in kernels.items()},
        length_scale=3.0,
        tolerance=1e-6,
        **kwargs,
    )


def matern(t):
    a = np.sqrt(3) * np.abs(t) / 3.0
    return (1 + a) * np.exp(-a)


@pytest.mark.parametrize(
    "kernel",
    [
        Gamma(1, 0.7, -0.3),
        Gamma(3, 0.7, 1.25),
        DoubleGamma(),
        DoubleGamma(3, 0.7, 7, 1.1, 0.3, -1.5),
    ],
)
def test_response_latent_covariance_matches_independent_finite_integral(kernel):
    state = realization({"response": kernel, "identity": Identity()})
    times = np.array([-5.0, 0.0, 2.0, 9.0, 100.0])
    actual = state.covariance(times, np.zeros(len(times), int), [0.0], [-1])[:, 0]
    expected = [
        quad(
            lambda u: float(kernel.evaluate(u)) * matern(t - u),
            *kernel.support,
            epsabs=2e-10,
            points=[t] if kernel.support[0] < t < kernel.support[1] else None,
        )[0]
        for t in times
    ]
    assert_allclose(actual, expected, atol=state.tail_bound + 2e-10, rtol=0)
    reverse = state.covariance([0.0], [-1], times, np.zeros(len(times), int))
    assert_allclose(actual, reverse[0], atol=1e-12)


def test_signed_response_variance_and_distinct_scale_cross_covariance():
    a, b = Gamma(3, 0.7, 0.2), DoubleGamma(3, 0.7, 7, 1.1, 0.3, -0.4)
    state = realization({"a": a, "b": b})
    expected = quad(
        lambda u: (
            float(a.evaluate(u))
            * quad(
                lambda v: float(b.evaluate(v)) * matern(1.5 - u + v),
                *b.support,
                epsabs=2e-9,
                points=[u - 1.5] if b.support[0] < u - 1.5 < b.support[1] else None,
            )[0]
        ),
        *a.support,
        epsabs=2e-9,
    )[0]
    actual = state.covariance([1.5], [0], [0.0], [1])[0, 0]
    assert_allclose(actual, expected, atol=state.tail_bound + 1e-8, rtol=0)
    matrix = state.covariance([0, 0, 0], [0, 1, -1], [0, 0, 0], [0, 1, -1])
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert matrix[1, 1] != pytest.approx(1.0)


def test_stationary_transitions_compose_and_handle_tiny_and_huge_intervals():
    state = realization({"brain": DoubleGamma(), "rating": Gamma(3, 0.7, -0.4)})
    A, Q = state.transitions(np.array([0.0, 1e-10, 0.3, 0.7, 1.0, 1e8]), 1)
    assert_allclose(A[0], np.eye(state.dimension), atol=0)
    assert_allclose(Q[0], 0, atol=0)
    assert_allclose(A[4], A[3] @ A[2], atol=2e-12)
    assert_allclose(Q[4], Q[3] + A[3] @ Q[2] @ A[3].T, atol=2e-12)
    assert_allclose(
        A @ A.transpose(0, 2, 1) + Q,
        np.broadcast_to(np.eye(state.dimension), Q.shape),
        atol=3e-10,
    )
    assert np.linalg.eigvalsh(Q).min() > -1e-13
    assert_allclose(A[-1], 0, atol=0)


@pytest.mark.parametrize(
    "kernel, reason",
    [
        (Gamma(2.5), "state_space"),
        (Gamma(100), "state_space"),
    ],
)
def test_unsupported_shapes_fail_explicitly(kernel, reason):
    with pytest.raises(ValueError, match=reason):
        realization({"response": kernel})


def test_tail_bound_is_enforced_and_free_response_is_rejected():
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    with pytest.raises(ValueError, match="tail.*covariance_tolerance"):
        ResponseStateSpace.prepare(
            {"a": Response(Gamma(3), estimate=False, pooling="shared")}, 3.0, 1e-12
        )
    with pytest.raises(ValueError, match="fixed"):
        ResponseStateSpace.prepare({"a": Response(Gamma(3), pooling="shared")}, 3.0, 1e-6)


@pytest.mark.parametrize("scales", [(0.7, 0.7000000001), (0.02, 20.0)])
def test_rational_response_matches_independent_transfer_function(scales):
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    kernels = [Gamma(3, s, 0.4) for s in scales]
    state = ResponseStateSpace.prepare(
        {str(i): Response(k, pooling="shared", estimate=False) for i, k in enumerate(kernels)},
        3.0,
        1e-4,
    )
    rate = np.sqrt(3) / 3
    for frequency in (0.0, 0.01, 0.2, 1.0, 10.0, 100.0):
        s = 1j * frequency
        latent = np.sqrt(4 * rate**3) / (s + rate) ** 2
        transfer = state.outputs @ np.linalg.solve(
            s * np.eye(state.dimension) - state.generator, state.driving
        )
        expected = [latent / (1 + s * k.scale) ** k.shape / k._energy for k in kernels] + [latent]
        assert_allclose(transfer, expected, atol=2e-10, rtol=2e-9)
