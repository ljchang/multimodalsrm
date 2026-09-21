"""Long-tail SCR integration must resolve noise-sensitive likelihood scores."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from multimodalsrm import BachSCR, Response

from .test_bayesian_problem import api


def test_high_order_scr_noise_score_matches_independent_integration():
    api()
    import jax
    import jax.numpy as jnp

    from multimodalsrm.bayesian.response_quadrature import (
        ResponseQuadrature,
    )

    kernel = BachSCR(lambda2=0.06)
    response = Response(kernel, estimate=False, pooling="shared")
    times = np.array([0.0, 0.1])
    residual = np.array([0.2, -0.2])
    noise = 0.002

    def reference_entry(delta):
        def inner(u):
            def integrand(v):
                z = np.sqrt(3.0) / 3.0 * abs(delta - u + v)
                return float(kernel.evaluate(v)) * (1 + z) * np.exp(-z)

            points = [u - delta] if 0 < u - delta < 90 else []
            return (
                float(kernel.evaluate(u))
                * quad(
                    integrand,
                    0,
                    90,
                    points=points,
                    epsabs=1e-11,
                    epsrel=1e-11,
                    limit=300,
                )[0]
            )

        return quad(inner, 0, 90, epsabs=1e-11, epsrel=1e-11, limit=300)[0]

    k0 = reference_entry(0.0)
    k1 = reference_entry(0.1)
    C = np.array([[k0, k1], [k1, k0]]) + noise * np.eye(2)
    alpha = np.linalg.solve(C, residual)
    expected = 0.5 * (np.trace(np.linalg.inv(C)) - alpha @ alpha)
    q = ResponseQuadrature({"eda": response}, {}, 3.0, 1024)
    covariance = q.covariance(jnp.zeros(0), times, np.zeros(2, int), times, np.zeros(2, int))

    def nll(variance):
        matrix = covariance + variance * jnp.eye(2)
        return 0.5 * (jnp.linalg.slogdet(matrix)[1] + residual @ jnp.linalg.solve(matrix, residual))

    actual = float(jax.grad(nll)(noise))
    assert_allclose(actual, expected, atol=1e-3, rtol=0)
