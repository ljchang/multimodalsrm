"""The analytic derivative must agree with independent dense likelihoods."""

import numpy as np
import pytest

from .test_bayesian_grouped import grouped_fixture


@pytest.mark.parametrize("case", ["ordinary", "zero", "high_signal"])
def test_analytic_score_matches_dense_gradient_and_finite_difference(case):
    dense, grouped, x = grouped_fixture()
    from multimodalsrm.bayesian._backend import runtime, temporal
    from multimodalsrm.bayesian.grouped_score import gaussian_score

    jax, jnp, _, _ = runtime()
    for i, name in enumerate(grouped.names):
        if case == "zero" and name[0] == "loading" and name[2] == "signal":
            x[i] = 0.0
        elif case == "high_signal" and name[0] == "loading":
            x[i] *= 10
        elif case == "high_signal" and name[0] == "noise":
            x[i] *= 0.001

    def objective(x):
        weights, offsets, noise, widths, lags = grouped.arrays(x)
        result = -grouped.log_prior(x)
        for run, nodes in grouped.grouped_systems.items():
            ki, gi, _ = grouped._packed[run]
            mi = nodes.modalities
            K = temporal(
                nodes.times,
                nodes.times,
                widths[mi],
                widths[mi],
                lags[mi],
                lags[mi],
                grouped.length_scale,
            )
            residual = jnp.asarray(grouped.systems[run].values) - offsets[ki]
            result += gaussian_score()(K, weights[ki], residual, noise[gi], nodes.observation_nodes)
        return result

    value, gradient = jax.jit(jax.value_and_grad(objective))(x)
    expected, expected_gradient = dense.value_gradient(x)
    tolerance = 2e-6 if case == "high_signal" else 1e-8
    np.testing.assert_allclose(value, expected, rtol=tolerance, atol=1e-7)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=tolerance, atol=1e-5)
    # Direction stays inside support, including positive noise at high SNR.
    direction = np.random.default_rng(998).normal(size=len(x)) * np.minimum(1, np.abs(x) + 1e-4)
    direction[0] = 0
    h = 1e-5
    finite_difference = (
        float(dense.objective(x + h * direction)) - float(dense.objective(x - h * direction))
    ) / (2 * h)
    np.testing.assert_allclose(
        np.asarray(gradient) @ direction, finite_difference, rtol=2e-5, atol=2e-4
    )
