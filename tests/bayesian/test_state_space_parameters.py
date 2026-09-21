"""Independent integral and transfer oracles for variable response realizations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import DoubleGamma, Gamma, Identity, Response


@pytest.mark.parametrize("kernel", [Gamma(3, 0.8), DoubleGamma(3, 0.7, 7, 1.1, 0.25)])
def test_finite_gamma_normalization_and_scale_ratio_derivatives(kernel):
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.state_space_parameters import (
        finite_gamma_energy,
    )

    jax, jnp, _, _ = runtime()
    names = [n for n in kernel.parameters if n != "lag" and "shape" not in n]
    x = np.array([kernel.parameters[n] for n in names])

    def energy(values):
        p = dict(kernel.parameters)
        p.update(zip(names, values))
        return finite_gamma_energy(kernel, p)

    value, gradient = jax.value_and_grad(energy)(jnp.asarray(x))
    assert_allclose(value, kernel._energy, atol=1e-12, rtol=1e-11)
    for i, name in enumerate(names):
        step = 1e-5
        expected = (
            kernel.with_parameters(**{name: x[i] + step})._energy
            - kernel.with_parameters(**{name: x[i] - step})._energy
        ) / (2 * step)
        assert_allclose(gradient[i], expected, atol=2e-8, rtol=2e-7)


def learned_responses():
    return {
        "ref": Response(Identity(), estimate=False, pooling="shared"),
        "a": Response(
            Gamma(3, 0.7, 0.2),
            pooling="shared",
            fixed={"shape": 3},
            bounds={"scale": (0.5, 0.9), "lag": (-1, 1)},
        ),
        "b": Response(
            DoubleGamma(3, 0.7, 7, 0.7, 0.2, -0.3),
            pooling="shared",
            fixed={"peak_shape": 3, "undershoot_shape": 7},
            bounds={
                "peak_scale": (0.6, 0.8),
                "undershoot_scale": (0.65, 0.9),
                "undershoot_ratio": (0.1, 0.3),
                "lag": (-1, 1),
            },
        ),
    }


def test_variable_realization_matches_independent_transfer_and_stationarity():
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    responses = learned_responses()
    model = ResponseStateSpace.prepare(responses, 3.0, 1e-5)
    indices = {
        ("filter", m, p): i
        for i, (m, p) in enumerate((m, p) for m, r in responses.items() for p in r.free_parameters)
    }
    x = np.array([responses[m].initial_kernel().parameters[p] for _, m, p in indices])
    dimensions = []
    for scale in (0.65, 0.7, 0.70000001, 0.85):
        x[indices["filter", "a", "scale"]] = scale
        F, b, C, lags = map(np.asarray, model.realize(x, indices))
        dimensions.append(len(F))
        assert_allclose(F + F.T + np.outer(b, b), 0, atol=1e-12)
        for i, (m, response) in enumerate(responses.items()):
            kernel = response.initial_kernel().with_parameters(
                **{p: x[indices["filter", m, p]] for p in response.free_parameters}
            )
            for omega in (0, 0.1, 1, 10):
                s = 1j * omega
                rate = np.sqrt(3) / 3
                latent = np.sqrt(4 * rate**3) / (s + rate) ** 2
                response_transfer = (
                    1
                    if type(kernel) is Identity
                    else (
                        (1 + s * kernel.scale) ** (-kernel.shape)
                        if type(kernel) is Gamma
                        else (1 + s * kernel.peak_scale) ** (-kernel.peak_shape)
                        - kernel.undershoot_ratio
                        * (1 + s * kernel.undershoot_scale) ** (-kernel.undershoot_shape)
                    )
                    / kernel._energy
                )
                expected = latent * response_transfer
                actual = C[i] @ np.linalg.solve(s * np.eye(len(F)) - F, b)
                assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
            assert_allclose(lags[i], getattr(kernel, "lag", 0))
        assert_allclose(C[-1] @ C[-1], 1, atol=1e-11)
    assert len(set(dimensions)) == 1


def test_parameterized_transitions_match_scipy_and_physical_derivatives():
    from scipy.linalg import expm

    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.state_space_parameters import (
        parameter_transitions,
    )
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    jax, jnp, _, _ = runtime()
    model = ResponseStateSpace.prepare(learned_responses(), 3.0, 1e-5)
    index = {("filter", "a", "scale"): 0}
    intervals = jnp.array([0, 1e-10, 0.01, 0.7, 10, 3000.0])

    def values(scale):
        F, b, _, _ = model.realize(jnp.array([scale]), index)
        return parameter_transitions(model, F, b, intervals, 1)

    actual = jax.jit(values)(0.7)
    F, b, _, _ = map(np.asarray, model.realize(np.array([0.7]), index))
    for i, dt in enumerate(np.asarray(intervals)):
        A = expm(F * dt)
        assert_allclose(actual[0][i], A, atol=3e-12, rtol=1e-10)
        assert_allclose(actual[0][i] @ actual[0][i].T + actual[1][i], np.eye(len(F)), atol=3e-12)
        assert np.linalg.eigvalsh(actual[1][i]).min() > -1e-12

    def value(s):
        return sum(jnp.sum(a) for a in values(s))

    derivative = jax.jit(jax.grad(value))(0.7)
    finite = (float(value(0.70001)) - float(value(0.69999))) / 2e-5
    assert np.isfinite(derivative)
    assert_allclose(derivative, finite, atol=2e-7, rtol=2e-6)
