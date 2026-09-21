"""Production compact Gaussian responses retain physical inference semantics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import DoubleGamma, Gamma, Gaussian, Identity, Response


def response(width=0.55, estimate=True):
    return Response(
        Gaussian(width, 0.35),
        pooling="shared",
        estimate=estimate,
        bounds={"width": (0.2, 1.1), "lag": (-0.8, 1.2)},
    )


def prepare(responses=None, tolerance=1e-6, method="auto"):
    pytest.importorskip("jax").config.update("jax_enable_x64", True)
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    responses = responses or {
        "ref": Response(Identity(), estimate=False),
        "signal": response(),
    }
    if method == "auto":
        return ResponseStateSpace.prepare(responses, 1.5, tolerance)
    return ResponseStateSpace.prepare(responses, 1.5, tolerance, gaussian_method=method)


def test_qualified_compact_gaussian_is_used_for_the_entire_width_box():
    model = prepare()
    assert model.dimension == 22
    metadata = model.metadata()
    assert metadata["error_bound_scope"] == "entire_declared_response_parameter_box"
    assert metadata["temporal_covariance_error_bound"] <= 1e-6
    assert metadata["responses"]["signal"]["gaussian_template"]["order"] == 20
    assert (
        metadata["response_approximation"]
        == "qualified_Gaussian_vector_fit_and_restored_gamma_tails"
    )


@pytest.mark.parametrize("width", [0.2, 0.55, 1.1])
def test_compact_realization_width_lag_derivatives_match_independent_transfer(width):
    from scipy.integrate import quad

    from multimodalsrm.bayesian._backend import runtime

    jax, jnp, _, _ = runtime()
    model = prepare()
    indices = {("filter", "signal", "width"): 0, ("filter", "signal", "lag"): 1}
    omega = np.array([0.0, 0.3, 1.1, 4.0])

    def transfer(x):
        F, b, C, lags = model.realize(x, indices)

        def one(w):
            return (
                C[1]
                @ jnp.linalg.solve(1j * w * jnp.eye(model.dimension) - F, b)
                * jnp.exp(-1j * w * lags[1])
            )

        y = jax.vmap(one)(jnp.asarray(omega))
        return jnp.stack([y.real, y.imag])

    def oracle(w, lag):
        k = Gaussian(w, lag)
        h = np.array(
            [
                quad(lambda u: float(k(u)) * np.cos(f * u), *k.support, epsabs=1e-11)[0]
                - 1j * quad(lambda u: float(k(u)) * np.sin(f * u), *k.support, epsabs=1e-11)[0]
                for f in omega
            ]
        )
        a = np.sqrt(3) / 1.5
        h *= np.sqrt(4 * a**3) / (1j * omega + a) ** 2
        return np.stack([h.real, h.imag])

    point = jnp.array([width, -0.2])
    actual = jax.jit(transfer)(point)
    jac = jax.jit(jax.jacfwd(transfer))(point)
    assert_allclose(actual, oracle(*point), atol=5e-8, rtol=1e-7)
    for i in range(2):
        step = np.eye(2)[i] * 1e-5
        expected = (oracle(*(point + step)) - oracle(*(point - step))) / 2e-5
        assert_allclose(jac[..., i], expected, atol=2e-6, rtol=2e-6)


def test_compact_mixed_banks_coincident_poles_and_stationarity():
    responses = {
        "ref": Response(Identity(), estimate=False),
        "signal": response(),
        "other": response(0.7),
        "gamma": Response(Gamma(3, 0.7), estimate=False),
    }
    model = prepare(responses)
    indices = {("filter", m, p): i for i, (m, p) in enumerate(model.variable_parameters)}
    x = np.array([responses[m].initial_kernel().parameters[p] for _, m, p in indices])
    x[indices["filter", "other", "width"]] = x[indices["filter", "signal", "width"]]
    F, b, C, lags = map(np.asarray, model.realize(x, indices))
    assert model.dimension == 45
    assert_allclose(F + F.T + np.outer(b, b), 0, atol=2e-12)
    assert_allclose(C[-1, 2:], 0, atol=1e-12)
    assert_allclose(C[-1] @ C[-1], 1, atol=1e-12)
    assert np.linalg.eigvals(F).real.max() < 0
    assert_allclose(C[1], C[2], atol=2e-9, rtol=1e-8)
    assert_allclose(lags[1], lags[2], atol=1e-14)


def test_legacy_selection_and_strict_tolerance_are_explicit():
    assert prepare(method="laguerre").dimension >= 42
    with pytest.raises(ValueError, match="bound|accuracy|tolerance"):
        prepare(tolerance=1e-12, method="rational")
    assert prepare(tolerance=1e-7, method="auto").dimension <= 26


def test_shared_compact_banks_preserve_distinct_signed_response_readouts():
    from scipy.integrate import quad

    kernels = [
        Gaussian(0.55, -0.3),
        Gaussian(0.55, 0.8),
        Gamma(2, 0.7, -0.2),
        Gamma(5, 0.7, 0.4),
        DoubleGamma(3, 0.7, 7, 0.7, 0.3, 0.2),
    ]
    model = prepare({str(i): Response(k, estimate=False) for i, k in enumerate(kernels)})
    assert model.dimension == 29  # Two latent, twenty Gaussian, seven Gamma states.
    rate = np.sqrt(3) / 1.5
    for omega in [0.0, 0.3, 1.1, 4.0]:
        actual = (
            model.outputs[:-1]
            @ np.linalg.solve(1j * omega * np.eye(model.dimension) - model.generator, model.driving)
            * np.exp(-1j * omega * model.lags[:-1])
        )
        expected = []
        for kernel in kernels:
            real = quad(
                lambda u: float(kernel(u)) * np.cos(omega * u),
                *kernel.support,
                epsabs=1e-11,
            )[0]
            imag = quad(
                lambda u: float(kernel(u)) * np.sin(omega * u),
                *kernel.support,
                epsabs=1e-11,
            )[0]
            expected.append((real - 1j * imag) * np.sqrt(4 * rate**3) / (1j * omega + rate) ** 2)
        assert_allclose(actual, expected, atol=8e-8, rtol=1e-7)


def test_two_factor_compact_inference_matches_grouped_with_coupled_posterior():
    from multimodalsrm.bayesian.prediction import project

    from .test_bayesian_problem import api, problem_fixture

    base, adapter, _ = problem_fixture(gaussian=True, two_runs=True)
    adapter.set_params(features=2)
    b = api()
    grouped = b.BayesianProblem(adapter, base.priors, anchor=base.anchor, linear_algebra="grouped")
    state = b.BayesianProblem(
        adapter, base.priors, anchor=base.anchor, linear_algebra="state_space"
    )
    x = grouped.initial.copy()
    rng = np.random.default_rng(1821)
    for i, name in enumerate(grouped.names):
        if name[0] == "loading":
            x[i] = rng.normal(0.5, 0.2)
        elif name[0] == "noise":
            x[i] = 0.2
    x[grouped.indices["filter", "signal", "width"]] = 0.4
    x[grouped.indices["filter", "signal", "lag"]] = 0.6
    actual, gradient = state.value_gradient(x)
    expected, reference = grouped.value_gradient(x)
    assert_allclose(actual, expected, atol=2e-6, rtol=1e-8)
    assert_allclose(gradient, reference, atol=5e-6, rtol=2e-6)
    for key in (None, ("b", "signal", 0)):
        kwargs = {"latent_loading": np.array([0.7, -0.4])} if key is None else {}
        actual = project(
            state,
            x[None],
            "train",
            np.array([1.8, 7.3, 17.0]),
            key=key,
            include_noise=True,
            **kwargs,
        )
        expected = project(
            grouped,
            x[None],
            "train",
            np.array([1.8, 7.3, 17.0]),
            key=key,
            include_noise=True,
            **kwargs,
        )
        assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
