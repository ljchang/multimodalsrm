"""Gaussian observation responses: independent covariance and derivative oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Gamma, Gaussian, Identity, Response

from .test_bayesian_problem import api, independent_parameters, problem_fixture


@pytest.mark.parametrize("width", [0.03, 0.5, 3.0])
def test_fixed_gaussian_covariances_against_analytic_and_finite_integrals(width):
    from multimodalsrm.bayesian._backend import temporal
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    from ..reference.continuous_covariance import response_covariance

    kernels = [Identity(), Gaussian(width, -0.3), Gaussian(width * 1.2, 0.6)]
    state = ResponseStateSpace.prepare(
        {str(i): Response(k, estimate=False, pooling="shared") for i, k in enumerate(kernels)},
        3.0,
        1e-7,
    )
    times = np.array([-0.8, 0.0, 1.2, 10.0])
    for a, ka in enumerate(kernels):
        for b, kb in enumerate(kernels):
            actual = state.covariance(times, np.full(len(times), a), [0.0], [b])
            expected = temporal(
                times,
                [0.0],
                np.full(len(times), getattr(ka, "width", 0.0)),
                np.array([getattr(kb, "width", 0.0)]),
                np.full(len(times), getattr(ka, "lag", 0.0)),
                np.array([getattr(kb, "lag", 0.0)]),
                3.0,
            )
            assert_allclose(actual, expected, atol=state.tail_bound + 5e-8, rtol=0)
    from scipy.integrate import quad

    gamma = Gamma(3, 0.7, 0.4)
    expected = quad(
        lambda u: (
            float(gamma.evaluate(u))
            * response_covariance([0.2 + u], [0.0], kernels[1], Identity(), 3.0, tolerance=1e-9)[
                0, 0
            ]
        ),
        *gamma.support,
        epsabs=1e-9,
    )[0]
    mixed = ResponseStateSpace.prepare(
        {
            "g": Response(kernels[1], estimate=False, pooling="shared"),
            "gamma": Response(Gamma(3, 0.7, 0.4), estimate=False, pooling="shared"),
        },
        3.0,
        1e-6,
    )
    assert_allclose(
        mixed.covariance([0.2], [0], [0.0], [1]),
        expected,
        atol=mixed.tail_bound + 2e-8,
        rtol=0,
    )


def fixture(two_runs=False):
    dense, _, data = problem_fixture(gaussian=True, two_runs=two_runs)
    state = api().BayesianProblem(
        dense.adapter, dense.priors, anchor=dense.anchor, linear_algebra="state_space"
    )
    return state, dense, independent_parameters(dense), data


def test_learned_gaussian_width_lag_objective_gradients_and_current_covariance():
    state, dense, x, _ = fixture(two_runs=True)
    for width, lag in ((0.3, 0.5), (0.55, 0.8), (0.4, 0.6)):
        x[state.indices["filter", "signal", "width"]] = width
        x[state.indices["filter", "signal", "lag"]] = lag
        actual, gradient = state.value_gradient(x)
        expected, reference = dense.value_gradient(x)
        assert np.isfinite(gradient).all()
        assert_allclose(actual, expected, atol=2e-7, rtol=1e-9)
        assert_allclose(gradient, reference, atol=2e-6, rtol=2e-6)
    assert_allclose(state.covariance(x, "train"), dense.covariance(x, "train"), atol=1e-7, rtol=0)


def test_gaussian_prediction_at_width_dependent_ties_and_draw_specific_variance(
    monkeypatch,
):
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.prediction import project
    from multimodalsrm.bayesian.state_space import smoother

    state, dense, x, _ = fixture()
    # lag - 6*width = -1.8: includes exact ties between a latent query and an
    # observation whose effective clock changes when width is differentiated.
    x[state.indices["filter", "signal", "width"]] = 0.4
    x[state.indices["filter", "signal", "lag"]] = 0.6
    query = np.array([-1.0, 0.0, 1.8, 3.0, 8.0, 21.0])
    draws = np.stack([x, x.copy()])
    draws[1, state.indices["filter", "signal", "width"]] = 0.63
    for key in (None, ("b", "signal", 0)):
        actual = project(state, draws, "train", query, key=key, include_noise=True)
        expected = project(dense, draws, "train", query, key=key, include_noise=True)
        assert np.isfinite(actual).all()
        assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    jax, jnp, _, _ = runtime()
    smooth = smoother(state, "train", query)

    def mean(z):
        return smooth(z)[0].sum()

    gradient = np.asarray(jax.jit(jax.grad(mean))(jnp.asarray(x)))

    def dense_mean(z):
        return project(dense, z[None], "train", query, key=None, include_noise=False)[0].sum()

    for parameter in ("width", "lag"):
        i = state.indices["filter", "signal", parameter]
        dx = np.eye(len(x))[i] * 1e-5
        finite = (dense_mean(x + dx) - dense_mean(x - dx)) / 2e-5
        assert np.isfinite(gradient[i])
        assert_allclose(gradient[i], finite, atol=2e-5, rtol=2e-5)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense fallback in Gaussian state-space inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    assert np.isfinite(state.value_gradient(x)[0])
    assert np.isfinite(project(state, draws, "train", query, key=None, include_noise=False)).all()


def test_gaussian_accuracy_floor_and_full_width_domain_are_explicit():
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    responses = {
        "g": Response(
            Gaussian(0.5, 0.2),
            pooling="shared",
            bounds={"width": (0.2, 2.0), "lag": (-1, 1)},
        )
    }
    model = ResponseStateSpace.prepare(responses, 3.0, 1e-7)
    assert model.metadata()["error_bound_scope"] == "entire_declared_response_parameter_box"
    fixed = ResponseStateSpace.prepare(
        {"g": Response(Gaussian(0.5, 0.2), estimate=False, pooling="shared")},
        3.0,
        1e-7,
    )
    assert model.dimension > fixed.dimension  # The full width box needs more states.
    assert model.tail_bound <= 1e-7
    with pytest.raises(ValueError, match="covariance_tolerance"):
        ResponseStateSpace.prepare(responses, 3.0, 1e-12)
