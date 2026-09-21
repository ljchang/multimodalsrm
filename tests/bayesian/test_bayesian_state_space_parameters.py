"""Physical-parameter and prediction oracles for learned response scales/ratios."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Response

from .test_bayesian_problem import api
from .test_bayesian_state_space_responses import fixture as fixed_fixture


def fixture(algebra="state_space", features=1, two_runs=False, order=192):
    from multimodalsrm.bayesian.persistence import _prepare

    model, fixed, base, data = fixed_fixture(features=features, two_runs=two_runs)
    b = api()
    responses = dict(model.responses)
    responses["brain"] = Response(
        responses["brain"].initial_kernel(),
        pooling="shared",
        fixed={"peak_shape": 3, "undershoot_shape": 7},
        bounds={
            "peak_scale": (0.6, 0.8),
            "undershoot_scale": (0.9, 1.3),
            "undershoot_ratio": (0.2, 0.4),
            "lag": (-2, 2),
        },
    )
    responses["aux"] = Response(
        responses["aux"].initial_kernel(),
        pooling="shared",
        fixed={"shape": 3},
        bounds={"scale": (0.5, 0.9), "lag": (-2, 2)},
    )
    filters = {
        m: {p: b.Prior.normal(r.initial_kernel().parameters[p], 0.5) for p in r.free_parameters}
        for m, r in responses.items()
        if r.free_parameters
    }
    model.set_params(
        responses=responses,
        priors=b.BayesianPriors(noise=b.Prior.lognormal(-1, 0.5), filters=filters),
        linear_algebra=algebra,
        response_quadrature_order=order if algebra != "state_space" else None,
        covariance_tolerance=1e-5,
    )
    _, problem = _prepare(model, data)
    parameters = dict(zip(fixed.names, base))
    parameters.update(
        {
            ("filter", m, p): r.initial_kernel().parameters[p]
            for m, r in responses.items()
            for p in r.free_parameters
        }
    )
    x = np.array([parameters[n] for n in problem.names])
    return model, problem, x, data


@pytest.mark.parametrize("features", [1, 2])
def test_changing_scales_ratios_and_lags_matches_dense_value_and_gradient(features):
    _, state, x, _ = fixture(features=features, two_runs=True)
    _, dense, _, _ = fixture("dense", features=features, two_runs=True)
    x[state.indices["filter", "aux", "scale"]] = 0.83
    x[state.indices["filter", "brain", "peak_scale"]] = 0.63
    x[state.indices["filter", "brain", "undershoot_ratio"]] = 0.36
    actual, gradient = state.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert np.isfinite(gradient).all()
    assert_allclose(actual, expected, atol=4e-6, rtol=1e-8)
    assert_allclose(gradient, expected_gradient, atol=3e-5, rtol=3e-6)


def test_scale_dependent_prediction_uses_draw_specific_outputs_and_variances(
    monkeypatch,
):
    from multimodalsrm.bayesian.prediction import project

    _, state, x, _ = fixture()
    _, dense, _, _ = fixture("dense")
    draws = np.stack([x, x.copy()])
    draws[1, state.indices["filter", "aux", "scale"]] = 0.85
    draws[1, state.indices["filter", "brain", "undershoot_ratio"]] = 0.21
    query = np.array([48.75, 50.0, 52.25, 58.4, 72.0])

    def unavailable(*args, **kwargs):
        raise AssertionError("dense fallback during parameterized state-space inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    for key in (None, ("a", "brain", 0), ("b", "aux", 1)):
        expected = project(dense, draws, "train", query, key=key, include_noise=True)
        actual = project(state, draws, "train", query, key=key, include_noise=True)
        assert np.isfinite(actual).all()
        assert_allclose(actual, expected, atol=3e-6, rtol=2e-6)
