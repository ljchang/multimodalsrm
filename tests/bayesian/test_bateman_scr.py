"""SCR transfer functions, parameter gradients, GP inference and persistence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import BatemanSCR, Identity, Response, TimeSeries
from multimodalsrm.bayesian._backend import runtime
from multimodalsrm.bayesian.state_space_responses import ResponseStateSpace


def test_transfer_stationarity_and_coincident_poles():
    jax, jnp, _, _ = runtime()
    response = Response(
        BatemanSCR(1, 2, 0.3),
        pooling="shared",
        bounds={"rise": (0.5, 3), "decay": (0.5, 4), "lag": (-1, 1)},
    )
    model = ResponseStateSpace.prepare({"scr": response}, np.sqrt(3) * 2, 1e-5)
    assert model.dimension == 4
    index = {("filter", "scr", p): i for i, p in enumerate(("rise", "decay", "lag"))}
    for values in ([1.0, 2.0, 0.3], [2.0, 2.0, 0.3], [2.0, 2.0 + 1e-9, 0.3], [3.0, 0.7, -0.2]):
        F, b, C, lags = map(np.asarray, model.realize(jnp.array(values), index))
        assert_allclose(F + F.T + np.outer(b, b), 0, atol=1e-13)
        k = BatemanSCR(*values)
        for omega in (0, 0.1, 1, 10):
            s = 1j * omega
            expected = (
                np.sqrt(4 * 0.5**3)
                / (s + 0.5) ** 2
                / (1 + s * k.rise)
                / (1 + s * k.decay)
                / k._energy
            )
            assert_allclose(
                C[0] @ np.linalg.solve(s * np.eye(len(F)) - F, b), expected, rtol=2e-10, atol=2e-12
            )
        assert_allclose(lags, [values[2], 0])

        def score(x):
            return jnp.sum(model.realize(x, index)[2] ** 2)

        grad = np.asarray(jax.grad(score)(jnp.array(values)))
        assert np.isfinite(grad).all()
        for i in range(3):
            step = np.eye(3)[i] * 1e-5
            expected = (
                float(score(jnp.array(values) + step)) - float(score(jnp.array(values) - step))
            ) / 2e-5
            assert_allclose(grad[i], expected, atol=2e-8, rtol=3e-6)


def fixture(algebra="state_space", order=64):
    from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
    from multimodalsrm.bayesian.persistence import _prepare

    kernel = BatemanSCR(0.7, 2.0, 0.1)
    response = Response(
        kernel,
        pooling="shared",
        bounds={"rise": (0.4, 1.2), "decay": (1.5, 3.0), "lag": (-0.5, 0.5)},
    )
    model = BayesianMultimodalSRM(
        features=1,
        anchor=("a", "ref", 0),
        reference_modality="ref",
        responses={"ref": Response(Identity(), estimate=False, pooling="shared"), "scr": response},
        priors=BayesianPriors(
            noise=Prior.lognormal(-1, 0.5),
            filters={"scr": {p: Prior.normal(v, 0.4) for p, v in kernel.parameters.items()}},
        ),
        length_scale=3.0,
        inference="map",
        linear_algebra=algebra,
        response_quadrature_order=order if algebra != "state_space" else None,
        covariance_tolerance=1e-6,
        search=SearchConfig(starts=1, maxiter=50),
        random_state=13,
    )
    t = np.r_[0.0, np.arange(94.0, 118.0, 3.0)]
    data = {
        s: {
            "train": {
                m: TimeSeries(
                    (np.sin(t / 4 + 0.2 * j) + 0.1 * np.cos(t / 2))[:, None],
                    t + (0.13 if m == "scr" else 0),
                )
                for m in ("ref", "scr")
            }
        }
        for j, s in enumerate(("a", "b"))
    }
    _, problem = _prepare(model, data)
    x = problem.initial.copy()
    for i, n in enumerate(problem.names):
        if n[0] == "loading":
            x[i] = 0.7
        if n[0] == "noise":
            x[i] = 0.3
    return model, problem, x, data


def test_gp_values_gradients_predictions_and_archive():
    from multimodalsrm.bayesian import _archive
    from multimodalsrm.bayesian.prediction import project

    model, state, x, _ = fixture()
    _, dense, _, _ = fixture("dense", 96)
    actual, gradient = state.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert_allclose(actual, expected, atol=3e-6, rtol=1e-8)
    assert_allclose(gradient, expected_gradient, atol=3e-5, rtol=3e-6)
    draws = np.stack([x, x.copy()])
    draws[1, state.indices["filter", "scr", "rise"]] = 1.0
    for key in (None, ("a", "scr", 0)):
        a = project(
            state, draws, "train", np.array([95.1, 102.2, 112.3]), key=key, include_noise=True
        )
        b = project(
            dense, draws, "train", np.array([95.1, 102.2, 112.3]), key=key, include_noise=True
        )
        assert_allclose(a, b, atol=3e-6, rtol=3e-6)
    arrays = {}
    restored = _archive.decode(_archive.encode(model.responses, arrays), arrays, set())
    assert _archive.same(restored, model.responses)


def test_tail_tolerance_enforced():
    with pytest.raises(ValueError, match="bound"):
        ResponseStateSpace.prepare(
            {"scr": Response(BatemanSCR(0.7, 20), estimate=False)}, 3.0, 1e-6
        )


def test_posterior_and_grouped_admit_same_density():
    from multimodalsrm.bayesian.persistence import _prepare
    from multimodalsrm.bayesian.response_scope import validate_posterior_response_target

    model, dense, x, data = fixture("dense", 32)
    model.set_params(inference="posterior", linear_algebra="grouped")
    _, posterior = _prepare(model, data)
    validate_posterior_response_target(posterior)
    assert dense.names == posterior.names
    actual = posterior.value_gradient(x)
    expected = dense.value_gradient(x)
    assert_allclose(actual[0], expected[0], atol=2e-9)
    assert_allclose(actual[1], expected[1], atol=2e-8)


def test_fitted_state_space_archive_roundtrip(tmp_path):
    import warnings

    from multimodalsrm.bayesian import SearchConfig
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, _, _, data = fixture()
    model.set_params(search=SearchConfig(starts=1, maxiter=300))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    save_model(tmp_path / "model", model)
    restored, _ = load_model(tmp_path / "model")
    assert type(restored.responses["scr"].kernel) is BatemanSCR
    assert_allclose(restored.map_parameters_, model.map_parameters_, rtol=0, atol=0)
    assert restored.configuration_["state_space"] == model.configuration_["state_space"]
    assert_allclose(
        restored.problem_.value_gradient(restored.map_parameters_)[0], model.objective_, atol=1e-9
    )


def posterior_fixture():
    from multimodalsrm.bayesian import BayesianPriors, Prior

    from .test_bayesian_gamma_posterior import gamma_fixture

    model, data = gamma_fixture(k=2, order=32, learned=True)
    kernel = BatemanSCR(0.7, 2.0, 0.1)
    model.responses["signal"] = Response(
        kernel,
        pooling="shared",
        bounds={"rise": (0.4, 1.2), "decay": (1.5, 3.0), "lag": (-0.1, 0.3)},
    )
    model.priors = BayesianPriors(
        noise=model.priors.noise,
        filters={"signal": {p: Prior.normal(v, 0.3) for p, v in kernel.parameters.items()}},
    )
    model.length_scale = Prior.lognormal(np.log(3.0), 0.35).bounded(2.0, 5.0)
    times = np.r_[0.0, np.arange(94.0, 112.0, 2.0)]
    for runs in data.values():
        for modalities in runs.values():
            for m, series in modalities.items():
                modalities[m] = TimeSeries(series.values, times)
    return model, data


def test_multifactor_learned_timescale_posterior_and_quadrature_convergence():
    from multimodalsrm.bayesian.persistence import _prepare

    model, data = posterior_fixture()
    _, grouped = _prepare(model, data)
    model.set_params(linear_algebra="dense")
    _, dense = _prepare(model, data)
    model.set_params(response_quadrature_order=64)
    _, fine = _prepare(model, data)
    x = grouped.initial.copy()
    x[grouped.indices["gp", "length_scale"]] = 3.0
    rng = np.random.default_rng(398)
    for i, n in enumerate(grouped.names):
        if n[0] == "loading":
            x[i] = rng.normal(scale=0.5)
        if n[0] == "noise":
            x[i] = 0.3
    expected = fine.value_gradient(x)
    for p in (grouped, dense):
        actual = p.value_gradient(x)
        assert_allclose(actual[0], expected[0], atol=2e-6, rtol=1e-8)
        assert_allclose(actual[1], expected[1], atol=2e-5, rtol=2e-6)


def test_multifactor_posterior_execution_and_archive(tmp_path):
    import warnings

    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data = posterior_fixture()
    # Tiny sampling budget verifies execution and serialization, not calibration.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert model.parameter_draws_.shape[:2] == (2, 4)
    assert np.isfinite(model.parameter_draws_).all()
    save_model(tmp_path / "posterior", model)
    restored, _ = load_model(tmp_path / "posterior")
    assert type(restored.responses["signal"].kernel) is BatemanSCR
    assert_allclose(restored.parameter_draws_, model.parameter_draws_, rtol=0, atol=0)
