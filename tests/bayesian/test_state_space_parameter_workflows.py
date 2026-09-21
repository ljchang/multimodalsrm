"""Public workflows preserve physical response parameters and native observations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from sklearn.base import clone

from multimodalsrm import (
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    Response,
    TimeSeries,
)

from .test_bayesian_problem import api


def response_with_one_parameter(parameter):
    if parameter == "scale":
        kernel = Gamma(3, 0.6, 0.25)
        bounds = (0.4, 1.0)
    else:
        kernel = DoubleGamma(3, 0.7, 7, 1.1, 0.15, 0.25)
        bounds = (0.05, 0.5)
    return Response(
        kernel,
        pooling="shared",
        fixed={p: v for p, v in kernel.parameters.items() if p != parameter},
        bounds={parameter: bounds},
    )


def workflow_fixture(response):
    """Small native-clock, masked data; no private preparation or fit injection."""
    b = api()
    kernel = response.initial_kernel()
    filters = {
        "signal": {
            p: b.Prior.normal(kernel.parameters[p], max(0.1, kernel.parameters[p] / 3))
            for p in response.free_parameters
        }
    }
    model = b.BayesianMultimodalSRM(
        responses={
            "ref": Response(Identity(), estimate=False, pooling="shared"),
            "signal": response,
        },
        priors=b.BayesianPriors(noise=b.Prior.lognormal(np.log(0.08), 0.5), filters=filters),
        anchor=("a", "ref", 0),
        reference_modality="ref",
        inference="map",
        linear_algebra="state_space",
        covariance_tolerance=1e-6,
        length_scale=3.0,
        search=b.SearchConfig(starts=1, maxiter=350),
        random_state=613,
    )
    times = np.array(
        [
            0.0,
            50.0,
            51.2,
            53.0,
            54.6,
            56.5,
            59.0,
            61.3,
            64.0,
            67.0,
            70.0,
            74.0,
            80.0,
            90.0,
            100.0,
        ]
    )
    rng = np.random.default_rng(617)
    data = {}
    for subject, loading in (("a", 0.8), ("b", 1.1)):
        data[subject] = {"train": {}}
        for modality in ("ref", "signal"):
            clock = times if modality == "ref" else times[::2] + 0.4

            # Independent finite-support convolution supplies a nontrivial
            # filter-sensitive signal; this is workflow data, not a recovery gate.
            def latent(t):
                return np.sin(t / 3) + 0.3 * np.cos(t / 1.4)

            values = (
                latent(clock)
                if modality == "ref"
                else np.array(
                    [
                        quad(
                            lambda u: float(kernel(u)) * latent(t - u),
                            *kernel.support,
                            epsabs=1e-10,
                        )[0]
                        for t in clock
                    ]
                )
            )
            values = (loading * values + rng.normal(0, 0.1, len(clock)))[:, None]
            mask = np.ones(values.shape, bool)
            mask[2, 0] = False
            data[subject]["train"][modality] = TimeSeries(values, clock, mask)
    return model, data


@pytest.mark.parametrize("parameter", ["scale", "undershoot_ratio"])
def test_fixed_lag_parameter_fit_condition_transform_and_archive(parameter, tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    response = response_with_one_parameter(parameter)
    state, data = workflow_fixture(response)
    grouped = clone(state).set_params(linear_algebra="grouped", response_quadrature_order=192)
    state.fit(data)
    grouped.fit(data)
    assert state.map_diagnostics_["meets_gradient_tolerance"]
    assert grouped.map_diagnostics_["meets_gradient_tolerance"]
    assert_allclose(state.objective_, grouped.objective_, atol=3e-5, rtol=0)
    key = ("filter", "signal", parameter)
    assert [n for n in state.parameter_names_ if n[0] == "filter"] == [key]
    fitted = state.map_parameters_[state.parameter_names_.index(key)]
    expected = grouped.map_parameters_[grouped.parameter_names_.index(key)]
    assert abs(fitted - response.initial_kernel().parameters[parameter]) > 1e-3
    assert_allclose(fitted, expected, atol=2e-4, rtol=0)
    assert_array_equal(state.relative_lag_draws()["signal"], [[0.25]])
    saved_parameters = state.map_parameters_.copy()

    donors = {s: {"held": runs["train"].copy()} for s, runs in data.items()}
    query = np.array([51.0, 55.0, 60.0, 70.0, 85.0])
    actual = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    reference = grouped.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    actual = actual.predict(times=query)["b"]["held"]["signal"]
    reference = reference.predict(times=query)["b"]["held"]["signal"]
    assert_allclose(actual.values, reference.values, atol=3e-4, rtol=0)
    assert_allclose(actual.variance, reference.variance, atol=3e-4, rtol=0)
    approximation = actual.metadata["covariance_approximation"]
    assert approximation["learned_response_parameters"] == {"signal": [parameter]}
    assert approximation["error_bound_scope"] == "entire_declared_response_parameter_box"

    before = state.transform(donors, times=query)
    ts = donors["b"]["held"]["ref"]
    donors["b"]["held"]["ref"] = TimeSeries(-5 * ts.values, ts.times, ts.mask)
    after = state.transform(donors, times=query)
    assert_array_equal(before["a"]["held"].values, after["a"]["held"].values)
    assert_array_equal(before["a"]["held"].variance, after["a"]["held"].variance)
    assert not np.allclose(before["b"]["held"].values, after["b"]["held"].values)

    expected = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    expected = expected.predict(times=query)["b"]["held"]["signal"]
    donors["b"]["held"]["signal"] = object()
    poisoned = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    poisoned = poisoned.predict(times=query)["b"]["held"]["signal"]
    assert_array_equal(poisoned.values, expected.values)
    assert_array_equal(poisoned.variance, expected.variance)
    assert_array_equal(state.map_parameters_, saved_parameters)

    save_model(tmp_path / parameter, state)
    restored, _ = load_model(tmp_path / parameter)
    assert restored.configuration_ == state.configuration_
    assert restored.parameter_names_ == state.parameter_names_
    assert_array_equal(restored.map_parameters_, saved_parameters)
    replay = restored.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    replay = replay.predict(times=query)["b"]["held"]["signal"]
    assert_array_equal(replay.values, expected.values)
    assert_array_equal(replay.variance, expected.variance)
    # Participant transforms must also retain the fitted physical response on load.
    replay = restored.transform({"a": donors["a"]}, times=query)["a"]["held"]
    assert_array_equal(replay.values, after["a"]["held"].values)
    assert_array_equal(replay.variance, after["a"]["held"].variance)


@pytest.mark.parametrize("estimate", [False, True], ids=["fixed", "learned-width-lag"])
def test_gaussian_public_fit_condition_transform_archive_and_physical_lag(estimate, tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    response = Response(
        Gaussian(0.45, 0.35),
        pooling="shared",
        estimate=estimate,
        bounds={"width": (0.2, 0.8), "lag": (-0.4, 1.0)},
    )
    state, data = workflow_fixture(response)
    # Different masks across participants supplement the modality-specific
    # native clocks; neither backend may fill these omitted observations.
    ts = data["b"]["train"]["signal"]
    mask = ts.mask.copy()
    mask[-2, 0] = False
    data["b"]["train"]["signal"] = TimeSeries(ts.values, ts.times, mask)
    grouped = clone(state).set_params(linear_algebra="grouped")
    assert grouped.response_quadrature_order is None  # Analytic Gaussian oracle.
    state.fit(data)
    grouped.fit(data)
    for model in (state, grouped):
        assert model.map_diagnostics_["meets_gradient_tolerance"]
        assert np.isfinite(model.objective_)
        assert np.isfinite(model.map_parameters_).all()
    assert_allclose(state.objective_, grouped.objective_, atol=3e-5, rtol=0)
    filter_names = [name for name in state.parameter_names_ if name[0] == "filter"]
    assert filter_names == (
        [("filter", "signal", "width"), ("filter", "signal", "lag")] if estimate else []
    )
    parameters = dict(response.initial_kernel().parameters)
    for name in filter_names:
        value = state.map_parameters_[state.parameter_names_.index(name)]
        reference = grouped.map_parameters_[grouped.parameter_names_.index(name)]
        assert_allclose(value, reference, atol=3e-4, rtol=0)
        parameters[name[-1]] = value
    physical_lag = parameters["lag"]
    # The public lag is the fitted Gaussian center, not the internal causal
    # bank origin at lag - 6 * width.
    assert_array_equal(state.relative_lag_draws()["signal"], [[physical_lag]])
    saved_parameters = state.map_parameters_.copy()

    donors = {s: {"held": runs["train"].copy()} for s, runs in data.items()}
    query = np.array([51.0, 55.0, 60.0, 70.0, 85.0])
    conditioned = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    reference = grouped.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    assert_array_equal(conditioned.map_parameters_, saved_parameters)
    assert_array_equal(conditioned.relative_lag_draws()["signal"], [[physical_lag]])
    actual = conditioned.predict(times=query, include_noise=True)["b"]["held"]["signal"]
    expected = reference.predict(times=query, include_noise=True)["b"]["held"]["signal"]
    assert np.isfinite(actual.values).all()
    assert np.isfinite(actual.variance).all()
    assert (actual.variance >= 0).all()
    assert_allclose(actual.values, expected.values, atol=4e-4, rtol=0)
    assert_allclose(actual.variance, expected.variance, atol=4e-4, rtol=0)

    before = state.transform(donors, times=query)
    expected_transform = grouped.transform(donors, times=query)
    for subject in donors:
        assert np.isfinite(before[subject]["held"].values).all()
        assert np.isfinite(before[subject]["held"].variance).all()
        assert_allclose(
            before[subject]["held"].values,
            expected_transform[subject]["held"].values,
            atol=4e-4,
            rtol=0,
        )
        assert_allclose(
            before[subject]["held"].variance,
            expected_transform[subject]["held"].variance,
            atol=4e-4,
            rtol=0,
        )
    ts = donors["b"]["held"]["ref"]
    donors["b"]["held"]["ref"] = TimeSeries(-5 * ts.values, ts.times, ts.mask)
    after = state.transform(donors, times=query)
    assert_array_equal(before["a"]["held"].values, after["a"]["held"].values)
    assert_array_equal(before["a"]["held"].variance, after["a"]["held"].variance)
    assert not np.allclose(before["b"]["held"].values, after["b"]["held"].values)

    conditioned = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    expected = conditioned.predict(times=query, include_noise=True)["b"]["held"]["signal"]
    donors["b"]["held"]["signal"] = object()
    poisoned = state.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    actual = poisoned.predict(times=query, include_noise=True)["b"]["held"]["signal"]
    assert_array_equal(actual.values, expected.values)
    assert_array_equal(actual.variance, expected.variance)
    assert_array_equal(state.map_parameters_, saved_parameters)

    save_model(tmp_path / "gaussian", state)
    restored, _ = load_model(tmp_path / "gaussian")
    assert restored.configuration_ == state.configuration_
    assert restored.parameter_names_ == state.parameter_names_
    assert_array_equal(restored.map_parameters_, saved_parameters)
    assert_array_equal(restored.relative_lag_draws()["signal"], [[physical_lag]])
    replay = restored.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    replay = replay.predict(times=query, include_noise=True)["b"]["held"]["signal"]
    assert_array_equal(replay.values, expected.values)
    assert_array_equal(replay.variance, expected.variance)
    replay = restored.transform({"a": donors["a"]}, times=query)["a"]["held"]
    assert_array_equal(replay.values, after["a"]["held"].values)
    assert_array_equal(replay.variance, after["a"]["held"].variance)


@pytest.mark.parametrize("shape", [2.5, 33.0])
def test_public_fit_rejects_unsupported_fixed_gamma_orders(shape):
    response = Response(Gamma(shape, 0.7), pooling="shared", fixed={"shape": shape, "lag": 0.0})
    model, data = workflow_fixture(response)
    with pytest.raises(ValueError, match="state_space.*integers"):
        model.fit(data)


def test_public_fit_rejects_learning_gamma_shape():
    model, data = workflow_fixture(Response(Gamma(3, 0.7), pooling="shared", fixed={"lag": 0.0}))
    with pytest.raises(ValueError, match="state_space.*fixed integer"):
        model.fit(data)


def test_public_fit_rejects_bounds_that_allow_peak_undershoot_reversal():
    response = Response(
        DoubleGamma(3, 0.7, 4, 1.0, 0.2),
        pooling="shared",
        fixed={
            "peak_shape": 3,
            "undershoot_shape": 4,
            "undershoot_ratio": 0.2,
            "lag": 0.0,
        },
        bounds={"peak_scale": (0.5, 1.5), "undershoot_scale": (0.9, 1.1)},
    )
    model, data = workflow_fixture(response)
    with pytest.raises(ValueError, match="bounds.*ordering"):
        model.fit(data)


@pytest.mark.parametrize("upper_scale, tolerance", [(1.0, 1e-12), (1000.0, 1e-6)])
def test_public_fit_enforces_response_error_over_the_entire_scale_domain(upper_scale, tolerance):
    response = Response(
        Gamma(3, 0.7),
        pooling="shared",
        fixed={"shape": 3, "lag": 0.0},
        bounds={"scale": (0.5, upper_scale)},
    )
    model, data = workflow_fixture(response)
    model.set_params(covariance_tolerance=tolerance)
    if upper_scale > 1:
        # Retain support-eligible observations even for the deliberately broad
        # domain, so the representation's accuracy gate is the failing boundary.
        for runs in data.values():
            for modality, ts in runs["train"].items():
                times = ts.times.copy()
                times[0] = -100000.0
                runs["train"][modality] = TimeSeries(ts.values, times, ts.mask)
    with pytest.raises(ValueError, match="error bound.*covariance_tolerance.*parameter bounds"):
        model.fit(data)
