"""Fast-tail Bach full inference against refined finite-response quadrature.

Independent finite convolutions generate data; this is a numerical backend
qualification, not a timing-recovery or general canonical-SCR accuracy claim.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from sklearn.base import clone

from multimodalsrm import BachSCR, Identity, Response, TimeSeries

from .test_bayesian_problem import api


def bach_fixture(learned_lag=True):
    b = api()
    kernel = BachSCR(lambda2=0.25, lag=0.3)
    response = (
        Response.lag_only(kernel, bounds={"lag": (-0.4, 0.8)}, pooling="shared")
        if learned_lag
        else Response(kernel, estimate=False, pooling="shared")
    )
    model = b.BayesianMultimodalSRM(
        responses={
            "ref": Response(Identity(), estimate=False, pooling="shared"),
            "signal": response,
        },
        priors=b.BayesianPriors(
            noise=b.Prior.lognormal(np.log(0.08), 0.5),
            filters={"signal": {"lag": b.Prior.normal(0.3, 0.3)}} if learned_lag else {},
        ),
        anchor=("a", "ref", 0),
        reference_modality="ref",
        inference="map",
        linear_algebra="state_space",
        covariance_tolerance=1e-6,
        length_scale=3.0,
        search=b.SearchConfig(starts=1, maxiter=250),
        random_state=819,
    )
    rng = np.random.default_rng(821)
    data = {}

    def latent(t):
        return np.sin(t / 3) + 0.3 * np.cos(t / 1.4)

    for subject, loading in [("a", 0.8), ("b", 1.1)]:
        data[subject] = {"train": {}}
        for modality in ["ref"] if subject == "a" else ["ref", "signal"]:
            clock = (
                np.array([0.0, 96.0, 98.1, 101.0, 105.2, 110.0, 116.4, 124.0, 240.0])
                if modality == "ref"
                else np.array([95.4, 100.2, 107.1, 114.3, 123.5])
            )
            if subject == "b" and modality == "ref":
                clock = clock.copy()
                clock[1:-1] += 0.17
            values = (
                latent(clock)
                if modality == "ref"
                else np.array(
                    [
                        quad(
                            lambda u: float(kernel(u)) * latent(t - u),
                            *kernel.support,
                            epsabs=1e-11,
                            points=[kernel.lag + kernel.t0],
                        )[0]
                        for t in clock
                    ]
                )
            )
            values = (loading * values + rng.normal(0, 0.1, len(clock)))[:, None]
            mask = np.ones(values.shape, bool)
            if modality == "ref":
                mask[3, 0] = False
            data[subject]["train"][modality] = TimeSeries(values, clock, mask)
    return model, data


@pytest.mark.parametrize("learned_lag", [False, True], ids=["fixed", "lag-only"])
def test_bach_objective_physical_gradient_and_posterior_refined_oracle(learned_lag, monkeypatch):
    from multimodalsrm.bayesian.persistence import _prepare
    from multimodalsrm.bayesian.prediction import project

    model, data = bach_fixture(learned_lag)
    _, state = _prepare(model, data)
    references = []
    for order in [192, 384]:
        oracle = clone(model).set_params(linear_algebra="grouped", response_quadrature_order=order)
        references.append(_prepare(oracle, data)[1])
    coarse, fine = references
    x = state.initial.copy()
    for i, name in enumerate(state.names):
        if name[0] == "loading":
            x[i] = 0.8 if name[1] == "a" else 1.1
        elif name[0] == "noise":
            x[i] = 0.12
        elif name[0] == "offset":
            x[i] = 0.05
    if learned_lag:
        x[state.indices["filter", "signal", "lag"]] = 0.45
    value, gradient = state.value_gradient(x)
    cvalue, cgradient = coarse.value_gradient(x)
    fvalue, fgradient = fine.value_gradient(x)
    assert_allclose(cvalue, fvalue, atol=2e-6, rtol=0)
    assert_allclose(cgradient, fgradient, atol=2e-5, rtol=0)
    assert_allclose(value, fvalue, atol=3e-6, rtol=0)
    assert_allclose(gradient, fgradient, atol=3e-5, rtol=0)
    if learned_lag:
        index = state.indices["filter", "signal", "lag"]
        dx = np.eye(len(x))[index] * 1e-5
        finite = (fine.value_gradient(x + dx)[0] - fine.value_gradient(x - dx)[0]) / 2e-5
        assert_allclose(gradient[index], finite, atol=3e-5, rtol=0)
    query = np.array([96.0, 101.7, 110.0, 121.0])
    for key in [None, ("b", "signal", 0)]:
        kwargs = dict(key=key, include_noise=True)
        coarse_prediction = project(coarse, x[None], "train", query, **kwargs)
        expected = project(fine, x[None], "train", query, **kwargs)
        assert_allclose(coarse_prediction, expected, atol=3e-6, rtol=0)
        actual = project(state, x[None], "train", query, **kwargs)
        assert_allclose(actual, expected, atol=5e-6, rtol=0)

    def unavailable(*args, **kwargs):
        raise AssertionError("dense fallback during Bach state inference")

    monkeypatch.setattr(state, "covariance", unavailable)
    monkeypatch.setattr(state, "temporal_covariance", unavailable)
    assert np.isfinite(state.value_gradient(x)[0])
    assert np.isfinite(project(state, x[None], "train", query, key=None, include_noise=False)).all()


def test_bach_public_fit_condition_target_exclusion_and_archive(tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    state, data = bach_fixture()
    grouped = clone(state).set_params(linear_algebra="grouped", response_quadrature_order=384)
    state.fit(data)
    grouped.fit(data)
    for model in [state, grouped]:
        assert model.map_diagnostics_["meets_gradient_tolerance"]
    assert_allclose(state.objective_, grouped.objective_, atol=3e-5, rtol=0)
    filters = [n for n in state.parameter_names_ if n[0] == "filter"]
    assert filters == [("filter", "signal", "lag")]
    index = state.parameter_names_.index(filters[0])
    physical_lag = state.map_parameters_[index]
    assert_allclose(physical_lag, grouped.map_parameters_[index], atol=3e-4, rtol=0)
    assert_array_equal(state.relative_lag_draws()["signal"], [[physical_lag]])
    donors = {s: {"held": runs["train"].copy()} for s, runs in data.items()}
    query = np.array([96.0, 101.7, 110.0, 121.0])

    def prediction(model):
        return model.condition(donors, targets={"b": ["signal"]}, mode="frozen").predict(
            times=query, include_noise=True
        )["b"]["held"]["signal"]

    actual, expected = prediction(state), prediction(grouped)
    assert_allclose(actual.values, expected.values, atol=4e-4, rtol=0)
    assert_allclose(actual.variance, expected.variance, atol=4e-4, rtol=0)
    assert (
        actual.metadata["covariance_approximation"]["bach_supported_learning"]
        == "additional_lag_only_fixed_shape"
    )
    parameters = state.map_parameters_.copy()
    donors["b"]["held"]["signal"] = object()
    poisoned = prediction(state)
    assert_array_equal(poisoned.values, actual.values)
    assert_array_equal(poisoned.variance, actual.variance)
    assert_array_equal(state.map_parameters_, parameters)
    save_model(tmp_path / "bach", state)
    restored, _ = load_model(tmp_path / "bach")
    assert restored.configuration_ == state.configuration_
    assert_array_equal(restored.map_parameters_, parameters)
    assert_array_equal(restored.relative_lag_draws()["signal"], [[physical_lag]])
    replay = prediction(restored)
    assert_array_equal(replay.values, actual.values)
    assert_array_equal(replay.variance, actual.variance)
