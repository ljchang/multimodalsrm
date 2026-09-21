"""Public routing and archive compatibility for Gaussian state-space methods."""

import copy

import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone

from .test_bayesian_problem import api, problem_fixture


def test_estimator_validates_and_clones_gaussian_method():
    dense, _, _ = problem_fixture()
    b = api()
    model = b.BayesianMultimodalSRM(
        priors=dense.priors,
        anchor=dense.anchor,
        responses=dense.responses,
        inference="map",
        linear_algebra="state_space",
        state_space_gaussian="rational",
    )
    assert clone(model).state_space_gaussian == "rational"
    model.set_params(state_space_gaussian="unknown")
    with pytest.raises(ValueError, match="state_space_gaussian.*auto.*laguerre.*rational"):
        model._config()


def test_problem_validates_gaussian_method():
    dense, _, _ = problem_fixture()
    with pytest.raises(ValueError, match="state_space_gaussian.*auto.*laguerre.*rational"):
        api().BayesianProblem(
            dense.adapter,
            dense.priors,
            anchor=dense.anchor,
            linear_algebra="state_space",
            state_space_gaussian="unknown",
        )


@pytest.mark.parametrize(
    "method,expected_kwargs",
    [
        ("auto", {}),
        ("laguerre", {"gaussian_method": "laguerre"}),
        ("rational", {"gaussian_method": "rational"}),
    ],
)
def test_problem_routes_explicit_method_but_preserves_legacy_prepare_call(
    monkeypatch, method, expected_kwargs
):
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    dense, _, _ = problem_fixture()
    original = ResponseStateSpace.prepare
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ResponseStateSpace, "prepare", capture)
    api().BayesianProblem(
        dense.adapter,
        dense.priors,
        anchor=dense.anchor,
        linear_algebra="state_space",
        state_space_gaussian=method,
    )
    assert calls == [expected_kwargs]


def test_nondefault_method_is_part_of_fitted_specification_only():
    dense, _, _ = problem_fixture()
    b = api()
    common = dict(
        priors=dense.priors,
        anchor=dense.anchor,
        responses=dense.responses,
        inference="map",
        linear_algebra="state_space",
    )
    assert "state_space_gaussian" not in b.BayesianMultimodalSRM(**common)._specification()
    explicit = b.BayesianMultimodalSRM(**common, state_space_gaussian="laguerre")._specification()
    assert explicit["state_space_gaussian"] == "laguerre"


def test_legacy_gaussian_archive_restores_laguerre_and_replays_exactly():
    from multimodalsrm import Gaussian, Response
    from multimodalsrm.bayesian.persistence import (
        _model_state,
        _restore_model,
    )

    from .test_state_space_parameter_workflows import workflow_fixture

    response = Response(
        Gaussian(0.8, 0.15),
        pooling="shared",
        fixed={"lag": 0.15},
    )
    model, data = workflow_fixture(response)
    model.set_params(state_space_gaussian="laguerre")
    model.fit(data)
    state = copy.deepcopy(_model_state(model))
    state["constructor"].pop("state_space_gaussian")
    state["fit"]["configuration"]["state_space"]["response_approximation"] = (
        "qualified_Gaussian_Laguerre_and_restored_gamma_tails"
    )

    restored, _ = _restore_model(state)
    assert restored.state_space_gaussian == "laguerre"
    assert restored.configuration_ == model.configuration_
    assert restored._specification()["state_space_gaussian"] == "laguerre"
    query = {"train": [52.0, 60.0, 72.0]}
    actual = restored.infer_latent(times=query)["train"]
    expected = model.infer_latent(times=query)["train"]
    assert_array_equal(actual.values, expected.values)
    assert_array_equal(actual.variance, expected.variance)
