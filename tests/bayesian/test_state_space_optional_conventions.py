"""State-space routing preserves optional training conventions during readout."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .test_bayesian_optional_conventions import prepared, setup


@pytest.mark.parametrize("features", [1, 2])
def test_state_space_unreferenced_lags_match_grouped_and_ignore_common_shift(features):
    model, grouped, _ = prepared(features, linear_algebra="grouped")
    _, state, _ = prepared(features, linear_algebra="state_space")
    assert state.anchor == grouped.anchor
    assert state.reference_convention == grouped.reference_convention
    x = grouped.initial.copy()
    rng = np.random.default_rng(292)
    for i, name in enumerate(grouped.names):
        if name[0] == "loading":
            x[i] = rng.uniform(0.2, 0.8)
    indices = [grouped.indices["filter", m, "lag"] for m in model.responses]
    x[indices] = [0.4, -0.2]
    actual, gradient = state.value_gradient(x)
    expected, reference = grouped.value_gradient(x)
    assert_allclose(actual, expected, atol=2e-6, rtol=1e-8)
    assert_allclose(gradient, reference, atol=5e-6, rtol=2e-6)
    shifted = x.copy()
    shifted[indices] += 0.25
    assert_allclose(state.nll(x), state.nll(shifted), atol=1e-9, rtol=0)


def test_state_space_optional_conventions_survive_condition_transform_and_archive(
    tmp_path,
):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    _, model, data = setup()
    model.set_params(linear_algebra="state_space", state_space_gaussian="rational")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert model.anchor is None and model.reference_modality is None
    path = tmp_path / "state-optional.zip"
    save_model(path, model)
    restored, _ = load_model(path)
    assert restored.configuration_ == model.configuration_
    query = np.array([3.0, 5.0, 7.0])
    # The held-out donor set omits the auto-selected sign-anchor participant.
    donors = {"b": {"heldout": copy.deepcopy(data["b"]["train"])}}
    before = model.transform(donors, times=query)["b"]["heldout"]
    after = restored.transform(donors, times=query)["b"]["heldout"]
    assert_array_equal(before.values, after.values)
    assert_array_equal(before.variance, after.variance)
    assert before.metadata["reference_convention"] == model.problem_.reference_convention
    conditioned = model.condition(donors, targets={"b": ["brain"]}, mode="frozen")
    assert conditioned.problem_.reference_convention == model.problem_.reference_convention
    assert conditioned.problem_.state_space_gaussian == "rational"
    assert_array_equal(conditioned.map_parameters_, model.map_parameters_)
    predicted = conditioned.predict(times=query)["b"]["heldout"]["brain"]
    assert predicted.valid.all()
    assert np.isfinite(predicted.values).all()
