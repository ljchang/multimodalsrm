"""The default fits one response and keeps its complete graph inspectable."""

import copy
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import MultimodalSRM

from .test_multimodal_fit import fixture


def fit_default(data=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return MultimodalSRM(features=2, latent_dt=1, max_iter=3, random_state=4, **kwargs).fit(
            fixture() if data is None else data
        )


def test_default_shares_exactly_and_predicts_from_donors_without_pooling_penalty():
    # A finite complete-graph penalty must not replace the exact constraint.
    model = fit_default(latent_strength=0)
    np.testing.assert_array_equal(
        model.training_latents_["s1"]["run-01"].values,
        model.training_latents_["s2"]["run-01"].values,
    )
    assert model.loadings_["s1"]["a"].shape == (1, 2)
    assert not np.allclose(model.loadings_["s1"]["a"], model.loadings_["s2"]["a"])
    expected = model.predict(fixture(), targets={"s1": ["a"]}, source="across")
    changed = copy.deepcopy(fixture())
    changed["s1"] = {}  # Donor-only inference must not need target observations.
    actual = model.predict(changed, targets={"s1": ["a"]}, source="across")
    result = actual["s1"]["run-01"]["a"]
    assert result.valid.any()
    np.testing.assert_array_equal(result.values, expected["s1"]["run-01"]["a"].values)


@pytest.mark.parametrize(
    "subjects, expected",
    [
        (["s1", "s2"], {"s1": {"s2": 1.0}, "s2": {"s1": 1.0}}),
        (["s1"], {"s1": {}}),
    ],
)
def test_shared_graph_representation_includes_single_participant(subjects, expected):
    data = {s: fixture()[s] for s in subjects}
    model = fit_default(data, latent_pooling="shared")
    assert model.affinity_ == expected


def test_shared_calibration_extends_graph_without_mutating_original():
    model = fit_default(latent_pooling="shared")
    calibrated = model.calibrate({"new": fixture()["s1"]}, reference="training")
    assert calibrated.affinity_ == {
        "s1": {"s2": 1.0, "new": 1.0},
        "s2": {"s1": 1.0, "new": 1.0},
        "new": {"s1": 1.0, "s2": 1.0},
    }
    assert model.affinity_ == {"s1": {"s2": 1.0}, "s2": {"s1": 1.0}}
    np.testing.assert_array_equal(model.loadings_["s1"]["a"], calibrated.loadings_["s1"]["a"])
    result = calibrated.predict(fixture(), targets={"new": ["a"]}, source="across")["new"][
        "run-01"
    ]["a"]
    assert result.valid.any()


def test_refit_replaces_complete_graph_and_explicit_population_remains_available():
    model = fit_default()
    fixed = {"s1": {"s2": 0.25}, "s2": {"s1": 0.25}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.set_params(latent_pooling="neighborhood").fit(fixture(), affinity=fixed)
        assert model.affinity_ == fixed
        model.set_params(latent_pooling="shared").fit({"s1": fixture()["s1"]})
        assert model.affinity_ == {"s1": {}}
        model.set_params(latent_pooling="population", latent_strength=0).fit(fixture())
    assert model.affinity_ is None
    assert not np.allclose(
        model.training_latents_["s1"]["run-01"].values,
        model.training_latents_["s2"]["run-01"].values,
    )
