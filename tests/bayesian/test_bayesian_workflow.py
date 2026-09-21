"""Catch leakage, count errors and lossy uncertainty exports in the MVP path."""

import importlib
import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import GaussianMixtureSeries


def workflow():
    name = "multimodalsrm.bayesian.workflow"
    assert importlib.util.find_spec(name) is not None, "MVP workflow helpers missing"
    return importlib.import_module(name)


def test_preprocessing_uses_training_masks_and_excludes_poisoned_targets():
    w = workflow()
    train = {
        "a": {
            "train": {
                "ref": TimeSeries([[1.0], [3.0], [999.0]], [0, 1, 2], [[True], [True], [False]]),
                "brain": TimeSeries([[10.0], [14.0], [12.0]], [0, 1, 2]),
            }
        }
    }
    scaler = w.TrainingStandardizer.fit(train)
    assert_allclose(scaler.statistics["a", "ref"]["mean"], [2])
    assert_allclose(scaler.statistics["a", "ref"]["scale"], [1])
    donors = {"a": {"test": {"ref": TimeSeries([[11.0], [13.0]], [7, 8]), "brain": object()}}}
    transformed = scaler.transform(donors, targets={"a": ["brain"]})
    assert "brain" not in transformed["a"]["test"]
    assert_allclose(transformed["a"]["test"]["ref"].values[:, 0], [9, 11])
    assert_allclose(transformed["a"]["test"]["ref"].times, [7, 8])
    result = GaussianMixtureSeries(
        np.array([[[0.0]], [[2.0]]]), np.ones((2, 1, 1)), [7], [True], {}
    )
    original = scaler.inverse_result(result, "a", "brain")
    assert_allclose(original.values, [[12 + np.sqrt(8 / 3)]])
    assert_allclose(original.variance, [[16 / 3]])


def test_scaler_rejects_constant_or_unobserved_features():
    w = workflow()
    for values, mask in [
        ([[1.0, 1.0], [2.0, 1.0]], None),
        ([[1.0, 2.0], [2.0, 3.0]], [[True, False], [True, False]]),
    ]:
        with pytest.raises(ValueError, match="feature"):
            w.TrainingStandardizer.fit({"a": {"ref": TimeSeries(values, [0, 1], mask)}})


def test_capacity_counts_feature_scalars_support_and_joint_donor_exclusion():
    w = workflow()
    t = np.arange(21.0)
    mask = np.ones((21, 2), bool)
    mask[10, 1] = False
    data = {
        "a": {
            "train": {
                "ref": TimeSeries(t[:, None], t),
                "brain": TimeSeries(np.column_stack([t, t]), t, mask),
            }
        }
    }
    responses = {
        "ref": Response(Identity(), estimate=False, pooling="shared"),
        "brain": Response(Gaussian(width=1.0, lag=2.0), estimate=False, pooling="shared"),
    }
    # Brain support [-4, 8] retains t=8..16 inclusive: 18 scalars minus one mask.
    report = w.capacity_report(data, responses, max_observations=37)
    assert report["runs"]["train"]["eligible_scalars"] == 38
    assert report["runs"]["train"]["observed_scalars"] == 62
    assert report["runs"]["train"]["one_covariance_bytes"] == 38 * 38 * 8
    assert not report["within_limit"]
    data["a"]["train"]["brain"] = object()
    excluded = w.capacity_report(data, responses, targets={"a": ["brain"]})
    assert excluded["runs"]["train"]["eligible_scalars"] == 21


def test_capacity_matches_backend_eligible_systems():
    w = workflow()
    from .test_bayesian_problem import problem_fixture

    problem, adapter, data = problem_fixture(gaussian=True, two_runs=True)
    actual = w.capacity_report(data, adapter.responses_)
    assert {r: d["eligible_scalars"] for r, d in actual["runs"].items()} == {
        r: len(system.times) for r, system in problem.systems.items()
    }


def test_archive_preserves_mixture_quantiles_masks_and_rejects_corruption(tmp_path):
    w = workflow()
    result = GaussianMixtureSeries(
        np.array([[[-2.0], [0.0]], [[3.0], [0.0]]]),
        np.array([[[0.25], [1.0]], [[1.0], [1.0]]]),
        [1.0, 2.0],
        [True, False],
        {"uncertainty": "parameter_posterior_mixture"},
    )
    path = tmp_path / "results"
    w.save_results(path, {("a", "test", "brain"): result}, metadata={"seed": 12})
    loaded, metadata = w.load_results(path)
    restored = loaded["a", "test", "brain"]
    assert metadata == {"seed": 12}
    assert_allclose(restored.interval(0.95), result.interval(0.95), equal_nan=True)
    assert_allclose(
        restored.between_parameter_variance,
        result.between_parameter_variance,
        equal_nan=True,
    )
    assert not restored.component_means.flags.writeable
    with pytest.raises(FileExistsError):
        w.save_results(path, {}, metadata={})
    with (path / "arrays.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        w.load_results(path)
