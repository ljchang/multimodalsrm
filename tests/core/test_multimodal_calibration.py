import copy

import numpy as np
import pytest

from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries

from .test_multimodal_inference import fixture


@pytest.mark.parametrize("pooling", ["shared", "population", "neighborhood"])
def test_calibration_freezes_donors_and_original(pooling):
    model, data, t = fixture(pooling)
    original = copy.deepcopy(model)
    data["c"] = {"r": {"x": TimeSeries(np.cos(t / 3)[:, None], t)}}
    edges = {"c": {"a": 1.0}} if pooling == "neighborhood" else None
    calibrated = model.calibrate(data, affinity=edges)
    assert "c" not in model.loadings_ and "c" in calibrated.loadings_
    for s in ["a", "b"]:
        for m in ["x", "y"]:
            np.testing.assert_array_equal(calibrated.loadings_[s][m], original.loadings_[s][m])
            np.testing.assert_array_equal(model.loadings_[s][m], original.loadings_[s][m])
    changed = copy.deepcopy(data)
    changed["c"]["r"]["x"] = TimeSeries((100 * np.sin(t))[:, None], t)
    again = model.calibrate(changed, affinity=edges)
    for s in calibrated.calibration_reference_latents_:
        np.testing.assert_array_equal(
            calibrated.calibration_reference_latents_[s]["r"],
            again.calibration_reference_latents_[s]["r"],
        )
    calibrated.loadings_["a"]["x"][0, 0] += 1
    np.testing.assert_array_equal(model.loadings_["a"]["x"], original.loadings_["a"]["x"])
    if pooling == "neighborhood":
        assert calibrated.affinity_["a"]["b"] == model.affinity_["a"]["b"]


def test_calibration_requires_named_alignment_and_new_mapping():
    model, data, t = fixture()
    new = {"c": {"other": {"x": TimeSeries(np.cos(t)[:, None], t)}}}
    with pytest.raises(ValueError, match="donor|anchor|align"):
        model.calibrate(new)
    with pytest.raises(ValueError, match="run|align"):
        model.calibrate(new, reference="training")
    with pytest.raises(ValueError, match="new|existing"):
        model.calibrate(data)


def test_training_reference_and_missing_modality_zero_weight():
    t = np.arange(10.0)
    data = {
        "a": {
            "r": {
                "x": TimeSeries(np.sin(t)[:, None], t),
                "y": TimeSeries(np.cos(t)[:, None], t),
            }
        }
    }
    model = MultimodalSRM(
        features=1,
        latent_dt=1,
        responses={m: Response(Identity(), estimate=False) for m in ["x", "y"]},
        modality_weights={"x": 1.0, "y": 0.0},
        max_iter=30,
        tol=1e-3,
        random_state=1,
    ).fit(data)
    result = model.calibrate(data, reference="training", modality_weights={"x": 0.5, "y": 0.5})
    assert "y" in result.loadings_["a"] and "y" not in model.loadings_["a"]
    assert result.predict(data, targets={"a": ["y"]}, times=t)["a"]["r"]["y"].valid.all()


def test_training_reference_rejects_nonoverlap_and_existing_edges():
    model, data, t = fixture("neighborhood")
    outside = {"c": {"r": {"x": TimeSeries(np.sin(t)[:, None], t + 100)}}}
    with pytest.raises(ValueError, match="usable|support"):
        model.calibrate(outside, reference="training", affinity={"c": {"a": 1}})
    with pytest.raises(ValueError, match="frozen"):
        model.calibrate(
            {"c": data["a"]},
            reference="training",
            affinity={"a": {"b": 0}, "c": {"a": 1}},
        )
    with pytest.raises(ValueError, match="disconnected"):
        model.calibrate({"c": data["a"]}, reference="training")


def test_calibrated_new_run_does_not_reuse_calibration_latents():
    model, data, t = fixture()
    calibrated = model.calibrate({"c": data["a"]}, reference="training")
    new = {"c": {"eval": {"y": data["a"]["r"]["y"]}}}
    first = calibrated.predict(new, targets={"c": ["x"]}, times=t)["c"]["eval"]["x"]
    for runs in calibrated.calibration_reference_latents_.values():
        for value in runs.values():
            value[:] = 1e9
    second = calibrated.predict(new, targets={"c": ["x"]}, times=t)["c"]["eval"]["x"]
    np.testing.assert_array_equal(first.values, second.values)


def test_fixed_conditional_solve_matches_independent_dense_reference():
    from multimodalsrm.objective import _penalty, solve_latents

    model, data, t = fixture()
    frozen = {"a": {"r": np.sin(t)[:, None]}}
    z, diagnostics = solve_latents(
        [],
        {"r": t},
        ["a", "b"],
        {},
        1,
        "population",
        2.0,
        0.1,
        0.3,
        fixed_latents=frozen,
    )
    P = _penalty({"r": t}, ["a", "b"], "population", 2.0, 0.1, 0.3, None)["r"].toarray()
    expected = np.linalg.solve(P[10:, 10:], -P[10:, :10] @ frozen["a"]["r"])
    np.testing.assert_allclose(z["b"]["r"], expected)
    np.testing.assert_array_equal(z["a"]["r"], frozen["a"]["r"])
    assert diagnostics[0]["success"]


def test_frozen_population_reference_quadratic_stationarity():
    from multimodalsrm.objective import complete_objective, solve_latents

    t = np.arange(6.0)
    center = {"r": np.sin(t)[:, None]}
    fixed = {"a": {"r": 2 * center["r"]}}
    z, _ = solve_latents(
        [],
        {"r": t},
        ["a", "b"],
        {},
        1,
        "population",
        2.0,
        0.1,
        0.0,
        fixed_latents=fixed,
        population_reference=center,
    )
    np.testing.assert_allclose(z["b"]["r"], center["r"] * 2 / 2.1)

    # Objective finite differences check the same anchored quadratic as the solve.
    def obj(value):
        return complete_objective(
            [],
            {"r": t},
            value,
            {"b": {"x": np.zeros((1, 1))}},
            "population",
            2.0,
            0.1,
            0.0,
            0.01,
            population_reference=center,
        )

    for j in range(len(t)):
        plus = copy.deepcopy(z)
        minus = copy.deepcopy(z)
        plus["b"]["r"][j, 0] += 1e-5
        minus["b"]["r"][j, 0] -= 1e-5
        assert abs((obj(plus) - obj(minus)) / 2e-5) < 1e-8


@pytest.mark.parametrize("pooling", ["shared", "partial", "none"])
def test_kernel_calibration_preserves_existing_and_group(pooling):
    from multimodalsrm import Gaussian

    t = np.arange(16.0)
    data = {
        s: {
            "r": {
                "x": TimeSeries(np.sin(t / 3)[:, None], t),
                "y": TimeSeries(np.cos(t / 3)[:, None], t),
            }
        }
        for s in ["a", "b"]
    }
    model = MultimodalSRM(
        features=1,
        latent_dt=0.5,
        responses={
            "x": Response(
                Gaussian(width=0.1),
                pooling=pooling,
                fixed={"lag": 0},
                bounds={"width": (0.05, 0.2)},
            ),
            "y": Response(Identity(), estimate=False),
        },
        max_iter=10,
        tol=0.01,
        random_state=1,
    ).fit(data)
    old = copy.deepcopy(model.subject_kernels_)
    group = copy.deepcopy(model.group_kernels_)
    result = model.calibrate({"c": data["a"]}, reference="training")
    for s in ["a", "b"]:
        assert result.subject_kernels_[s]["x"].parameters == old[s]["x"].parameters
    assert {m: k.parameters for m, k in result.group_kernels_.items()} == {
        m: k.parameters for m, k in group.items()
    }
    if pooling == "shared":
        assert result.subject_kernels_["c"]["x"].parameters == group["x"].parameters
    if pooling == "partial":
        assert 0.05 <= result.subject_kernels_["c"]["x"].parameters["width"] <= 0.2
    assert np.all(np.diff(result.calibration_diagnostics_["objective_history"]) <= 1e-9)


@pytest.mark.parametrize("pooling", ["population", "shared", "neighborhood"])
def test_repeated_training_reference_calibration(pooling):
    model, data, t = fixture(pooling)
    first = model.calibrate(
        {"c": data["a"]},
        reference="training",
        affinity={"c": {"a": 1}} if pooling == "neighborhood" else None,
    )
    second = first.calibrate(
        {"d": data["a"]},
        reference="training",
        affinity={"d": {"c": 1}} if pooling == "neighborhood" else None,
    )
    assert "d" in second.loadings_ and "d" not in first.loadings_
    np.testing.assert_array_equal(first.loadings_["c"]["x"], second.loadings_["c"]["x"])


def test_unknown_calibration_modality_is_not_silently_dropped():
    model, data, t = fixture()
    new = {"c": {"r": dict(data["a"]["r"], typo=data["a"]["r"]["x"])}}
    with pytest.raises(ValueError, match="configured responses"):
        model.calibrate(new, reference="training")
