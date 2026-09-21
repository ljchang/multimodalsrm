"""New participant maps condition on a frozen group, never on held-out data."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    Prior,
    SearchConfig,
)

from .test_bayesian_persistence import multiple  # noqa: F401 - shared native-rate fixture


def make_group(anchor=("a", "ref", 0)):
    # Coherent shared data: the frozen-conditioning stress fixture deliberately
    # used unrelated sine/cosine modalities and cannot assert mapping recovery.
    times = np.arange(0.0, 21.0, 0.5)
    rng = np.random.default_rng(871)
    distance = np.sqrt(3) * np.abs(times[:, None] - times[None, :]) / 3
    latent = np.linalg.cholesky(
        (1 + distance) * np.exp(-distance) + 1e-9 * np.eye(len(times))
    ) @ rng.normal(size=len(times))
    data = {
        s: {
            "train": {
                m: TimeSeries(
                    (w * latent + rng.normal(scale=0.03, size=len(times)))[:, None],
                    times,
                )
            }
        }
        for s, m, w in [("a", "ref", 1.0), ("b", "signal", -0.8)]
    }
    model = BayesianMultimodalSRM(
        priors=BayesianPriors(
            noise=Prior.lognormal(np.log(0.001), 0.7), loading_sd=1.5, offset_sd=0.8
        ),
        anchor=anchor,
        inference="map",
        random_state=871,
        search=SearchConfig(starts=1, maxiter=300),
    ).fit(data)
    return model, data


@pytest.fixture(scope="module")
def group():
    return make_group()


def test_optional_anchor_calibration_signed_mapping_archive_and_transform(tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data = make_group(anchor=None)
    new = {"0-new": newcomer(data)["new-person"]}
    result = calibrate(model, new)
    assert result.problem_.joint.anchor == model.problem_.anchor
    index = result.problem_.indices[("loading", "0-new", "ref", 0)]
    assert result.problem_.bounds[index] == (-np.inf, np.inf)
    assert result.map_parameters_[index] < 0
    test = {"0-new": {"test": new["0-new"]["train"]}}
    times = [4.0, 8.0, 12.0]
    before = result.transform(test, times=times)["0-new"]["test"]
    expected = model.transform({"a": {"test": data["a"]["train"]}}, times=times)["a"]["test"]
    assert_allclose(before.values, expected.values, atol=0.15)
    assert before.metadata["reference_convention"] == model.configuration_["reference_convention"]
    save_model(tmp_path / "optional_calibration", result)
    restored, _ = load_model(tmp_path / "optional_calibration")
    after = restored.transform(test, times=times)["0-new"]["test"]
    assert_array_equal(after.values, before.values)
    assert after.metadata["reference_convention"] == before.metadata["reference_convention"]


def newcomer(data):
    ts = data["a"]["train"]["ref"]
    return {
        "new-person": {"train": {"ref": TimeSeries(-1.3 * ts.values + 0.25, ts.times, ts.mask)}}
    }


def calibrate(model, data):
    assert callable(getattr(model, "calibrate", None)), "calibrate is missing"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return model.calibrate(data, search=SearchConfig(starts=2, maxiter=150), random_state=123)


def test_calibration_freezes_group_and_learns_signed_new_mapping(group):
    model, data = group
    params = model.map_parameters_.copy()
    config = copy.deepcopy(model.configuration_)
    fit = copy.deepcopy(model.map_diagnostics_)
    result = calibrate(model, newcomer(data))
    assert_array_equal(model.map_parameters_, params)
    assert model.configuration_ == config
    assert model.map_diagnostics_ == fit
    names = result.parameter_names_
    assert all(n[0] in ("loading", "offset", "noise") and n[1] == "new-person" for n in names)
    mapped = dict(zip(names, result.map_parameters_))
    old = dict(zip(model.parameter_names_, params))
    assert mapped["loading", "new-person", "ref", 0] < 0
    assert_allclose(
        mapped["loading", "new-person", "ref", 0],
        -1.3 * old["loading", "a", "ref", 0],
        rtol=0.15,
    )
    test = {"new-person": {"test": newcomer(data)["new-person"]["train"]}}
    output = result.transform(test, times=[4.0, 8.0, 12.0])["new-person"]["test"]
    expected = model.transform({"a": {"test": data["a"]["train"]}}, times=[4.0, 8.0, 12.0])["a"][
        "test"
    ]
    assert_allclose(output.values, expected.values, atol=0.15)
    assert output.metadata["parameter_source"] == "group_MAP_plus_participant_calibration_MAP"
    assert output.metadata["parameter_conditioning"] == output.metadata["parameter_source"]
    assert output.metadata["calibration_runs"] == ["train"]


def test_instantaneous_calibration_uses_new_map_without_gp_conditioning(group, monkeypatch):
    from multimodalsrm.bayesian.problem import BayesianProblem

    model, data = group
    mapped = calibrate(model, newcomer(data))
    test = {"new-person": {"test": newcomer(data)["new-person"]["train"]}}

    def forbidden(*args, **kwargs):
        raise AssertionError("instantaneous calibration must not evaluate a GP")

    monkeypatch.setattr(BayesianProblem, "nll", forbidden)
    out = mapped.transform(test, times=[4.0, 8.0], readout="instantaneous")["new-person"]["test"]
    p = dict(zip(mapped.parameter_names_, mapped.map_parameters_))
    w, b, noise = (
        p["loading", "new-person", "ref", 0],
        p["offset", "new-person", "ref", 0],
        p["noise", "new-person", "ref"],
    )
    ts = test["new-person"]["test"]["ref"]
    y = ts.values[np.isin(ts.times, out.times), 0]
    assert_allclose(out.values[:, 0], w * (y - b) / (noise + w * w))
    assert_allclose(out.variance[:, 0], noise / (noise + w * w))
    assert out.metadata["alignment_readout"] == "instantaneous"
    assert out.metadata["parameter_source"] == "group_MAP_plus_participant_calibration_MAP"


@pytest.mark.parametrize("change", ["known_subject", "new_run", "outside", "new_modality"])
def test_calibration_rejects_unmatched_reference_inputs(group, change):
    model, data = group
    supplied = newcomer(data)
    mods = supplied["new-person"]["train"]
    if change == "known_subject":
        supplied = {"a": supplied["new-person"]}
    elif change == "new_run":
        supplied = {"new-person": {"unknown": mods}}
    elif change == "outside":
        ts = mods["ref"]
        mods["ref"] = TimeSeries(ts.values, ts.times + 50, ts.mask)
    else:
        mods["untrained"] = mods.pop("ref")
    assert callable(getattr(model, "calibrate", None)), "calibrate is missing"
    with pytest.raises(ValueError):
        model.calibrate(supplied)


def test_conditional_objective_matches_schur_gaussian_and_physical_gradient(group):
    model, data = group
    result = calibrate(model, newcomer(data))
    problem = result.problem_
    point = result.map_parameters_.copy()
    # Check derivatives away from a tiny-noise optimum, where subtracting two
    # nearly identical floating-point objectives is a poor gradient reference.
    point[:2] *= 0.9
    point[-1] = 0.02
    joint = problem.joint
    full = np.asarray(problem.expand(point))
    C = np.asarray(joint.covariance(full, "train"))
    offsets = np.asarray(joint.arrays(full)[1])
    y = joint.systems["train"].values - offsets[joint._packed["train"][0]]
    new = np.array([k[0] == "new-person" for k in joint.systems["train"].keys])
    aa, bb, ab = C[~new][:, ~new], C[new][:, new], C[new][:, ~new]
    mean = ab @ np.linalg.solve(aa, y[~new])
    covariance = bb - ab @ np.linalg.solve(aa, ab.T)
    residual = y[new] - mean
    expected = 0.5 * (
        residual @ np.linalg.solve(covariance, residual)
        + 2 * np.log(np.diag(np.linalg.cholesky(covariance))).sum()
        + new.sum() * np.log(2 * np.pi)
    )
    assert_allclose(float(problem.nll(point)), expected, atol=1e-8)
    from scipy.stats import lognorm, norm

    prior = model.priors
    log_prior = norm.logpdf(point[0], scale=prior.loading_sd) + norm.logpdf(
        point[1], scale=prior.offset_sd
    )
    log_prior += lognorm.logpdf(point[2], s=prior.noise.scale, scale=np.exp(prior.noise.loc))
    assert_allclose(float(problem.objective(point)), expected - log_prior, atol=1e-8)
    numerical = []
    for i in range(len(point)):
        step = 1e-5 * max(abs(point[i]), 0.01)
        a, b = point.copy(), point.copy()
        a[i] += step
        b[i] -= step
        numerical.append((float(problem.objective(a)) - float(problem.objective(b))) / (2 * step))
    assert_allclose(problem.value_gradient(point)[1], numerical, rtol=1e-4, atol=1e-4)


def test_alignment_accepts_calibrated_and_group_only_with_same_reference(group):
    from dataclasses import replace

    from multimodalsrm.alignment import (
        temporal_isc,
        time_segment_matching,
    )

    model, data = group
    result = calibrate(model, newcomer(data))
    query = np.arange(4.0, 16.0)
    new = result.transform(
        {"new-person": {"test": newcomer(data)["new-person"]["train"]}}, times=query
    )["new-person"]["test"]
    old = model.transform({"a": {"test": data["a"]["train"]}}, times=query)["a"]["test"]
    series = {"a": old, "new-person": new}
    assert temporal_isc(series)["n_subjects"] == 2
    assert time_segment_matching(series, window_size=3)["n_subjects"] == 2
    series["new-person"] = replace(
        new, metadata={**new.metadata, "training_reference_id": "different"}
    )
    with pytest.raises(ValueError, match="reference"):
        temporal_isc(series)


def test_calibration_archive_reuses_mapping_and_new_only_scaler(group, tmp_path, monkeypatch):
    from multimodalsrm.bayesian import calibration, fitting
    from multimodalsrm.bayesian import model as model_module
    from multimodalsrm.bayesian.workflow import (
        TrainingStandardizer,
        load_model,
        save_model,
    )

    model, data = group
    raw = newcomer(data)
    scaler = TrainingStandardizer.fit(raw)
    result = calibrate(model, scaler.transform(raw))
    test = {"new-person": {"test": raw["new-person"]["train"]}}
    expected = result.transform(scaler.transform(test), times=[4.0, 8.0, 12.0])["new-person"][
        "test"
    ]
    save_model(tmp_path / "calibrated", result, standardizer=scaler)

    def forbidden(*args, **kwargs):
        raise AssertionError("reloading must not optimize or fit preprocessing")

    for module in (fitting, model_module, calibration):
        monkeypatch.setattr(module, "search", forbidden)
    monkeypatch.setattr(TrainingStandardizer, "fit", forbidden)
    restored, restored_scaler = load_model(tmp_path / "calibrated")
    actual = restored.transform(restored_scaler.transform(test), times=[4.0, 8.0, 12.0])[
        "new-person"
    ]["test"]
    for field in ("component_means", "component_variances", "valid", "times"):
        assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.metadata == expected.metadata
    assert restored.configuration_ == result.configuration_


@pytest.mark.parametrize(
    "missing",
    [
        {"n_jobs"},
        {"n_jobs", "polish_max_parameters"},
        {"conditioning", "n_jobs", "polish_max_parameters"},
    ],
)
def test_legacy_calibration_search_defaults_preserve_provenance(
    group, tmp_path, monkeypatch, missing
):
    from multimodalsrm.bayesian import _archive, calibration
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data = group
    result = calibrate(model, newcomer(data))
    test = {"new-person": {"test": newcomer(data)["new-person"]["train"]}}
    expected = result.transform(test, times=[4.0, 8.0, 12.0])["new-person"]["test"]
    save_model(tmp_path / "current", result)
    state = _archive.read(tmp_path / "current")
    for field in missing:
        del state["fit"]["configuration"]["search"][field]
    for configuration in (
        state["reference"]["fit"]["configuration"],
        state["fit"]["configuration"]["reference_configuration"],
    ):
        del configuration["sampler"]["mass_matrix"]
        del configuration["sampler"]["max_dense_parameters"]
    from multimodalsrm.bayesian.identity import training_reference_id

    legacy_group = copy.deepcopy(model)
    legacy_group.training_fit_["configuration"] = copy.deepcopy(
        state["reference"]["fit"]["configuration"]
    )
    state["fit"]["configuration"]["reference_id"] = training_reference_id(legacy_group)
    _archive.write(tmp_path / "legacy", state)

    def forbidden(*args, **kwargs):
        raise AssertionError("archive migration must not refit the model")

    monkeypatch.setattr(calibration, "search", forbidden)
    restored, _ = load_model(tmp_path / "legacy")
    actual = restored.transform(test, times=[4.0, 8.0, 12.0])["new-person"]["test"]
    assert_array_equal(actual.values, expected.values)
    assert_array_equal(actual.component_variances, expected.component_variances)
    assert restored.configuration_ == state["fit"]["configuration"]
    assert "mass_matrix" not in restored._group.configuration_["sampler"]
    assert "max_dense_parameters" not in restored._group.configuration_["sampler"]
    assert restored.search_config_.n_jobs == 1
    save_model(tmp_path / "resaved", restored)
    again, _ = load_model(tmp_path / "resaved")
    assert again.configuration_ == restored.configuration_


def test_multifactor_native_partial_calibration_keeps_filters_and_coordinates(request):
    from multimodalsrm.bayesian.calibration import prepare
    from multimodalsrm.bayesian.persistence import _prepare

    model, scaler, test = request.getfixturevalue("multiple")
    calibration_data = {"s4": {"train": {}}}
    for m, ts in model.training_data_["s2"]["train"].items():
        rows = (ts.times >= 4) & (ts.times <= 12)
        calibration_data["s4"]["train"][m] = TimeSeries(
            ts.values[rows], ts.times[rows], ts.mask[rows]
        )
    result = calibrate(model, calibration_data)
    full = np.asarray(result.problem_.expand(result.map_parameters_))
    for name, value in zip(model.parameter_names_, model.map_parameters_):
        assert full[result.problem_.joint.indices[name]] == value
    assert (
        result.configuration_["reference_configuration"]["factor_orientation"]
        == model.factor_orientation_
    )
    scaled_test = scaler.transform(test)
    new_test = {"s4": scaled_test["s2"]}
    transformed = result.transform(new_test, times=np.arange(4.0, 12.0, 0.5))["s4"]["test"]
    assert transformed.values.shape == (16, 2)
    assert np.isfinite(transformed.values[transformed.valid]).all()
    # Same physical parameters in a separately built dense joint problem.
    dense = copy.deepcopy(model)
    dense.linear_algebra = "dense"
    _, joint = _prepare(dense, {**model.training_data_, **calibration_data})
    from multimodalsrm.bayesian.calibration_problem import (
        CalibrationProblem,
    )

    # Reference nll uses the original grouped group fit, the same conditional constant.
    dense_problem = CalibrationProblem(joint, model, "s4")
    point = result.map_parameters_
    assert_allclose(
        dense_problem.value_gradient(point)[0],
        result.problem_.value_gradient(point)[0],
        atol=1e-7,
    )
    assert_allclose(
        dense_problem.value_gradient(point)[1],
        result.problem_.value_gradient(point)[1],
        atol=1e-6,
    )
    # Masked endpoints can lie outside a shared run; observed support cannot.
    ts = calibration_data["s4"]["train"]["brain"]
    times = np.r_[-5.0, ts.times, 25.0]
    values = np.vstack([np.zeros((1, 3)), ts.values, np.zeros((1, 3))])
    mask = np.vstack([np.zeros((1, 3), bool), ts.mask, np.zeros((1, 3), bool)])
    calibration_data["s4"]["train"]["brain"] = TimeSeries(values, times, mask)
    _, adapter, _ = prepare(model, calibration_data)
    assert adapter.domains_ == model.adapter_.domains_


@pytest.mark.parametrize(
    "change", ["posterior", "conditioned", "baseline", "empty_feature", "two_subjects"]
)
def test_calibration_rejects_unsupported_models_and_empty_feature_support(group, change):
    model, data = group
    model = copy.deepcopy(model)
    supplied = newcomer(data)
    if change == "posterior":
        model.inference = "posterior"
    elif change == "conditioned":
        model = model.condition(
            {"a": {"test": data["a"]["train"]}},
            targets={"b": ["signal"]},
            mode="frozen",
        )
    elif change == "baseline":
        model.run_baseline_sd = 0.5
    elif change == "empty_feature":
        ts = supplied["new-person"]["train"]["ref"]
        supplied["new-person"]["train"]["ref"] = TimeSeries(
            np.column_stack([ts.values[:, 0], ts.values[:, 0]]),
            ts.times,
            np.column_stack([ts.mask[:, 0], np.zeros(len(ts.times), bool)]),
        )
    else:
        supplied["second-person"] = supplied["new-person"]
    with pytest.raises(ValueError):
        model.calibrate(supplied)


@pytest.mark.parametrize(
    "damage", ["order", "reference", "flag", "noise", "group_scaler", "seed", "data"]
)
def test_calibration_archive_rejects_inconsistent_state(group, tmp_path, damage):
    from multimodalsrm.bayesian import _archive
    from multimodalsrm.bayesian.workflow import (
        TrainingStandardizer,
        load_model,
        save_model,
    )

    model, data = group
    result = calibrate(model, newcomer(data))
    save_model(tmp_path / "good", result)
    state = _archive.read(tmp_path / "good")
    if damage == "order":
        state["parameter_names"] = state["parameter_names"][::-1]
    elif damage == "reference":
        state["fit"]["configuration"]["reference_id"] = "other"
    elif damage == "flag":
        state["fit"]["map_diagnostics"]["meets_gradient_tolerance"] = "false"
    elif damage == "noise":
        point = state["fit"]["map_parameters"].copy()
        point[-1] = 0.0
        state["fit"]["map_parameters"] = point
    elif damage == "group_scaler":
        state["standardizer"] = TrainingStandardizer.fit(data)
    elif damage == "seed":
        state["seed"] += 1
    else:
        ts = result.calibration_data_["new-person"]["train"]["ref"]
        result.calibration_data_["new-person"]["train"]["ref"] = TimeSeries(ts.values + 2, ts.times)
        with pytest.raises(ValueError, match="observations"):
            save_model(tmp_path / "changed", result)
        return
    _archive.write(tmp_path / "bad", state, version=2)
    with pytest.raises(ValueError):
        load_model(tmp_path / "bad")


@pytest.mark.parametrize("change", ["fixed_map", "active", "group_nll", "kernel"])
def test_mutated_calibration_contract_cannot_transform_or_save(group, tmp_path, change):
    from multimodalsrm.bayesian.workflow import save_model

    model, data = group
    result = calibrate(model, newcomer(data))
    if change == "fixed_map":
        point = result.problem_.reference.copy()
        point[0] *= 0.5
        result.problem_.reference = point
    elif change == "active":
        result.problem_.active = result.problem_.active[::-1]
    elif change == "group_nll":
        result.problem_.group_nll += 2
    else:
        result.problem_.joint.length_scale *= 2
    with pytest.raises(ValueError):
        save_model(tmp_path / "bad", result)
    with pytest.raises(ValueError):
        result.transform(
            {"new-person": {"test": newcomer(data)["new-person"]["train"]}},
            times=[4.0, 8.0, 12.0],
        )


@pytest.mark.parametrize("invalid", [False, 0, {}, []])
def test_calibration_does_not_silently_replace_invalid_search(group, invalid):
    model, data = group
    with pytest.raises(ValueError):
        model.calibrate(newcomer(data), search=invalid)


def test_failed_calibration_gate_survives_archive_without_inheriting_group_pass(group, tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data = group
    with pytest.warns(UserWarning, match="calibration physical gradient"):
        result = model.calibrate(
            newcomer(data), search=SearchConfig(starts=1, maxiter=1), random_state=9
        )
    assert result.map_diagnostics_["meets_gradient_tolerance"] is False
    save_model(tmp_path / "limited", result)
    restored, _ = load_model(tmp_path / "limited")
    assert restored.map_diagnostics_["meets_gradient_tolerance"] is False
    assert (
        restored.configuration_["original_group_meets_gradient_tolerance"]
        == model.map_diagnostics_["meets_gradient_tolerance"]
    )


def test_calibration_reload_and_transform_in_fresh_process_without_optimization(group, tmp_path):
    import os
    import subprocess
    import sys

    from multimodalsrm.bayesian.workflow import save_model

    model, data = group
    result = calibrate(model, newcomer(data))
    expected = result.transform(
        {"new-person": {"test": newcomer(data)["new-person"]["train"]}},
        times=[4.0, 8.0, 12.0],
    )["new-person"]["test"]
    save_model(tmp_path / "model", result)
    script = """
import sys
import numpy as np
from multimodalsrm.bayesian import calibration, model, fitting
from multimodalsrm.bayesian.workflow import load_model, TrainingStandardizer
def forbidden(*a, **k):
    raise AssertionError("unexpected optimizer, sampler or preprocessing fit")
calibration.search = model.search = fitting.search = model.sample = fitting.sample = forbidden
TrainingStandardizer.fit = forbidden
restored, scaler = load_model(sys.argv[1])
data = {restored.subject_: {"test": restored.calibration_data_[restored.subject_]["train"]}}
result = restored.transform(data, times=[4.,8.,12.])[restored.subject_]["test"]
np.savez(sys.argv[2], **{k: getattr(result,k) for k in ("component_means","component_variances","valid","times")})
"""
    env = dict(os.environ)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "model"),
            str(tmp_path / "result.npz"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert done.returncode == 0, done.stderr
    with np.load(tmp_path / "result.npz", allow_pickle=False) as actual:
        for field in actual.files:
            assert_array_equal(actual[field], getattr(expected, field))
