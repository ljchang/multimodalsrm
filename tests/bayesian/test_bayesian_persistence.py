"""Fitted-model archives must preserve inference, coordinates and failure flags."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from multimodalsrm import Gaussian, Identity, Response
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    Prior,
    SamplerConfig,
    SearchConfig,
    workflow,
)

from .test_bayesian_frozen import donors, fitted


def archive_api():
    assert callable(getattr(workflow, "save_model", None)), "save_model is missing"
    assert callable(getattr(workflow, "load_model", None)), "load_model is missing"
    return workflow.save_model, workflow.load_model


def assert_result_equal(a, b):
    for field in ("component_means", "component_variances", "times", "valid"):
        assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.metadata == b.metadata


def forbid_inference_fitting(monkeypatch):
    from multimodalsrm.bayesian import fitting, model

    def forbidden(*args, **kwargs):
        raise AssertionError("archive reuse must not optimize, sample or fit scaling")

    for module in (model, fitting):
        monkeypatch.setattr(module, "search", forbidden)
        monkeypatch.setattr(module, "sample", forbidden)
    monkeypatch.setattr(workflow.TrainingStandardizer, "fit", forbidden)


@pytest.fixture(scope="module")
def scalar():
    return fitted("grouped", {"ref": 0.6, "signal": 0.8})


@pytest.fixture(scope="module")
def multiple():
    from ..reference.multifactor_fixture import synthetic

    train, test, _ = synthetic(seed=872, duration=16, aux_hz=2)
    scaler = workflow.TrainingStandardizer.fit(train)
    model = BayesianMultimodalSRM(
        features=2,
        anchor=("s1", "brain", 0),
        factor_anchors=(("s1", "brain", 0), ("s1", "brain", 1)),
        priors=BayesianPriors(
            noise=Prior.lognormal(-2.0, 1.0),
            loading_sd=1.5,
            offset_sd=0.5,
            filters={
                "aux": {
                    "width": Prior.lognormal(np.log(0.35), 0.4),
                    "lag": Prior.normal(0.5, 0.5),
                }
            },
        ),
        responses={
            "brain": Response(Identity(), estimate=False, pooling="shared"),
            "aux": Response(
                Gaussian(0.35, 0.5),
                pooling="shared",
                bounds={"width": (0.2, 0.6), "lag": (0.0, 1.0)},
            ),
        },
        length_scale=1.5,
        inference="map",
        random_state=872,
        linear_algebra="grouped",
        max_observations=1000,
        search=SearchConfig(starts=1, maxiter=80),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model.fit(scaler.transform(train))
    return model, scaler, test


def test_scalar_archive_preserves_frozen_prediction_and_training_latent(
    scalar, tmp_path, monkeypatch
):
    save, load = archive_api()
    model, train = scalar
    test = donors(train)
    query = [4.0, 8.0, 12.0]
    expected = model.condition(test, targets={"b": ["signal"]}, mode="frozen").predict(times=query)
    latent = model.infer_latent(times=query)
    save(tmp_path / "model", model)
    forbid_inference_fitting(monkeypatch)
    restored, scaler = load(tmp_path / "model")
    from ..core.test_observation_preparation import assert_preparation_equal

    assert_preparation_equal(restored.adapter_, model.adapter_)
    # Target payloads must be excluded before normalization after archive reuse.
    test["b"]["new"]["signal"] = object()
    assert scaler is None
    assert restored.configuration_ == model.configuration_
    assert restored.map_diagnostics_ == model.map_diagnostics_
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    actual = restored.condition(test, targets={"b": ["signal"]}, mode="frozen").predict(times=query)
    assert_result_equal(actual["b"]["new"]["signal"], expected["b"]["new"]["signal"])
    assert_result_equal(restored.infer_latent(times=query)["train"], latent["train"])


def test_multiple_learned_filters_scaler_and_native_data_round_trip(
    multiple, tmp_path, monkeypatch
):
    save, load = archive_api()
    model, scaler, test = multiple
    query = np.arange(4.0, 12.0, 0.5)
    expected = model.transform(scaler.transform(test), times=query)
    save(tmp_path / "group", model, standardizer=scaler)
    forbid_inference_fitting(monkeypatch)
    restored, restored_scaler = load(tmp_path / "group")
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    assert not restored.map_parameters_.flags.writeable
    assert restored.parameter_names_ == model.parameter_names_
    assert restored.configuration_ == model.configuration_
    assert restored.factor_orientation_ == model.factor_orientation_
    assert restored.map_diagnostics_ == model.map_diagnostics_
    assert restored.restart_diagnostics_ == model.restart_diagnostics_
    for key, stats in scaler.statistics.items():
        for name in ("mean", "scale"):
            assert_array_equal(restored_scaler.statistics[key][name], stats[name])
            assert not np.shares_memory(restored_scaler.statistics[key][name], stats[name])
    for s, runs in model.training_data_.items():
        for r, mods in runs.items():
            assert set(restored.training_data_[s][r]) == set(mods)
            for m, ts in mods.items():
                for name in ("values", "times", "mask"):
                    assert_array_equal(
                        getattr(restored.training_data_[s][r][m], name),
                        getattr(ts, name),
                    )
    actual = restored.transform(restored_scaler.transform(test), times=query)
    for s in test:
        assert_result_equal(actual[s]["test"], expected[s]["test"])


@pytest.mark.parametrize("angle", [1e-13, 0.01])
def test_reload_preserves_saved_coordinates_despite_qr_roundoff(
    multiple, tmp_path, monkeypatch, angle
):
    from multimodalsrm.bayesian import persistence

    save, load = archive_api()
    model, scaler, test = multiple
    test = scaler.transform(test)
    save(tmp_path / "model", model)
    expected = model.transform(test, times=[4.0, 8.0])
    original = persistence.orientation

    def perturbed(*args):
        result = original(*args)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        result["rotation"] = (np.asarray(result["rotation"]) @ rotation).tolist()
        return result

    monkeypatch.setattr(persistence, "orientation", perturbed)
    forbid_inference_fitting(monkeypatch)
    if angle > 1e-10:
        with pytest.raises(ValueError, match="orientation"):
            load(tmp_path / "model")
    else:
        restored, _ = load(tmp_path / "model")
        assert restored.factor_orientation_ == model.factor_orientation_
        assert restored.configuration_ == model.configuration_
        actual = restored.transform(test, times=[4.0, 8.0])
        for s in test:
            assert_result_equal(actual[s]["test"], expected[s]["test"])


def test_finite_nonpassing_fit_keeps_original_diagnostics(scalar, tmp_path):
    save, load = archive_api()
    model = copy.deepcopy(scalar[0])
    # Retained failure metadata is an explicit archive contract, independent of
    # whether this small fixture happens to converge on a particular platform.
    model.map_diagnostics_["meets_gradient_tolerance"] = False
    model.training_fit_["map_diagnostics"]["meets_gradient_tolerance"] = False
    model.restart_diagnostics_.append(dict(objective=None, failure="retained failed restart"))
    model.training_fit_["restart_diagnostics"] = copy.deepcopy(model.restart_diagnostics_)
    save(tmp_path / "failed", model)
    restored, _ = load(tmp_path / "failed")
    assert restored.map_diagnostics_["meets_gradient_tolerance"] is False
    assert restored.restart_diagnostics_[-1] == dict(
        objective=None, failure="retained failed restart"
    )


@pytest.mark.parametrize("change", ["observations", "preprocessing"])
def test_save_rejects_training_state_that_no_longer_matches_fitted_problem(
    scalar, tmp_path, change
):
    from multimodalsrm import TimeSeries

    save, _ = archive_api()
    model = copy.deepcopy(scalar[0])
    if change == "observations":
        ts = model.training_data_["a"]["train"]["ref"]
        model.training_data_["a"]["train"]["ref"] = TimeSeries(ts.values + 3, ts.times, ts.mask)
    else:
        model.adapter_.preprocessing_["a"]["ref"]["mean"] = np.array([3.0])
    with pytest.raises(ValueError, match="training|preprocessing"):
        save(tmp_path / "changed", model)
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize(
    "change",
    ["posterior", "conditioned", "specification", "search", "draws", "orientation"],
)
def test_save_rejects_changed_or_unsupported_fitted_state(multiple, tmp_path, change):
    save, _ = archive_api()
    model, scaler, test = multiple
    model = copy.deepcopy(model)
    if change == "posterior":
        model.inference = "posterior"
    elif change == "conditioned":
        model = model.condition(scaler.transform(test), targets={"s1": ["brain"]}, mode="frozen")
    elif change == "specification":
        model.length_scale *= 2
    elif change == "search":
        model.search = SearchConfig(starts=3)
    elif change == "draws":
        model.parameter_draws_ = model.parameter_draws_ + 0.1
    else:
        model.factor_orientation_["rotation"][0][0] *= -1
    with pytest.raises(ValueError):
        save(tmp_path / "changed", model, standardizer=scaler)
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("change", ["missing", "dimension", "scale", "mean"])
def test_save_validates_scaler_mapping_dimensions_and_units(multiple, tmp_path, change):
    save, _ = archive_api()
    model, original, _ = multiple
    scaler = copy.deepcopy(original)
    stats = scaler.statistics["s1", "brain"]
    if change == "missing":
        del scaler.statistics["s1", "brain"]
    elif change == "dimension":
        stats["scale"] = np.ones(2)
    elif change == "scale":
        stats["scale"][0] = 0
    else:
        stats["mean"][0] = np.inf
    with pytest.raises(ValueError, match="standardizer"):
        save(tmp_path / "bad", model, standardizer=scaler)
    assert not (tmp_path / "bad").exists()


def reseal_manifest(path, manifest):
    text = json.dumps(manifest, allow_nan=False)
    (path / "manifest.json").write_text(text)
    (path / "manifest.sha256").write_text(hashlib.sha256(text.encode()).hexdigest())


def test_load_search_config_before_polishing_limit(scalar, tmp_path, monkeypatch):
    """The additive optimizer guard must not invalidate existing fitted archives."""
    save, load = archive_api()
    path = tmp_path / "legacy"
    model = scalar[0]
    save(path, model)
    manifest = json.loads((path / "manifest.json").read_text())
    records = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "record" and value.get("class") == "SearchConfig":
                records.append(value)
            if value.get("type") == "dict":
                value["items"] = [
                    pair for pair in value["items"] if pair[0] != "polish_max_parameters"
                ]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    assert records
    for record in records:
        del record["fields"]["polish_max_parameters"]
    reseal_manifest(path, manifest)
    forbid_inference_fitting(monkeypatch)
    restored, _ = load(path)
    assert restored.search.polish_max_parameters == 256
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    assert restored.map_diagnostics_ == model.map_diagnostics_
    assert "polish_max_parameters" not in restored.configuration_["search"]
    save(tmp_path / "resaved", restored)
    assert_result_equal(
        restored.infer_latent(times=[4.0, 8.0])["train"],
        model.infer_latent(times=[4.0, 8.0])["train"],
    )


def test_load_sampler_config_before_metric_fields_preserves_provenance(
    scalar, tmp_path, monkeypatch
):
    """Additive sampler defaults validate legacy MAP archives without rewriting them."""
    save, load = archive_api()
    path = tmp_path / "legacy"
    model = scalar[0]
    save(path, model)
    manifest = json.loads((path / "manifest.json").read_text())
    records = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "record" and value.get("class") == "SamplerConfig":
                records.append(value)
            if value.get("type") == "dict":
                value["items"] = [
                    pair
                    for pair in value["items"]
                    if pair[0] not in ("mass_matrix", "max_dense_parameters")
                ]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    for record in records:
        del record["fields"]["mass_matrix"]
        del record["fields"]["max_dense_parameters"]
    reseal_manifest(path, manifest)
    forbid_inference_fitting(monkeypatch)

    restored, _ = load(path)
    assert restored.sampler_config_ == SamplerConfig()
    assert "mass_matrix" not in restored.configuration_["sampler"]
    assert "max_dense_parameters" not in restored.configuration_["sampler"]
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    save(tmp_path / "resaved", restored)


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_legacy_sampler_config_still_rejects_other_field_changes(change):
    from multimodalsrm.bayesian import _archive

    record = _archive.encode(SamplerConfig(), {})
    del record["fields"]["mass_matrix"]
    del record["fields"]["max_dense_parameters"]
    if change == "missing":
        del record["fields"]["draws"]
    else:
        record["fields"]["unsupported"] = 1
    with pytest.raises(ValueError, match="record or fields"):
        _archive.decode(record, None, set())


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_legacy_search_config_still_rejects_other_field_changes(change):
    from multimodalsrm.bayesian import _archive

    record = _archive.encode(SearchConfig(), {})
    del record["fields"]["polish_max_parameters"]
    if change == "missing":
        del record["fields"]["maxiter"]
    else:
        record["fields"]["unsupported"] = 1
    with pytest.raises(ValueError, match="record or fields"):
        _archive.decode(record, None, set())


@pytest.mark.parametrize("damage", ["arrays", "manifest", "version", "record"])
def test_load_rejects_corrupt_or_unsupported_archives(scalar, tmp_path, damage):
    save, load = archive_api()
    path = tmp_path / "model"
    save(path, scalar[0])
    if damage == "arrays":
        with (path / "arrays.npz").open("ab") as out:
            out.write(b"corrupt")
    elif damage == "manifest":
        with (path / "manifest.json").open("a") as out:
            out.write(" ")
    else:
        manifest = json.loads((path / "manifest.json").read_text())
        if damage == "version":
            manifest["schema_version"] = 999
        else:
            manifest["state"] = {"type": "record", "class": "Executable", "fields": {}}
        reseal_manifest(path, manifest)
    with pytest.raises(ValueError, match="hash|version|record"):
        load(path)


@pytest.mark.parametrize("damage", ["parameter_order", "noise", "orientation", "scaler"])
def test_loader_rejects_semantically_inconsistent_resealed_state(multiple, tmp_path, damage):
    from multimodalsrm.bayesian import _archive

    save, load = archive_api()
    model, scaler, _ = multiple
    save(tmp_path / "valid", model, standardizer=scaler)
    state = _archive.read(tmp_path / "valid")
    if damage == "parameter_order":
        state["parameter_names"][0], state["parameter_names"][1] = (
            state["parameter_names"][1],
            state["parameter_names"][0],
        )
    elif damage == "noise":
        point = state["fit"]["map_parameters"].copy()
        point[next(i for i, n in enumerate(state["parameter_names"]) if n[0] == "noise")] = 0
        state["fit"]["map_parameters"] = point
    elif damage == "orientation":
        state["fit"]["configuration"]["factor_orientation"]["rotation"][0][0] *= -1
    else:
        state["standardizer"].statistics["s1", "brain"]["scale"] = np.zeros(3)
    _archive.write(tmp_path / "bad", state)
    with pytest.raises(ValueError):
        load(tmp_path / "bad")


@pytest.mark.parametrize("damage", ["flag", "optimizer", "gradient", "restarts", "timing", "prior"])
def test_loader_rejects_malformed_provenance_and_whitelisted_records(scalar, tmp_path, damage):
    from multimodalsrm.bayesian import _archive

    save, load = archive_api()
    save(tmp_path / "valid", scalar[0])
    state = _archive.read(tmp_path / "valid")
    if damage == "flag":
        state["fit"]["map_diagnostics"]["meets_gradient_tolerance"] = "false"
    elif damage == "optimizer":
        state["fit"]["map_diagnostics"]["optimizer_success"] = None
    elif damage == "gradient":
        state["fit"]["map_diagnostics"]["physical_projected_gradient"] = np.nan
    elif damage == "restarts":
        state["fit"]["restart_diagnostics"] = {"malformed": "not a list"}
    elif damage == "timing":
        state["phase_seconds"] = ["malformed timing"]
    else:
        object.__setattr__(state["constructor"]["priors"], "filters", None)
    _archive.write(tmp_path / "bad", state)
    with pytest.raises(ValueError):
        load(tmp_path / "bad")


def test_writer_never_replaces_existing_paths_and_failed_write_is_not_loadable(
    scalar, tmp_path, monkeypatch
):
    from multimodalsrm.bayesian import _archive

    save, load = archive_api()
    model = scalar[0]
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        save(destination, model)
    assert list(destination.iterdir()) == []
    link = tmp_path / "dangling"
    try:
        link.symlink_to(tmp_path / "missing")
    except OSError:
        pass  # Windows without symlink privileges still checks existing paths.
    else:
        with pytest.raises(FileExistsError):
            save(link, model)
        assert link.is_symlink()
    replace = _archive.os.replace

    def interrupted(source, target):
        if Path(target).name == "manifest.sha256":
            raise OSError("interrupted publication")
        return replace(source, target)

    monkeypatch.setattr(_archive.os, "replace", interrupted)
    with pytest.raises(OSError, match="publication"):
        save(tmp_path / "partial", model)
    with pytest.raises(ValueError, match="incomplete"):
        load(tmp_path / "partial")


@pytest.mark.parametrize("algebra", ["dense", "spectral"])
def test_scalar_default_responses_and_generated_seed_survive_reload(scalar, tmp_path, algebra):
    from multimodalsrm.bayesian import SpectralConfig

    save, load = archive_api()
    original, data = scalar
    params = original.get_params(deep=False)
    params.update(
        responses=None,
        random_state=None,
        run_baseline_sd=None,
        linear_algebra=algebra,
        spectral=SpectralConfig(12, 10.0) if algebra == "spectral" else None,
    )
    model = BayesianMultimodalSRM(**params).fit(data)
    expected = model.transform(donors(data), times=[4.0, 8.0, 12.0])
    save(tmp_path / "model", model)
    restored, _ = load(tmp_path / "model")
    assert restored.random_state is None
    assert restored.seed_ == model.seed_
    assert restored.configuration_ == model.configuration_
    actual = restored.transform(donors(data), times=[4.0, 8.0, 12.0])
    for s in actual:
        assert_result_equal(actual[s]["new"], expected[s]["new"])


def test_dense_multifactor_multiple_run_order_and_masked_nan_survive_reload(
    multiple, tmp_path, monkeypatch
):
    from multimodalsrm import TimeSeries

    original, scaler, test = multiple
    training = {}
    for s, runs in original.training_data_.items():
        mods = runs["train"]
        second = {}
        for m, ts in mods.items():
            values = ts.values.copy() + 0.05
            values[~ts.mask] = np.nan
            second[m] = TimeSeries(values, ts.times, ts.mask)
        training[s] = {"z-first": mods, "a-second": second}
    params = original.get_params(deep=False)
    params.update(linear_algebra="dense", search=SearchConfig(starts=1, maxiter=4))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model = BayesianMultimodalSRM(**params).fit(training)
    query = [4.0, 8.0]
    expected = model.infer_latent(times=query)
    transformed = model.transform(scaler.transform(test), times=query)
    save, load = archive_api()
    save(tmp_path / "model", model, standardizer=scaler)
    forbid_inference_fitting(monkeypatch)
    restored, restored_scaler = load(tmp_path / "model")
    actual = restored.infer_latent(times=query)
    assert list(actual) == ["z-first", "a-second"]
    for r in actual:
        assert_result_equal(actual[r], expected[r])
    ts = restored.training_data_["s1"]["a-second"]["aux"]
    assert np.isnan(ts.values[~ts.mask]).all()
    actual = restored.transform(restored_scaler.transform(test), times=query)
    for s in actual:
        assert_result_equal(actual[s]["test"], transformed[s]["test"])


def test_archive_reuse_in_fresh_process_has_no_optimizer_or_parent_object(scalar, tmp_path):
    save, _ = archive_api()
    model, data = scalar
    expected = model.transform(donors(data), times=[4.0, 8.0, 12.0])
    prediction = model.condition(donors(data), targets={"b": ["signal"]}, mode="frozen").predict(
        times=[4.0, 8.0, 12.0]
    )["b"]["new"]["signal"]
    path = tmp_path / "model"
    save(path, model)
    script = """
import sys
from pathlib import Path
import numpy as np
from multimodalsrm.bayesian import model, fitting
from multimodalsrm.bayesian.workflow import load_model, TrainingStandardizer
def forbidden(*a, **k):
    raise AssertionError("unexpected fitting")
model.search = model.sample = fitting.search = fitting.sample = forbidden
TrainingStandardizer.fit = forbidden
restored, scaler = load_model(Path(sys.argv[1]))
data = {s: {"new": r["train"]} for s,r in restored.training_data_.items()}
output = restored.transform(data, times=[4.,8.,12.])
prediction = restored.condition(data, targets={"b": ["signal"]}, mode="frozen").predict(
    times=[4.,8.,12.]
)["b"]["new"]["signal"]
results = {s: v["new"] for s, v in output.items()}
results["prediction"] = prediction
assert "multimodalsrm.probabilistic" not in sys.modules
assert "multimodalsrm.continuous.model" not in sys.modules
np.savez(sys.argv[2], **{
    f"{s}_{field}": getattr(v, field)
    for s, v in results.items()
    for field in ("component_means", "component_variances", "times", "valid")
})
"""
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(tmp_path / "actual.npz")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / "actual.npz", allow_pickle=False) as actual:
        results = {s: v["new"] for s, v in expected.items()}
        results["prediction"] = prediction
        for s, v in results.items():
            for field in ("component_means", "component_variances", "times", "valid"):
                assert_array_equal(actual[f"{s}_{field}"], getattr(v, field))
