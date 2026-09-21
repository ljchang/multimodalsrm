"""Posterior archives preserve raw identity and reporting without inference replay."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from multimodalsrm.bayesian import _archive, workflow
from multimodalsrm.bayesian.fitting import SearchConfig

from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting


@pytest.fixture(scope="module", params=["multifactor", "scalar", "conditional"])
def posterior(request):
    import warnings

    from ..reference.posterior_fixture import dataset, estimator

    data, _, _ = dataset()
    scaler = workflow.TrainingStandardizer.fit(data)
    data = scaler.transform(data)
    model = estimator("grouped_diagonal")
    model.inference = "posterior"
    model.search = SearchConfig(starts=1, maxiter=8, refine_maxiter=0)
    model.sampler = replace(model.sampler, chains=2, warmup=4, draws=6, max_tree_depth=2)
    if request.param == "multifactor":
        model.sampler = replace(model.sampler, mass_matrix="dense")
    if request.param == "scalar":
        model.features, model.factor_anchors = 1, None
    if request.param == "conditional":
        model.sample_blocks = ("noise",)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model.fit(data)
    assert model.sampling_diagnostics_["passes"] is False
    return model, scaler


def test_roundtrip_preserves_raw_arrays_mixtures_and_failed_diagnostics(
    posterior, tmp_path, monkeypatch
):
    model, scaler = posterior
    raw = model.parameter_draws_.copy()
    latent = model.infer_latent(times=[10.0, 16.0, 22.0], max_draws=None)["train"]
    reported = model.reported_parameter_draws()
    trajectories = model.sample_latent(
        times=[10.0, 16.0, 22.0],
        max_draws=2,
        random_state=92,
    )["train"]
    workflow.save_model(tmp_path / "posterior", model, standardizer=scaler)
    manifest = json.loads((tmp_path / "posterior/manifest.json").read_text())
    assert manifest["schema_version"] == 3
    forbid_inference_fitting(monkeypatch)
    restored, restored_scaler = workflow.load_model(tmp_path / "posterior")
    assert restored.parameter_names_ == model.parameter_names_
    assert _archive.same(restored.training_fit_, model.training_fit_)
    assert _archive.same(restored.configuration_, model.configuration_)
    assert _archive.same(restored.sampling_diagnostics_, model.sampling_diagnostics_)
    assert restored.sampling_diagnostics_["passes"] is False
    for name in ("map_parameters_", "parameter_draws_", "log_likelihood_draws_"):
        expected, actual = getattr(model, name), getattr(restored, name)
        assert_array_equal(actual, expected)
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, expected)
    for name, values in model.sample_stats_.items():
        assert_array_equal(restored.sample_stats_[name], values)
        assert not restored.sample_stats_[name].flags.writeable
    for subject, runs in model.training_data_.items():
        for run, modalities in runs.items():
            for modality, series in modalities.items():
                loaded = restored.training_data_[subject][run][modality]
                for name in ("values", "times", "mask"):
                    assert_array_equal(getattr(loaded, name), getattr(series, name))
                    assert not getattr(loaded, name).flags.writeable
    for key, stats in scaler.statistics.items():
        for field, values in stats.items():
            assert_array_equal(restored_scaler.statistics[key][field], values)
            assert not restored_scaler.statistics[key][field].flags.writeable
    assert_array_equal(restored.reported_parameter_draws(), reported)
    restored_trajectories = restored.sample_latent(
        times=[10.0, 16.0, 22.0],
        max_draws=2,
        random_state=92,
    )["train"]
    assert_array_equal(restored_trajectories.samples, trajectories.samples)
    assert _archive.same(restored_trajectories.metadata, trajectories.metadata)
    assert_result_equal(
        restored.infer_latent(times=[10.0, 16.0, 22.0], max_draws=None)["train"], latent
    )
    # Frozen fitted anchors still govern reporting if constructor options change.
    if model.features > 1:
        restored.factor_anchors = (("missing", "brain", 0),) * 2
        assert_array_equal(restored.reported_parameter_draws(), reported)
    assert_array_equal(model.parameter_draws_, raw)


def test_unknown_training_provenance_stays_unknown_and_writer_is_separate(posterior, tmp_path):
    model, _ = posterior
    workflow.save_model(tmp_path / "one", model)
    first, _ = workflow.load_model(tmp_path / "one")
    assert first.posterior_provenance_ is None
    assert first.archive_writer_provenance_["role"] == "archive_writer_not_training"
    assert "numpy" in first.archive_writer_provenance_["packages"]
    first.posterior_provenance_ = {
        "source_sha256": "caller-supplied",
        "seed_roles": {"map": 719},
        "notes": ["original retained fit"],
    }
    workflow.save_model(tmp_path / "two", first)
    second, _ = workflow.load_model(tmp_path / "two")
    assert second.posterior_provenance_ == first.posterior_provenance_
    assert second.archive_writer_provenance_["role"] == "archive_writer_not_training"


@pytest.mark.parametrize(
    "damage",
    [
        "shape",
        "nan",
        "noise",
        "stat",
        "likelihood",
        "chains",
        "active",
        "fixed",
        "seed",
        "conditioned",
        "anchor",
        "provenance",
    ],
)
def test_save_rejects_changed_posterior_state_without_creating_path(posterior, tmp_path, damage):
    original, scaler = posterior
    model = copy.deepcopy(original)
    if damage == "shape":
        model.parameter_draws_ = model.parameter_draws_[:, :-1]
    elif damage == "nan":
        model.parameter_draws_[0, 0, 0] = np.nan
    elif damage == "noise":
        i = next(i for i, name in enumerate(model.parameter_names_) if name[0] == "noise")
        model.parameter_draws_[0, 0, i] = 0.0
    elif damage == "stat":
        model.sample_stats_["energy"] = model.sample_stats_["energy"][:, :-1]
    elif damage == "likelihood":
        model.log_likelihood_draws_ = np.zeros((1, 1))
    elif damage == "chains":
        model.sampling_diagnostics_["chains"] += 1
    elif damage == "active":
        model.sampling_diagnostics_["active_parameters"].reverse()
    elif damage == "fixed":
        model.configuration_["fixed_parameters"] = [{"name": ["noise", "unknown"], "value": 1.0}]
    elif damage == "seed":
        model.random_state += 1
    elif damage == "conditioned":
        model.targets_ = {"a": ["brain"]}
    elif damage == "anchor":
        if model.features == 1:
            model.anchor = ("b", "brain", 0)
        else:
            model._factor_anchor_keys_ = (("a", "brain", 0), ("b", "brain", 0))
    else:
        model.posterior_provenance_ = {"callable": lambda: None}
    with pytest.raises(ValueError):
        workflow.save_model(tmp_path / "invalid", model, standardizer=scaler)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "names",
        "axes",
        "stats",
        "dtype",
        "diag",
        "config",
        "kind",
        "version",
        "training",
    ],
)
def test_load_rejects_incompatible_resealed_state(posterior, tmp_path, damage):
    model, scaler = posterior
    workflow.save_model(tmp_path / "good", model, standardizer=scaler)
    state = _archive.read(tmp_path / "good")
    version = 3
    if damage == "names":
        state["parameter_names"][:2] = state["parameter_names"][:2][::-1]
    elif damage == "axes":
        state["draw_axes"] = ["draw", "chain", "parameter"]
    elif damage == "stats":
        del state["sample_stats"]["diverging"]
    elif damage == "dtype":
        state["parameter_draws"] = np.ones(state["parameter_draws"].shape, dtype=int)
    elif damage == "diag":
        state["sampling_diagnostics"]["passes"] = "true"
    elif damage == "config":
        state["fit"]["configuration"]["factor_orientation"] = {"method": "wrong"}
    elif damage == "kind":
        state["kind"] = "participant_calibration"
    elif damage == "version":
        version = 1
    else:
        # Replacing native clocks with incompatible support must not silently
        # reinterpret the recorded original fit/configuration.
        state["training"] = {}
    _archive.write(tmp_path / "bad", state, version=version)
    with pytest.raises(ValueError):
        workflow.load_model(tmp_path / "bad")


def test_writer_never_overwrites_or_accepts_unsealed_corruption(posterior, tmp_path):
    model, _ = posterior
    path = tmp_path / "posterior"
    workflow.save_model(path, model)
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.save_model(path, model)
    assert (path / "manifest.json").read_bytes() == before
    with (path / "arrays.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        workflow.load_model(path)


def test_rejects_hybrid_map_provenance_even_with_valid_posterior_arrays(posterior, tmp_path):
    model = copy.deepcopy(posterior[0])
    model.training_fit_["inference"] = "map"
    model.training_fit_["configuration"]["inference"] = "map"
    with pytest.raises(ValueError, match="fit|posterior"):
        workflow.save_model(tmp_path / "hybrid", model)
    assert not (tmp_path / "hybrid").exists()


@pytest.mark.parametrize("damage", ["observations", "aggregate", "metric_bytes", "original_fit"])
def test_semantic_contract_detects_resealed_inconsistency(posterior, tmp_path, damage):
    from multimodalsrm import TimeSeries

    model, _ = posterior
    workflow.save_model(tmp_path / "good", model)
    state = _archive.read(tmp_path / "good")
    if damage == "observations":
        ts = state["training"]["a"]["train"]["brain"]
        state["training"]["a"]["train"]["brain"] = TimeSeries(ts.values + 0.01, ts.times, ts.mask)
    elif damage == "aggregate":
        state["sampling_diagnostics"]["min_bulk_ess"] += 1
    elif damage == "metric_bytes":
        state["sampling_diagnostics"]["metric"]["estimated_bytes_per_chain"] += 8
    else:
        state["original_training_fit"]["inference"] = "map"
    _archive.write(tmp_path / "bad", state, version=3)
    with pytest.raises(ValueError):
        workflow.load_model(tmp_path / "bad")


def test_nonfinite_failed_diagnostic_is_retained_without_qualification(posterior, tmp_path):
    model = copy.deepcopy(posterior[0])
    # Undefined R-hat is a legitimate failed diagnostic, not archive corruption.
    for row in model.sampling_diagnostics_["parameters"]:
        row["r_hat"] = np.nan
    model.sampling_diagnostics_["max_rank_rhat"] = np.nan
    workflow.save_model(tmp_path / "failed", model)
    restored, _ = workflow.load_model(tmp_path / "failed")
    assert np.isnan(restored.sampling_diagnostics_["max_rank_rhat"])
    assert restored.sampling_diagnostics_["passes"] is False


def test_nested_fit_arrays_are_readonly_after_restore(posterior, tmp_path):
    workflow.save_model(tmp_path / "posterior", posterior[0])
    restored, _ = workflow.load_model(tmp_path / "posterior")
    assert not restored.training_fit_["map_parameters"].flags.writeable


def test_observed_noise_and_subselected_latent_mixtures_survive_restore(
    posterior, tmp_path, monkeypatch
):
    from multimodalsrm.bayesian.prediction import result, selected_draws

    model, _ = posterior
    times = np.array([10.0, 16.0, 22.0])
    draws, ids = selected_draws(model, None, return_indices=True)
    expected = {
        noise: result(
            model,
            "train",
            times,
            draws,
            [("a", "brain", 0)],
            include_noise=noise,
            draw_indices=ids,
        )
        for noise in (False, True)
    }
    subset = model.infer_latent(times=times, max_draws=4)["train"]
    workflow.save_model(tmp_path / "posterior", model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / "posterior")
    draws, ids = selected_draws(restored, None, return_indices=True)
    for noise in (False, True):
        actual = result(
            restored,
            "train",
            times,
            draws,
            [("a", "brain", 0)],
            include_noise=noise,
            draw_indices=ids,
        )
        assert_result_equal(actual, expected[noise])
        assert_array_equal(actual.variance, expected[noise].variance)
        assert_array_equal(
            actual.between_parameter_variance,
            expected[noise].between_parameter_variance,
        )
    assert np.all(expected[True].component_variances > expected[False].component_variances)
    assert_result_equal(restored.infer_latent(times=times, max_draws=4)["train"], subset)
    assert subset.metadata["parameter_draw_indices"] == [[0, 0], [0, 5], [1, 0], [1, 5]]


# These contradictions previously survived both live saves and resealed loads.
RECORDED_IDENTITY_DAMAGE = [
    (("sampler_seed",), 123456),
    (("initialization_seed",), 123456),
    (("execution", "chains"), 999),
    (("execution", "requested_chain_method"), "parallel"),
    (("execution", "effective_chain_method"), "parallel"),
    (("execution", "local_device_count"), 0),
    (("execution", "devices"), []),
    (("execution", "parallel_verified"), True),
    (("start_indices",), [0]),
    (("start_indices",), [999, 999]),
    (("start_indices",), [False, False]),
    (("initial_unconstrained",), [[0.0]]),
    (("initial_unconstrained",), "non-numeric"),
    (("adaptation", "inverse_mass_matrix_shape"), [999, 999]),
    (("adaptation", "inverse_mass_matrix_dtype"), "float32"),
    (("adaptation", "step_size_shape"), [999]),
    (("adaptation", "step_size_dtype"), "int64"),
    (("adaptation", "step_size"), [0.0, 0.0]),
    (("adaptation", "step_size"), [float("nan"), 1.0]),
    (("adaptation", "inverse_mass_matrix"), [[0.0]]),
    (("adaptation", "values_stored"), False),
    (("metric", "adapted_values_stored"), False),
]


@pytest.mark.parametrize("path,value", RECORDED_IDENTITY_DAMAGE)
@pytest.mark.parametrize("route", ["live", "resealed"])
def test_rejects_contradictory_sampler_records(
    posterior, tmp_path, monkeypatch, path, value, route
):
    model = copy.deepcopy(posterior[0])
    forbid_inference_fitting(monkeypatch)
    if route == "resealed":
        workflow.save_model(tmp_path / "good", model)
        state = _archive.read(tmp_path / "good")
        diagnostic = state["sampling_diagnostics"]
    else:
        diagnostic = model.sampling_diagnostics_
    target = diagnostic
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if route == "resealed":
        _archive.write(tmp_path / "bad", state, version=3)
        with pytest.raises(ValueError):
            workflow.load_model(tmp_path / "bad")
    else:
        with pytest.raises(ValueError):
            workflow.save_model(tmp_path / "bad", model)
        assert not (tmp_path / "bad").exists()


def test_preserves_historical_absent_sampler_metadata(posterior, tmp_path):
    model = copy.deepcopy(posterior[0])
    diagnostic = model.sampling_diagnostics_
    for key in ("sampler_seed", "initialization_seed", "adaptation"):
        del diagnostic[key]
    diagnostic["metric"]["adapted_values_stored"] = False
    workflow.save_model(tmp_path / "historical", model)
    restored, _ = workflow.load_model(tmp_path / "historical")
    assert _archive.same(restored.sampling_diagnostics_, diagnostic)
    assert restored.posterior_provenance_ is None


def test_recorded_execution_is_not_compared_to_loader_host(posterior, tmp_path, monkeypatch):
    model = copy.deepcopy(posterior[0])
    # A consistent historical execution can name devices unavailable here.
    model.sampling_diagnostics_["execution"].update(
        local_device_count=2, devices=["historical-0", "historical-1"], backend="gpu"
    )
    workflow.save_model(tmp_path / "other-host", model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / "other-host")
    assert _archive.same(restored.sampling_diagnostics_, model.sampling_diagnostics_)


@pytest.mark.parametrize("damage", ["initial_nan", "metric_nonpositive", "metric_nan"])
def test_rejects_invalid_recorded_numeric_values(posterior, tmp_path, damage):
    workflow.save_model(tmp_path / "good", posterior[0])
    state = _archive.read(tmp_path / "good")
    diagnostic = state["sampling_diagnostics"]
    if damage == "initial_nan":
        diagnostic["initial_unconstrained"][0][0] = float("nan")
    else:
        metric = np.asarray(diagnostic["adaptation"]["inverse_mass_matrix"])
        metric.flat[0] = -1.0 if damage == "metric_nonpositive" else float("nan")
        diagnostic["adaptation"]["inverse_mass_matrix"] = metric.tolist()
    _archive.write(tmp_path / "bad", state, version=3)
    with pytest.raises(ValueError):
        workflow.load_model(tmp_path / "bad")
