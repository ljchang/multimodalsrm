"""Updated posterior archives authenticate the target without repeating inference."""

import copy
import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from multimodalsrm.bayesian import _archive, workflow
from multimodalsrm.bayesian.posterior_updates import content_id

from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting
from .test_bayesian_posterior_updates import donors, prepared


@pytest.fixture(scope="module")
def updated():
    model = prepared()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(model.training_data_)
        return model.condition(donors(model), targets={"a": ["aux"]})


def test_donor_roundtrip_without_preprocessing_or_inference(updated, tmp_path, monkeypatch):
    from multimodalsrm._observation_preparation import (
        ObservationPreparation,
    )

    query = {"new": np.array([7.0, 9.0])}
    expected = updated.predict(times=query, include_noise=True, max_draws=None)
    expected_latent = updated.infer_latent(times=query, max_draws=None)
    expected_paths = updated.sample_latent(times=query, max_draws=2, random_state=92)
    workflow.save_model(tmp_path / "updated", updated)
    assert json.loads((tmp_path / "updated/manifest.json").read_text())["schema_version"] == 4
    forbid_inference_fitting(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("loading must not estimate preprocessing")

    monkeypatch.setattr(ObservationPreparation, "_prepare", forbidden)
    restored, scaler = workflow.load_model(tmp_path / "updated")
    assert scaler is None
    assert _archive.same(restored.posterior_update_, updated.posterior_update_)
    assert _archive.same(restored.sampling_diagnostics_, updated.sampling_diagnostics_)
    assert restored.sampling_diagnostics_["passes"] is False
    assert_array_equal(restored.parameter_draws_, updated.parameter_draws_)
    assert_array_equal(restored.reported_parameter_draws(), updated.reported_parameter_draws())
    assert not restored.parameter_draws_.flags.writeable
    assert_result_equal(
        restored.predict(times=query, include_noise=True, max_draws=None)["a"]["new"]["aux"],
        expected["a"]["new"]["aux"],
    )
    assert_result_equal(
        restored.infer_latent(times=query, max_draws=None)["new"],
        expected_latent["new"],
    )
    paths = restored.sample_latent(times=query, max_draws=2, random_state=92)
    assert_array_equal(paths["new"].samples, expected_paths["new"].samples)
    assert _archive.same(paths["new"].metadata, expected_paths["new"].metadata)


@pytest.mark.parametrize(
    "damage",
    ["evidence", "source", "scope", "anchors", "prior", "names", "queries", "adapter"],
)
def test_updated_save_rejects_inconsistent_contract(updated, tmp_path, damage):
    model = copy.deepcopy(updated)
    if damage == "evidence":
        model.posterior_update_["evidence"]["a"].pop("new")
    elif damage == "source":
        model.posterior_update_["source_fit_id"] = "f" * 64
    elif damage == "scope":
        model.configuration_["diagnostic_scope"] = "training_only"
    elif damage == "anchors":
        model._factor_anchor_keys_ = tuple(reversed(model._factor_anchor_keys_))
    elif damage == "prior":
        model.problem_.parameter_priors[-1] = model.problem_.parameter_priors[0]
    elif damage == "queries":
        model.prediction_runs_ = {"other": (0.0, 16.0)}
    elif damage == "adapter":
        model.adapter_ = copy.deepcopy(model.adapter_)
        model.adapter_.preprocessing_["a"]["brain"]["mean"][0] = 1.0
    else:
        model.parameter_names_ = list(reversed(model.parameter_names_))
    with pytest.raises(ValueError):
        workflow.save_model(tmp_path / "bad", model)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "damage", ["ledger", "duplicate", "prior", "arrays", "finite_arrays", "source"]
)
def test_restore_rejects_tampering_with_fresh_payload_hash(updated, tmp_path, damage):
    workflow.save_model(tmp_path / "good", updated)
    saved = _archive.read(tmp_path / "good")
    if damage == "ledger":
        saved["posterior_update"]["evidence"]["a"].pop("new")
    elif damage == "duplicate":
        saved["posterior_update"]["evidence"]["a"]["train"] = saved["posterior_update"]["evidence"][
            "a"
        ].pop("new")
        saved["posterior_update_sha256"] = content_id(saved["posterior_update"])
    elif damage == "prior":
        from dataclasses import replace

        saved["posterior_update"]["priors"] = replace(
            saved["posterior_update"]["priors"], loading_sd=3.0
        )
        saved["posterior_update_sha256"] = content_id(saved["posterior_update"])
    elif damage == "source":
        saved["posterior_update"]["source_fit_id"] = "0" * 64
        saved["posterior_update"]["reference"]["source_fit_id"] = "0" * 64
        saved["posterior_update_sha256"] = content_id(saved["posterior_update"])
    else:
        saved["parameter_draws"] = saved["parameter_draws"].copy()
        saved["parameter_draws"][0, 0, -1] = (
            saved["parameter_draws"][0, 0, -1] + 0.001 if damage == "finite_arrays" else np.nan
        )
    _archive.write(tmp_path / "tampered", saved, version=4)
    with pytest.raises(ValueError):
        workflow.load_model(tmp_path / "tampered")


def test_updated_kind_cannot_be_disguised_as_version3(updated, tmp_path):
    workflow.save_model(tmp_path / "good", updated)
    saved = _archive.read(tmp_path / "good")
    _archive.write(tmp_path / "bad", saved, version=3)
    with pytest.raises(ValueError, match="schema version"):
        workflow.load_model(tmp_path / "bad")


@pytest.fixture(scope="module")
def participant_states(updated):
    original = updated.posterior_update_["reference"]["adapter"]["fitted"]["_training_data"]
    calibration_data = {"third": {"train": {"brain": original["a"]["train"]["brain"]}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = updated.calibrate_posterior(calibration_data, random_state=77)
        individual = calibrated.condition_participants(
            {"third": {"new": calibration_data["third"]["train"]}},
            random_state=78,
        )["third"]
    return {"calibration": calibrated, "participant": individual}


@pytest.mark.parametrize("kind", ["calibration", "participant"])
def test_calibrated_and_independent_roundtrip(participant_states, kind, tmp_path, monkeypatch):
    from multimodalsrm._observation_preparation import (
        ObservationPreparation,
    )
    from multimodalsrm.bayesian.posterior_updates import reference_state

    model = participant_states[kind]
    run = "train" if kind == "calibration" else "new"
    times = {run: [7.0, 9.0]}
    expected = model.infer_latent(times=times, max_draws=None)[run]
    expected_paths = model.sample_latent(times=times, max_draws=2, random_state=93)[run]
    expected_reference = reference_state(model)
    workflow.save_model(tmp_path / kind, model)
    forbid_inference_fitting(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("preprocessing must not be estimated on restore")

    monkeypatch.setattr(ObservationPreparation, "_prepare", forbidden)
    loaded, _ = workflow.load_model(tmp_path / kind)
    assert _archive.same(reference_state(loaded), expected_reference)
    assert _archive.same(loaded.configuration_, model.configuration_)
    assert _archive.same(loaded.sampling_diagnostics_, model.sampling_diagnostics_)
    assert_array_equal(loaded.parameter_draws_, model.parameter_draws_)
    assert_result_equal(loaded.infer_latent(times=times, max_draws=None)[run], expected)
    paths = loaded.sample_latent(times=times, max_draws=2, random_state=93)[run]
    assert_array_equal(paths.samples, expected_paths.samples)
    assert _archive.same(paths.metadata, expected_paths.metadata)
    assert "third" in loaded.training_data_
    assert set(loaded.posterior_update_["reference"]["adapter"]["fitted"]["_training_data"]) == (
        {"a", "b", "c"} if kind == "calibration" else {"a", "b", "c", "third"}
    )


def test_restored_calibration_remains_reference_for_new_updates(participant_states, tmp_path):
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    calibrated = participant_states["calibration"]
    workflow.save_model(tmp_path / "calibrated", calibrated)
    restored, _ = workflow.load_model(tmp_path / "calibrated")
    direct = prepare_condition(calibrated, donors(calibrated), targets={"a": ["aux"]})
    roundtrip = prepare_condition(restored, donors(restored), targets={"a": ["aux"]})
    assert _archive.same(direct.posterior_update_, roundtrip.posterior_update_)
    assert _archive.same(direct.problem_.systems, roundtrip.problem_.systems)


@pytest.mark.parametrize("entry", ["fit", "sample"])
def test_updated_target_checked_before_any_inference(entry, monkeypatch):
    from dataclasses import replace

    import numpyro.distributions as dist

    from multimodalsrm.bayesian import fitting
    from multimodalsrm.bayesian import model as model_module
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    original = prepared()
    model = prepare_condition(original, donors(original), targets={"a": ["aux"]})
    model.problem_.distributions[0] = dist.Normal(1.0, 1.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid target reached optimization")

    monkeypatch.setattr(model_module, "search", forbidden)
    with pytest.raises(ValueError, match="update.*density|update.*target"):
        if entry == "fit":
            model._fit_problem()
        else:
            fitting.sample(
                model.problem_,
                [],
                replace(model.sampler_config_, orientation_refresh="none"),
                1,
            )


def test_full_scalar_update_uses_schema4_and_roundtrips(tmp_path):
    from dataclasses import replace

    group = prepared()
    group.features, group.factor_anchors = 1, None
    group.sampler = replace(group.sampler, orientation_refresh="none")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group.fit(group.training_data_)
        model = group.condition(donors(group), targets={"a": ["aux"]})
    assert model.posterior_update_["kind"] == "donor"
    workflow.save_model(tmp_path / "scalar", model)
    restored, _ = workflow.load_model(tmp_path / "scalar")
    assert_array_equal(restored.parameter_draws_, model.parameter_draws_)
    assert_result_equal(
        restored.predict(times={"new": [7.0, 9.0]})["a"]["new"]["aux"],
        model.predict(times={"new": [7.0, 9.0]})["a"]["new"]["aux"],
    )
