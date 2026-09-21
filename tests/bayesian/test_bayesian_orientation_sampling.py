"""Public integration and archive identity for in-chain orientation refreshes."""

import copy
import warnings
from dataclasses import asdict

import numpy as np
import pytest

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    Prior,
    SamplerConfig,
    SearchConfig,
    _archive,
    workflow,
)
from multimodalsrm.bayesian.persistence import _sampler_for_validation

from .test_bayesian_multifactor_posterior import posterior as _ordinary_fixture

ordinary_posterior = _ordinary_fixture


def test_refresh_is_opt_in_and_legacy_config_is_not_rewritten():
    assert SamplerConfig().orientation_refresh == "none"
    assert SamplerConfig(orientation_refresh="haar").orientation_refresh == "haar"
    old = asdict(SamplerConfig())
    old.pop("orientation_refresh")
    original = copy.deepcopy(old)
    assert _sampler_for_validation(old)["orientation_refresh"] == "none"
    assert old == original


@pytest.mark.parametrize("value", [True, None, "rotate", 1, []])
def test_invalid_refresh_mode_rejected(value):
    with pytest.raises(ValueError, match="orientation_refresh"):
        SamplerConfig(orientation_refresh=value)


@pytest.fixture(scope="module")
def refreshed_posterior():
    times = np.arange(6.0)
    data = {
        "a": {
            "train": {"brain": TimeSeries(np.column_stack([np.sin(times), np.cos(times)]), times)}
        }
    }
    model = BayesianMultimodalSRM(
        priors=BayesianPriors(noise=Prior.lognormal(-1.0, 0.5)),
        features=2,
        factor_anchors=(("a", "brain", 1), ("a", "brain", 0)),
        inference="posterior",
        linear_algebra="grouped",
        random_state=648,
        search=SearchConfig(starts=1, maxiter=6),
        sampler=SamplerConfig(
            chains=2,
            warmup=6,
            draws=8,
            max_tree_depth=3,
            mass_matrix="diagonal",
            orientation_refresh="haar",
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    return model


def test_public_posterior_retains_orientation_and_raw_gates(refreshed_posterior):
    model = refreshed_posterior
    d = model.sampling_diagnostics_
    assert d["orientation_refresh"] == "haar"
    assert d["orientation"]["applicable"] is True
    assert d["orientation"]["anchor_keys"] == [["a", "brain", 1], ["a", "brain", 0]]
    assert d["orientation"]["anchor_indices"] == [1, 0]
    assert d["passes"] == (d["raw_passes"] and d["orientation"]["passes"])
    assert model.parameter_draws_.shape == (2, 8, len(model.parameter_names_))
    assert all(np.shape(v) == (2, 8) for v in model.sample_stats_.values())


def test_refresh_archive_roundtrip_does_not_run_sampling(
    refreshed_posterior, tmp_path, monkeypatch
):
    from .test_bayesian_persistence import forbid_inference_fitting

    model = refreshed_posterior
    workflow.save_model(tmp_path / "fit", model)
    forbid_inference_fitting(monkeypatch)
    loaded, _ = workflow.load_model(tmp_path / "fit")
    assert loaded.sampler_config_.orientation_refresh == "haar"
    np.testing.assert_array_equal(loaded.parameter_draws_, model.parameter_draws_)
    assert _archive.same(loaded.sampling_diagnostics_, model.sampling_diagnostics_)
    np.testing.assert_array_equal(
        loaded.reported_parameter_draws(), model.reported_parameter_draws()
    )


@pytest.mark.parametrize("damage", ["method", "occupancy", "pass_flag", "missing"])
def test_archive_rejects_inconsistent_orientation_evidence(refreshed_posterior, tmp_path, damage):
    model = copy.deepcopy(refreshed_posterior)
    d = model.sampling_diagnostics_
    if damage == "method":
        d["orientation_refresh"] = "none"
    elif damage == "occupancy":
        old = d["orientation"]["chains"][0]["positive_determinant_fraction"]
        d["orientation"]["chains"][0]["positive_determinant_fraction"] = (old + 0.25) % 1
    elif damage == "pass_flag":
        d["orientation"]["passes"] = not d["orientation"]["passes"]
    else:
        del d["orientation"]
    with pytest.raises(ValueError, match="orientation"):
        workflow.save_model(tmp_path / "bad", model)
    assert not (tmp_path / "bad").exists()


def test_constant_reflection_cannot_claim_finite_diagnostics(refreshed_posterior):
    from multimodalsrm.bayesian.orientation_persistence import (
        validate_orientation_record,
    )

    model = copy.deepcopy(refreshed_posterior)
    model.parameter_draws_ = model.parameter_draws_.copy()
    w = model.parameter_draws_[..., :4].reshape(2, 8, 2, 2)
    sign = np.linalg.slogdet(w[..., [1, 0], :])[0]
    w[..., 1] *= sign[..., None]
    o = model.sampling_diagnostics_["orientation"]
    for record in o["per_quantity"]:
        record.update(status="computed", r_hat=1.0, ess_bulk=1000.0, ess_tail=1000.0, passes=True)
    o.update(passes=True, max_rank_rhat=1.0, min_bulk_ess=1000.0, min_tail_ess=1000.0)
    for record in o["chains"]:
        record.update(positive_determinant_fraction=1.0, reflection_transitions=0)
    model.sampling_diagnostics_["passes"] = model.sampling_diagnostics_["raw_passes"]
    with pytest.raises(ValueError, match="orientation.*constant"):
        validate_orientation_record(model)


def test_new_baseline_cannot_erase_orientation_evidence(ordinary_posterior, tmp_path):
    model = copy.deepcopy(ordinary_posterior[0])
    for key in ("orientation", "orientation_refresh", "raw_passes"):
        del model.sampling_diagnostics_[key]
    with pytest.raises(ValueError, match="orientation"):
        workflow.save_model(tmp_path / "bad", model)


def test_legacy_absence_is_preserved_when_saved_config_predates_orientation(
    ordinary_posterior, tmp_path
):
    model = copy.deepcopy(ordinary_posterior[0])
    for key in ("orientation", "orientation_refresh", "raw_passes"):
        del model.sampling_diagnostics_[key]
    model.configuration_["sampler"].pop("orientation_refresh")
    model.training_fit_["configuration"]["sampler"].pop("orientation_refresh")
    workflow.save_model(tmp_path / "legacy", model)
    restored, _ = workflow.load_model(tmp_path / "legacy")
    assert restored.sampler_config_.orientation_refresh == "none"
    assert "orientation" not in restored.sampling_diagnostics_
    assert "orientation_refresh" not in restored.configuration_["sampler"]
    np.testing.assert_array_equal(restored.parameter_draws_, model.parameter_draws_)


def test_partly_undefined_orientation_evidence_is_valid_failed_record(
    refreshed_posterior,
):
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )
    from multimodalsrm.bayesian.orientation_persistence import (
        validate_orientation_record,
    )

    model = copy.deepcopy(refreshed_posterior)
    model.parameter_draws_ = model.parameter_draws_.copy()
    # Both split chains have a constant reflection, though each full chain moves.
    # Rank R-hat is undefined while ESS remains finite: preserve that evidence.
    theta = np.arange(16).reshape(2, 8) * 0.17
    sign = np.tile([1, 1, 1, 1, -1, -1, -1, -1], (2, 1))
    c, s = np.cos(theta), np.sin(theta)
    q = np.stack([c, -sign * s, s, sign * c], axis=-1).reshape(2, 8, 2, 2)
    w = q.swapaxes(-1, -2)[..., [1, 0], :]
    model.parameter_draws_[..., :4] = w.reshape(2, 8, 4)
    d = model.sampling_diagnostics_
    context = {
        k: d["orientation"][k]
        for k in ("applicable", "anchor_indices", "anchor_keys", "anchor_source")
    }
    d["orientation"] = orientation_diagnostics(w, anchor_indices=[1, 0])
    d["orientation"].update(context)
    d["passes"] = False
    reflection = d["orientation"]["per_quantity"][0]
    assert reflection["status"] == "undefined_diagnostic"
    assert reflection["r_hat"] is None
    assert reflection["ess_bulk"] > 0
    assert reflection["ess_tail"] > 0
    assert reflection["passes"] is False
    validate_orientation_record(model)

    reflection["r_hat"] = 1.0
    with pytest.raises(ValueError, match="orientation.*undefined"):
        validate_orientation_record(model)


@pytest.fixture(scope="module", params=[(3, "haar"), (5, "haar"), (3, "none"), (5, "none")])
def general_k_posterior(request):
    k, refresh = request.param
    times = np.arange(6.0)
    data = {
        "a": {
            "train": {
                "brain": TimeSeries(
                    np.column_stack([np.sin(times + j * 0.6) for j in range(k)]), times
                )
            }
        }
    }
    model = BayesianMultimodalSRM(
        priors=BayesianPriors(noise=Prior.lognormal(-1.0, 0.5)),
        features=k,
        factor_anchors=tuple(("a", "brain", j) for j in range(k)),
        inference="posterior",
        linear_algebra="grouped",
        random_state=648,
        search=SearchConfig(starts=1, maxiter=2),
        sampler=SamplerConfig(
            chains=2,
            warmup=2,
            draws=4,
            max_tree_depth=2,
            mass_matrix="diagonal",
            orientation_refresh=refresh,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    return model


def test_general_k_public_smoke_roundtrip(general_k_posterior, tmp_path, monkeypatch):
    from .test_bayesian_persistence import forbid_inference_fitting

    model = general_k_posterior
    d = model.sampling_diagnostics_
    k = model.problem_.features
    assert d["orientation_probe_schema"] == "qr_entries_squares_v1"
    assert d["orientation"]["anchor_indices"] == list(range(k))
    assert len(d["orientation"]["per_quantity"]) == 1 + 4 * k * k
    assert d["orientation"]["passes"] is False
    assert d["passes"] is False
    workflow.save_model(tmp_path / "fit", model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / "fit")
    assert _archive.same(restored.sampling_diagnostics_, d)
    np.testing.assert_array_equal(restored.parameter_draws_, model.parameter_draws_)


@pytest.mark.parametrize(
    "damage", ["marker_removed", "marker_changed", "schema_removed", "name", "anchor", "occupancy"]
)
@pytest.mark.parametrize("operation", ["save", "load"])
def test_general_k_archive_tampering_rejected(general_k_posterior, tmp_path, damage, operation):
    model = copy.deepcopy(general_k_posterior)
    from multimodalsrm.bayesian.posterior_persistence import state

    saved = state(model, None) if operation == "load" else None
    d = model.sampling_diagnostics_
    o = d["orientation"]
    if damage == "marker_removed":
        del d["orientation_probe_schema"]
    elif damage == "marker_changed":
        d["orientation_probe_schema"] = "bad"
    elif damage == "schema_removed":
        del o["probe_schema"]
    elif damage == "name":
        o["per_quantity"][1]["name"] = "bad"
    elif damage == "anchor":
        o["anchor_indices"] = [0, 1]
    else:
        o["chains"][0]["reflection_transitions"] += 1
    if operation == "load":
        _archive.write(tmp_path / "bad", saved, version=3)
        with pytest.raises(ValueError, match="orientation"):
            workflow.load_model(tmp_path / "bad")
    else:
        with pytest.raises(ValueError, match="orientation"):
            workflow.save_model(tmp_path / "bad", model)


def test_general_k_historical_not_assessed_preserves_semantics(general_k_posterior, tmp_path):
    from dataclasses import replace

    model = copy.deepcopy(general_k_posterior)
    model.sampler_config_ = replace(model.sampler_config_, orientation_refresh="none")
    model.sampler = model.sampler_config_
    model.configuration_["sampler"]["orientation_refresh"] = "none"
    model.training_fit_["configuration"]["sampler"]["orientation_refresh"] = "none"
    d = model.sampling_diagnostics_
    d.pop("orientation_probe_schema")
    d["orientation_refresh"] = "none"
    d["orientation"] = dict(
        applicable=False,
        status="not_assessed",
        passes=None,
        reason="Haar refresh requires exactly two factors",
    )
    d["passes"] = d["raw_passes"]
    workflow.save_model(tmp_path / "legacy", model)
    loaded, _ = workflow.load_model(tmp_path / "legacy")
    assert _archive.same(loaded.sampling_diagnostics_, d)
    assert loaded.sampling_diagnostics_["orientation"]["passes"] is None


def test_general_k_context_default_and_invalid_anchors(general_k_posterior):
    from multimodalsrm.bayesian.blocks import ParameterSubspace
    from multimodalsrm.bayesian.fitting import _orientation_context

    model = general_k_posterior
    space = ParameterSubspace(model.problem_, model.map_parameters_, blocks=model.sample_blocks)
    context = _orientation_context(space, None)
    assert context["anchor_indices"] == list(range(model.problem_.features))
    assert context["anchor_source"] == "first_k_loading_rows"
    with pytest.raises(ValueError, match="distinct loading rows"):
        _orientation_context(space, list(model.problem_.keys[:2]))


def test_general_k_record_cannot_downgrade_with_marker_retained(general_k_posterior):
    from multimodalsrm.bayesian.orientation_persistence import validate_orientation_record

    model = copy.deepcopy(general_k_posterior)
    model.sampling_diagnostics_["orientation"] = dict(
        applicable=False,
        status="not_assessed",
        passes=None,
        reason="Haar refresh requires exactly two factors",
    )
    with pytest.raises(ValueError, match="orientation"):
        validate_orientation_record(model)
