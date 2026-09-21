"""Joint calibration and isolated participant posterior targets."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import _archive

from .test_bayesian_multifactor import independent, log_prior
from .test_bayesian_posterior_updates import prepared


def calibration(model):
    data = {"newperson": copy.deepcopy(model.training_data_["a"])}
    ts = data["newperson"]["train"]["aux"]
    times = np.linspace(ts.times[0], ts.times[-1], 23)
    values = np.stack([np.interp(times, ts.times, column) for column in ts.values.T], axis=1)
    data["newperson"]["train"]["aux"] = TimeSeries(values, times)
    return data


def physical(result, model):
    old = dict(zip(model.problem_.names, model.map_parameters_))
    x = result.problem_.initial.copy()
    for i, name in enumerate(result.problem_.names):
        replacement = (
            (name[0], "a", *name[2:]) if len(name) > 2 and name[1] == "newperson" else name
        )
        x[i] = old.get(replacement, x[i])
    return x


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_calibration_is_joint_gaussian_with_cross_participant_covariance(k, algebra):
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
    )

    model = prepared(k, algebra)
    before = copy.deepcopy(
        (model.training_data_, model.adapter_.preprocessing_, model.parameter_draws_)
    )
    result = prepare_calibration(model, calibration(model))
    p = result.problem_
    x = physical(result, model)
    C, _, offsets, idx, _ = independent(p, x)
    old = np.array([key[0] != "newperson" for key in p.systems["train"].keys])
    assert np.max(abs(C[np.ix_(old, ~old)])) > 0.01
    expected = -multivariate_normal.logpdf(
        p.systems["train"].values, mean=offsets[idx], cov=C
    ) - log_prior(p, x)
    assert_allclose(p.objective(x), expected, atol=1e-7)
    assert set(p.systems) == {"train"}
    assert set(model.problem_.names) < set(p.names)
    assert result._factor_anchor_keys_ == model._factor_anchor_keys_
    assert result.adapter_.domains_ == model.adapter_.domains_
    for subject, stats in model.adapter_.preprocessing_.items():
        assert _archive.same(stats, result.adapter_.preprocessing_[subject])
    assert _archive.same(
        before,
        (model.training_data_, model.adapter_.preprocessing_, model.parameter_draws_),
    )


@pytest.mark.parametrize("damage", ["existing", "multiple", "modality", "run", "domain", "few"])
def test_invalid_calibration_rejected(damage):
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
    )

    model = prepared()
    data = calibration(model)
    if damage == "existing":
        data = {"a": data["newperson"]}
    elif damage == "multiple":
        data["other"] = copy.deepcopy(data["newperson"])
    elif damage == "modality":
        data["newperson"]["train"]["alien"] = data["newperson"]["train"].pop("brain")
    elif damage == "run":
        data["newperson"]["elsewhere"] = data["newperson"].pop("train")
    else:
        ts = data["newperson"]["train"]["brain"]
        data["newperson"]["train"]["brain"] = TimeSeries(
            ts.values[:2] if damage == "few" else ts.values,
            ts.times[:2] if damage == "few" else ts.times + 50,
        )
    with pytest.raises(
        ValueError,
        match="new participant|one new|modality|run|observations|training stimulus",
    ):
        prepare_calibration(model, data)


def test_new_preprocessing_uses_original_domain_and_masks():
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
    )

    model = prepared()
    data = calibration(model)
    ts = data["newperson"]["train"]["aux"]
    # Narrow calibration coverage must not narrow the training support envelope.
    keep = (ts.times >= 4.5) & (ts.times <= 12)
    data["newperson"]["train"]["aux"] = TimeSeries(ts.values[keep], ts.times[keep], ts.mask[keep])
    result = prepare_calibration(model, data)
    stats = result.adapter_.preprocessing_["newperson"]["aux"]
    assert_array_equal(stats["mean"], np.zeros(2))
    assert_array_equal(stats["scale"], np.ones(2))
    assert (
        sum(key[:2] == ("newperson", "aux") for key in result.problem_.systems["train"].keys) >= 8
    )


def test_participant_selection_never_reads_other_payload_and_replaces_batch():
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_participant,
    )

    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("other participant touched")

    model = prepared()
    ts = model.training_data_["a"]["train"]["brain"]
    data = {"a": {"new": {"brain": ts}}, "b": Poison()}
    first = prepare_participant(model, data, participant="a")
    assert set(first.problem_.systems) == {"train", "new"}
    assert set(first.posterior_update_["evidence"]) == {"a"}
    first.parameter_draws_ = model.parameter_draws_
    first.configuration_ = {"inference": "posterior"}
    second = prepare_participant(
        first, {"a": {"later": {"brain": ts}}, "b": Poison()}, participant="a"
    )
    assert set(second.problem_.systems) == {"train", "later"}


def test_calibration_dense_grouped_gradient_parity():
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
    )

    models = [prepared(3, a) for a in ("dense", "grouped")]
    results = [prepare_calibration(m, calibration(m)) for m in models]
    outputs = [r.problem_.value_gradient(physical(r, m)) for r, m in zip(results, models)]
    assert_allclose(outputs[0][0], outputs[1][0], atol=1e-7)
    assert_allclose(outputs[0][1], outputs[1][1], atol=1e-7)


@pytest.mark.parametrize("damage", ["unknown", "run", "modality", "features"])
def test_invalid_independent_participant_rejected(damage):
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_participant,
    )

    model = prepared()
    ts = model.training_data_["a"]["train"]["brain"]
    participant = "unknown" if damage == "unknown" else "a"
    run = "train" if damage == "run" else "new"
    modality = "alien" if damage == "modality" else "brain"
    if damage == "features":
        ts = TimeSeries(ts.values[:, :1], ts.times)
    with pytest.raises(ValueError, match="mapping|overlap|feature count"):
        prepare_participant(model, {participant: {run: {modality: ts}}}, participant=participant)


def test_other_participant_changes_cannot_change_target_identity():
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_participant,
    )

    model = prepared()
    a = model.training_data_["a"]["train"]["brain"]
    b = model.training_data_["b"]["train"]["brain"]
    first = prepare_participant(
        model, {"a": {"new": {"brain": a}}, "b": {"new": {"brain": b}}}, participant="a"
    )
    second = prepare_participant(
        model, {"a": {"new": {"brain": a}}, "b": object()}, participant="a"
    )
    assert first.problem_._posterior_update_id == second.problem_._posterior_update_id
    assert _archive.same(first.problem_.systems, second.problem_.systems)
    own_b = prepare_participant(model, {"a": object(), "b": {"new": {"brain": b}}}, participant="b")
    assert all(key[0] == "b" for key in own_b.problem_.systems["new"].keys)


def test_public_calibration_and_independent_posterior_queries():
    from multimodalsrm.bayesian.posterior_updates import reference_state

    from ..reference.participant_conditioning import reference

    model = prepared(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(model.training_data_)
        before = model.parameter_draws_.copy()
        result = model.calibrate_posterior(calibration(model), random_state=77)
        ts = result.training_data_["newperson"]["train"]["brain"]
        individual = result.condition_participants(
            {"newperson": {"new": {"brain": ts}}}, random_state=78
        )["newperson"]
    assert_array_equal(model.parameter_draws_, before)
    assert result.parameter_draws_.shape[-1] > model.parameter_draws_.shape[-1]
    assert set(reference_state(result)["adapter"]["fitted"]["_training_data"]) == {
        "a",
        "b",
        "c",
        "newperson",
    }
    assert set(
        individual.posterior_update_["reference"]["adapter"]["fitted"]["_training_data"]
    ) == {"a", "b", "c", "newperson"}
    assert individual.sampling_diagnostics_["orientation_refresh"] == "haar"
    assert (
        individual.configuration_["diagnostic_scope"] == "reference_plus_one_participant_new_runs"
    )
    query = individual.infer_latent(times={"new": np.array([7.0, 9.0])}, max_draws=4)["new"]
    assert np.isfinite(query.variance).all()
    assert_allclose(
        query.variance,
        query.component_variances.mean(axis=0) + query.component_means.var(axis=0),
        atol=1e-12,
    )
    # Reference helper uses train; a copied problem exposes only the new-run system.
    p = copy.copy(individual.problem_)
    p.systems = {"train": p.systems["new"]}
    for i, (c, d) in enumerate(query.metadata["parameter_draw_indices"]):
        x = individual.parameter_draws_[c, d]
        _, mean, cov = reference(p, x, np.array([7.0, 9.0]))
        nw = len(p.keys) * p.features
        w = x[:nw].reshape(-1, p.features)
        anchor = [p.keys.index(key) for key in individual._factor_anchor_keys_]
        q, r = np.linalg.qr(w[anchor].T)
        q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
        assert_allclose(query.component_means[i], mean @ q, atol=1e-7)
        assert_allclose(
            query.component_variances[i],
            np.diagonal(q.T @ cov @ q, axis1=-2, axis2=-1),
            atol=1e-7,
        )
