"""Real native-time fits detect selection leakage and incomplete denominators."""

from copy import deepcopy

import numpy as np
import pytest

import multimodalsrm as mm


def api(name):
    assert hasattr(mm, name), f"Missing public selection API: {name}"
    return getattr(mm, name)


def native_data(n_runs=3):
    data = {}
    for subject, brain_map, rating_map in (("a", [1.2, -0.7], 0.9), ("b", [-0.4, 1.8], 1.3)):
        data[subject] = {}
        for i, run in enumerate(("film-A", "film-B", "film-C", "film-D")[:n_runs]):
            modalities = {}
            for modality, times, mapping in (
                ("brain", np.arange(1, 19, 1.1) + (0.15 if subject == "b" else 0), brain_map),
                ("rating", np.arange(0, 20, 0.7), [rating_map]),
            ):
                latent = np.sin(times / 1.9 + i * 0.35) + 0.35 * np.cos(times / 0.8 - i)
                values = latent[:, None] * np.asarray(mapping)[None, :] + i * 0.1
                mask = np.ones(values.shape, bool)
                if modality == "brain":
                    mask[3:5, 1] = False
                    values[~mask] = np.nan
                modalities[modality] = mm.TimeSeries(values, times, mask)
            data[subject][run] = modalities
    return data


def estimator(**updates):
    params = dict(
        features=1,
        latent_dt=0.7,
        latent_pooling="population",
        latent_strength=0.2,
        temporal_strength=0.001,
        max_iter=35,
        tol=1e-4,
        init="spectral",
        random_state=41,
    )
    params.update(updates)
    return mm.MultimodalSRM(**params)


def subset(data, runs):
    return {
        s: {r: mods for r, mods in entries.items() if r in runs}
        for s, entries in data.items()
        if any(r in runs for r in entries)
    }


class RunFolds:
    def __init__(self, pairs):
        self.pairs = pairs
        self.calls = 0

    def split(self, data):
        self.calls += 1
        for train, test in self.pairs:
            yield subset(data, train), subset(data, test)


def expected_mean(data, runs, subject="a", modality="brain"):
    series = [data[subject][r][modality] for r in runs]
    values = np.concatenate([ts.values for ts in series])
    mask = np.concatenate([ts.mask for ts in series])
    return np.where(mask, values, 0).sum(0) / mask.sum(0)


def test_validation_prefers_finite_regularization_and_refits_all_development_runs():
    # Break caught: training-objective selection, reversed metric direction, or
    # reusing the last inner fit instead of a fresh full-development fit.
    select = api("select_model")
    data = native_data()
    candidates = {"oversmoothed": estimator(temporal_strength=1e6), "suitable": estimator()}
    cv = RunFolds([(["film-A", "film-B"], ["film-C"]), (["film-A", "film-C"], ["film-B"])])
    result = select(candidates, data, targets={"a": ["brain"]}, cv=cv)
    assert result.best_name == "suitable"
    assert result.status == "ok"
    assert cv.calls == 1
    assert not hasattr(candidates["suitable"], "loadings_")
    assert not hasattr(candidates["oversmoothed"], "loadings_")
    np.testing.assert_allclose(
        result.best_estimator.preprocessing_["a"]["brain"]["mean"],
        expected_mean(data, ["film-A", "film-B", "film-C"]),
    )
    assert set(result.best_estimator.run_grids_) == {"film-A", "film-B", "film-C"}
    for fold in result.folds:
        np.testing.assert_allclose(
            fold["preprocessing"]["a"]["brain"]["mean"], expected_mean(data, fold["train_runs"])
        )
        assert np.isfinite(fold["objective"])
        assert fold["restarts"]
    assert all(row["expected_rows"] == 2 for row in result.summary)
    assert all(row["eligible"] for row in result.summary)
    assert max(row["score"] for row in result.summary) > 0.8
    assert not np.allclose(
        result.best_estimator.loadings_["a"]["brain"], result.best_estimator.loadings_["b"]["brain"]
    )


def test_compare_preserves_native_common_support_and_full_target_exclusion():
    # Break caught: candidate-specific scoring masks, resampling truth, or
    # conditioning on another requested target in the same union.
    compare = api("compare_models")
    data = native_data()
    models = {
        "identity": estimator().fit(subset(data, ["film-A", "film-B"])),
        "wide": estimator(
            responses={
                "brain": mm.Response(mm.Gaussian(width=0.3), estimate=False),
                "rating": mm.Response(mm.Identity(), estimate=False),
            }
        ).fit(subset(data, ["film-A", "film-B"])),
    }
    evaluation = subset(data, ["film-C"])
    targets = {"a": ["brain"], "b": ["brain"]}
    result = compare(models, evaluation, targets=targets)
    assert result.errors == []
    assert len(result.scores) == len(result.predictions) == 12
    records = [r for r in result.predictions if r["subject"] == "a"]
    truth = data["a"]["film-C"]["brain"]
    expected = truth.mask.copy()
    # Wide fixed Gaussian support is [-1.8, 1.8], conditioning domain [0, 19.6].
    expected &= ((truth.times >= 1.8) & (truth.times <= 17.8))[:, None]
    for record in records:
        np.testing.assert_array_equal(record["times"], truth.times)
        np.testing.assert_array_equal(record["truth"], truth.values)
        np.testing.assert_array_equal(record["observation_mask"], truth.mask)
        np.testing.assert_array_equal(record["common_mask"], expected)
        assert record["used_sources"]
        assert all(source[2] == "rating" for source in record["used_sources"])
    assert not np.array_equal(records[0]["valid"], records[-1]["valid"])
    poisoned = deepcopy(evaluation)
    original = poisoned["b"]["film-C"]["brain"]
    poisoned["b"]["film-C"]["brain"] = mm.TimeSeries(
        original.values * 100 + 50, original.times, original.mask
    )
    changed = compare(models, poisoned, targets=targets)
    for before, after in zip(result.predictions, changed.predictions):
        np.testing.assert_array_equal(before["values"], after["values"])


def test_outer_target_poisoning_changes_score_but_not_inner_selection_or_refit():
    # Break caught: outer observations passed into selection, preprocessing,
    # initialization, kernel estimation, or any full-development refit.
    nested = api("nested_cross_validate")
    data = native_data(4)
    outer = RunFolds([(["film-A", "film-B", "film-C"], ["film-D"])])
    candidates = {"suitable": estimator(), "oversmoothed": estimator(temporal_strength=1e6)}
    clean = nested(candidates, data, targets={"a": ["brain"]}, outer_cv=outer)
    poisoned = deepcopy(data)
    old = data["a"]["film-D"]["brain"]
    poisoned["a"]["film-D"]["brain"] = mm.TimeSeries(old.values * 30 + 100, old.times, old.mask)
    changed = nested(candidates, poisoned, targets={"a": ["brain"]}, outer_cv=outer)
    assert clean.selections[0].best_name == changed.selections[0].best_name == "suitable"
    assert clean.selections[0].summary == changed.selections[0].summary
    assert clean.models[0] is not changed.models[0]
    for subject in data:
        for modality in ("brain", "rating"):
            np.testing.assert_array_equal(
                clean.models[0].loadings_[subject][modality],
                changed.models[0].loadings_[subject][modality],
            )
            assert (
                clean.models[0].subject_kernels_[subject][modality]
                == changed.models[0].subject_kernels_[subject][modality]
            )
            for stat in ("mean", "scale"):
                np.testing.assert_array_equal(
                    clean.models[0].preprocessing_[subject][modality][stat],
                    changed.models[0].preprocessing_[subject][modality][stat],
                )
    assert min(row["r2"] for row in clean.scores) > 0.7
    assert max(row["r2"] for row in changed.scores) < 0
    assert all(row["candidate"] == "suitable" and row["fold"] == 0 for row in clean.scores)
    for before, after in zip(clean.predictions, changed.predictions):
        np.testing.assert_array_equal(before["values"], after["values"])


def test_failed_candidates_keep_complete_denominators_and_inspectable_errors():
    # Break caught: dropping failed fits or allowing favorable partial scores.
    result = api("select_model")(
        {"invalid": estimator(features=0), "valid": estimator()},
        native_data(),
        targets={"a": ["brain"]},
        refit=False,
    )
    assert result.best_name == "valid"
    assert result.best_estimator is None
    summary = {row["candidate"]: row for row in result.summary}
    assert summary["invalid"]["expected_rows"] == 3
    assert summary["invalid"]["defined_rows"] == 0
    assert summary["invalid"]["failed_folds"] == [0, 1, 2]
    assert not summary["invalid"]["eligible"]
    failed = [row for row in result.scores if row["candidate"] == "invalid"]
    assert len(failed) == 3 and all(np.isnan(row["r2"]) for row in failed)
    diagnostics = [row for row in result.folds if row["candidate"] == "invalid"]
    assert all(row["status"] == "fit_failed" and "features" in row["error"] for row in diagnostics)


def test_prediction_failure_invalidates_all_sources_for_that_model():
    # Zero coupling cannot predict a subject from across-subject observations,
    # although its within prediction works. Both rows must show failure.
    compare = api("compare_models")
    data = native_data()
    train, test = subset(data, ["film-A", "film-B"]), subset(data, ["film-C"])
    models = {
        "disconnected": estimator(latent_strength=0).fit(train),
        "usable": estimator().fit(train),
    }
    result = compare(models, test, targets={"a": ["brain"]}, sources=("within", "across"))
    assert any(
        error["model"] == "disconnected"
        and error["source"] == "across"
        and "coupling" in error["error"]
        for error in result.errors
    )
    failed = [row for row in result.scores if row["model"] == "disconnected"]
    assert len(failed) == 2 and all(np.isnan(row["r2"]) for row in failed)
    assert all(row["status"] == "prediction_failed" for row in failed)
    assert all(np.isfinite(row["r2"]) for row in result.scores if row["model"] == "usable")


def test_all_invalid_candidates_and_outer_folds_remain_explicit():
    candidates = {"bad": estimator(features=0)}
    targets = {"a": ["brain"]}
    selected = api("select_model")(candidates, native_data(), targets=targets)
    assert selected.status == "no_eligible_candidate"
    assert selected.best_name is selected.best_estimator is None
    result = api("nested_cross_validate")(candidates, native_data(), targets=targets)
    assert result.models == [None, None, None]
    assert len(result.folds) == len(result.selections) == 3
    assert len(result.scores) == len(result.predictions) == 9
    assert all(row["status"] == "no_eligible_candidate" for row in result.folds)
    assert all(row["candidate"] is None and np.isnan(row["r2"]) for row in result.scores)
    assert all(
        row["expected_rows"] == 3 and row["r2_defined_scores"] == 0 and np.isnan(row["r2"])
        for row in result.summary
    )


def constant_target_data():
    data = native_data()
    for runs in data.values():
        for modalities in runs.values():
            old = modalities["brain"]
            modalities["brain"] = mm.TimeSeries(np.ones_like(old.values) * 7, old.times, old.mask)
    return data


def test_constant_targets_are_ineligible_for_r2_but_mse_remains_defined():
    # Break caught: treating undefined R2 as zero, or requiring R2 for MSE selection.
    select = api("select_model")
    candidates, data = {"constant": estimator()}, constant_target_data()
    r2 = select(candidates, data, targets={"a": ["brain"]}, refit=False)
    assert r2.status == "no_eligible_candidate"
    assert r2.summary[0]["expected_rows"] == 3 and r2.summary[0]["defined_rows"] == 0
    mse = select(candidates, data, targets={"a": ["brain"]}, metric="mse", refit=False)
    assert mse.best_name == "constant"
    assert mse.summary[0]["score"] == pytest.approx(0, abs=1e-10)


def test_zero_weight_undefined_modality_does_not_contaminate_selection():
    # Break caught: 0 * NaN in weighted aggregation.
    select = api("select_model")
    data = constant_target_data()
    result = select(
        {"usable": estimator()},
        data,
        targets={"a": ["brain", "rating"]},
        evaluation_weights={"brain": 0, "rating": 1},
        refit=False,
    )
    assert result.best_name == "usable"
    assert np.isfinite(result.summary[0]["score"])
    assert result.summary[0]["expected_rows"] == 6
    assert result.summary[0]["defined_rows"] == 3


@pytest.mark.parametrize("metric", ["r2", "mse", "correlation"])
def test_exact_ties_use_lexicographic_names_independent_of_insertion_order(metric):
    # Break caught: insertion-order tie breaks or wrong MSE direction.
    select = api("select_model")
    data = native_data()
    cv = RunFolds([(["film-A", "film-B"], ["film-C"])])
    for names in (("zeta", "alpha"), ("alpha", "zeta")):
        result = select(
            {name: estimator() for name in names},
            data,
            targets={"a": ["brain"]},
            metric=metric,
            cv=cv,
            refit=False,
        )
        assert result.best_name == "alpha"
        assert result.summary[0]["score"] == result.summary[1]["score"]
    if metric == "mse":
        result = select(
            {"oversmoothed": estimator(temporal_strength=1e6), "suitable": estimator()},
            data,
            targets={"a": ["brain"]},
            metric=metric,
            cv=cv,
            refit=False,
        )
        assert result.best_name == "suitable"


def test_unconverged_finite_fits_keep_warnings_and_remain_eligible():
    # Break caught: equating a stopping flag with prediction eligibility.
    result = api("select_model")(
        {"unfinished": estimator(max_iter=1, tol=1e-12)},
        native_data(),
        targets={"a": ["brain"]},
        refit=False,
    )
    assert result.best_name == "unfinished"
    assert result.summary[0]["unfinished_folds"] == [0, 1, 2]
    assert all(
        not fold["converged"] and fold["warnings"] and fold["restarts"] for fold in result.folds
    )


def test_unfinished_optimization_remains_counted_when_prediction_also_fails():
    # Break caught: letting prediction status erase independent fit stopping diagnostics.
    result = api("select_model")(
        {"unfinished": estimator(max_iter=1, tol=1e-12, latent_strength=0)},
        native_data(),
        targets={"a": ["brain"]},
        source="across",
        refit=False,
    )
    assert result.status == "no_eligible_candidate"
    assert result.summary[0]["failed_folds"] == [0, 1, 2]
    assert result.summary[0]["unfinished_folds"] == [0, 1, 2]


@pytest.mark.parametrize(
    "weights",
    [{"brain": 2}, {}, {"brain": 0.5}, {"brain": np.nan}, {"brain": -1}, {"brain": 1, "rating": 0}],
)
def test_invalid_evaluation_weights_raise_input_errors(weights):
    with pytest.raises(ValueError, match="evaluation_weights"):
        api("select_model")(
            {"valid": estimator()},
            native_data(),
            targets={"a": ["brain"]},
            evaluation_weights=weights,
        )


@pytest.mark.parametrize("sources", [(), ("both", "both"), ("invalid",), "both"])
def test_duplicate_or_invalid_sources_raise_input_errors(sources):
    with pytest.raises(ValueError, match="sources"):
        api("compare_models")(
            {"valid": estimator()}, native_data(), targets={"a": ["brain"]}, sources=sources
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "invalid"},
        {"metric": "rmse"},
        {"targets": {}},
        {"targets": {"a": "brain"}},
        {"targets": {"a": ["brain", "brain"]}},
    ],
)
def test_invalid_selection_requests_raise_instead_of_becoming_candidate_failures(kwargs):
    options = dict(targets={"a": ["brain"]})
    options.update(kwargs)
    with pytest.raises(ValueError):
        api("select_model")({"valid": estimator()}, native_data(), **options)


class RawFolds:
    def __init__(self, pairs):
        self.pairs = pairs

    def split(self, data):
        return iter(self.pairs)


@pytest.mark.parametrize("case", ["unknown", "overlap", "empty_train", "empty_test", "no_folds"])
def test_invalid_run_partitions_are_input_errors(case):
    data = native_data()
    train, test = subset(data, ["film-A", "film-B"]), subset(data, ["film-C"])
    if case == "unknown":
        test = {"a": {"foreign-film": data["a"]["film-C"]}}
    elif case == "overlap":
        test = subset(data, ["film-B", "film-C"])
    elif case == "empty_train":
        train = {}
    elif case == "empty_test":
        test = {}
    cv = RawFolds([] if case == "no_folds" else [(train, test)])
    with pytest.raises(ValueError, match="run|fold|partition"):
        api("select_model")({"valid": estimator()}, data, targets={"a": ["brain"]}, cv=cv)


def test_custom_splitter_cannot_inject_training_or_validation_observations():
    # Break caught: trusting splitter-supplied values/subject subsets instead of
    # reconstructing every participant from original named runs.
    select = api("select_model")
    data = native_data()
    train, test = subset(data, ["film-A", "film-B"]), subset(data, ["film-C"])
    changed_train, changed_test = deepcopy(train), deepcopy(test)
    for partition in (changed_train, changed_test):
        del partition["b"]
        for modalities in partition["a"].values():
            old = modalities["brain"]
            modalities["brain"] = mm.TimeSeries(old.values * 100 + 20, old.times, old.mask)
    clean = select(
        {"valid": estimator()},
        data,
        targets={"a": ["brain"]},
        cv=RawFolds([(train, test)]),
        refit=False,
    )
    changed = select(
        {"valid": estimator()},
        data,
        targets={"a": ["brain"]},
        cv=RawFolds([(changed_train, changed_test)]),
        refit=False,
    )
    assert clean.summary == changed.summary
    for subject in ("a", "b"):
        np.testing.assert_array_equal(
            clean.folds[0]["preprocessing"][subject]["brain"]["mean"],
            changed.folds[0]["preprocessing"][subject]["brain"]["mean"],
        )


def test_failed_full_development_refit_is_not_reported_as_a_deployable_model():
    # Inner folds omit a genuine unfit singleton-duration run; full refit must
    # include it and retain its numerical input failure without returning a model.
    data = native_data()
    for subject, runs in data.items():
        runs["singleton"] = {
            "brain": mm.TimeSeries([[1, 2]], [0]),
            "rating": mm.TimeSeries([[1]], [0]),
        }
    result = api("select_model")(
        {"valid": estimator()},
        data,
        targets={"a": ["brain"]},
        cv=RunFolds([(["film-A", "film-B"], ["film-C"])]),
    )
    assert result.best_name == "valid"
    assert result.best_estimator is None
    assert result.status == "refit_failed"
    assert "duration" in result.refit_diagnostics["error"]
    assert "singleton" in result.refit_diagnostics["train_runs"]


def test_nested_default_requires_three_runs_and_validates_inner_partitions():
    nested = api("nested_cross_validate")
    with pytest.raises(ValueError, match="three|3|two|2"):
        nested({"valid": estimator()}, native_data(2), targets={"a": ["brain"]})
    data = native_data()
    injected = RawFolds([(subset(data, ["film-A"]), subset(data, ["film-C"]))])
    with pytest.raises(ValueError, match="unknown|run"):
        nested(
            {"valid": estimator()},
            data,
            targets={"a": ["brain"]},
            outer_cv=RunFolds([(["film-A", "film-B"], ["film-C"])]),
            inner_cv=injected,
        )


def test_one_failed_inner_fold_disqualifies_an_otherwise_finite_candidate():
    # Break caught: averaging only finite inner rows and dropping one failed fit.
    data = native_data()
    for run in ("film-B", "film-C"):
        old = data["a"][run]["brain"]
        mask = old.mask.copy()
        mask[:, 1] = False
        data["a"][run]["brain"] = mm.TimeSeries(old.values, old.times, mask)
    result = api("select_model")(
        {"partly_fittable": estimator()}, data, targets={"a": ["brain"]}, refit=False
    )
    assert result.status == "no_eligible_candidate"
    assert result.summary[0]["failed_folds"] == [0]
    assert result.summary[0]["defined_rows"] == 2
    assert result.summary[0]["expected_rows"] == 3
    assert np.isnan(result.summary[0]["score"])


def test_one_failed_outer_fold_cannot_disappear_from_summary_denominators():
    # Outer film-D has no across-subject donor. Film-C is fully predictable;
    # available-case averaging would incorrectly report its score as complete.
    data = native_data(4)
    del data["b"]["film-D"]
    outer = RunFolds(
        [(["film-A", "film-B", "film-C"], ["film-D"]), (["film-A", "film-B", "film-D"], ["film-C"])]
    )
    result = api("nested_cross_validate")(
        {"valid": estimator()}, data, targets={"a": ["brain"]}, outer_cv=outer
    )
    assert [fold["status"] for fold in result.folds] == ["prediction_failed", "ok"]
    assert result.folds[0]["errors"]
    assert all(
        row["expected_rows"] == 2
        and row["r2_defined_scores"] == 1
        and np.isnan(row["r2"])
        and row["failed_folds"] == [0]
        for row in result.summary
    )
    assert len(result.scores) == len(result.predictions) == 6


def test_modality_weights_average_within_modality_and_do_not_reuse_fit_weights():
    # Break caught: row weighting that favors modalities with more target pairs,
    # silently normalizing weights, or using the estimator's fit weights.
    result = api("select_model")(
        {"valid": estimator(modality_weights={"brain": 0.9, "rating": 0.1})},
        native_data(),
        targets={"a": ["brain"], "b": ["brain", "rating"]},
        evaluation_weights={"brain": 0.25, "rating": 0.75},
        refit=False,
    )
    brain = [row["r2"] for row in result.scores if row["modality"] == "brain"]
    rating = [row["r2"] for row in result.scores if row["modality"] == "rating"]
    assert len(brain) == 6 and len(rating) == 3
    expected = sum(brain) / 24 + sum(rating) / 4
    assert result.summary[0]["score"] == pytest.approx(expected)
    assert abs(expected - sum(brain + rating) / 9) > 1e-4


def test_fixed_affinity_is_passed_through_to_independent_candidate_clones():
    # Break caught: dropping the supplied graph or changing it during selection.
    graph = {"a": {"b": 0.35}, "b": {"a": 0.35}}
    original = deepcopy(graph)
    result = api("select_model")(
        {"graph": estimator(latent_pooling="neighborhood")},
        native_data(),
        targets={"a": ["brain"]},
        affinity=graph,
    )
    assert result.best_name == "graph"
    assert result.best_estimator.affinity_ == graph == original
