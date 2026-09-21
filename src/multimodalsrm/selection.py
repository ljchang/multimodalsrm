"""Native-run selection with explicit failure and information boundaries.

``ModelComparisonResult`` retains score/prediction rows and execution errors.
``ModelSelectionResult`` adds strict candidate summaries, candidate/fold fit
diagnostics (including preprocessing snapshots), and separate refit diagnostics.
``NestedCrossValidationResult`` retains aligned outer models, folds and inner
selections; a failed outer model occupies its slot as ``None``. Numeric undefined
metrics remain NaN. No recovery truth or outer observations enter selection.
"""

import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone

from ._parallel import ordered_tasks
from .data import normalize_data
from .validation import LeaveOneRunOut, common_valid_mask, score_prediction


@dataclass
class ModelComparisonResult:
    scores: list
    predictions: list
    errors: list
    warnings: list = field(default_factory=list)


@dataclass
class ModelSelectionResult:
    best_name: str | None
    best_estimator: object | None
    scores: list
    summary: list
    folds: list
    predictions: list
    source: str
    metric: str
    status: str
    refit_diagnostics: dict | None = None


@dataclass
class NestedCrossValidationResult:
    scores: list
    summary: list
    models: list
    folds: list
    selections: list
    predictions: list


def _named(models, method):
    if (
        not isinstance(models, Mapping)
        or not models
        or any(not isinstance(name, str) or not name for name in models)
        or any(not callable(getattr(model, method, None)) for model in models.values())
    ):
        raise ValueError(f"models/candidates must map nonempty names to estimators with {method}.")


def _requests(targets, sources, metric="r2", evaluation_weights=None):
    if isinstance(sources, str):
        raise ValueError("sources must be distinct within/across/both values.")
    sources = tuple(sources)
    if (
        not sources
        or len(set(sources)) != len(sources)
        or any(s not in ("within", "across", "both") for s in sources)
    ):
        raise ValueError("sources must be distinct within/across/both values.")
    if metric not in ("r2", "mse", "correlation"):
        raise ValueError("metric must be r2, mse, or correlation.")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must map participants to modality lists.")
    modalities = set()
    for names in targets.values():
        if (
            isinstance(names, str)
            or not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("targets need nonempty lists of distinct modality names.")
        modalities.update(names)
    if evaluation_weights is not None:
        if not isinstance(evaluation_weights, Mapping) or set(evaluation_weights) != modalities:
            raise ValueError("evaluation_weights must cover exactly the targeted modalities.")
        weights = np.asarray(list(evaluation_weights.values()), float)
        if (
            not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("evaluation_weights must be finite, nonnegative and sum to one.")
    return sources


def _runs(partition):
    if not isinstance(partition, Mapping) or any(
        not isinstance(runs, Mapping) for runs in partition.values()
    ):
        raise ValueError("Each run partition must be a participant/run mapping.")
    return {run for runs in partition.values() for run in runs}


def _partition(data, names):
    return {
        subject: {r: mods for r, mods in runs.items() if r in names}
        for subject, runs in data.items()
        if any(r in names for r in runs)
    }


def _folds(data, cv):
    available, result = _runs(data), []
    splitter = LeaveOneRunOut() if cv is None else cv
    # Give the splitter copied dictionaries; only its named runs are trusted.
    for train, test in splitter.split(normalize_data(data)):
        train_runs, test_runs = _runs(train), _runs(test)
        if not train_runs or not test_runs:
            raise ValueError("Run partitions must have nonempty train and test sides.")
        if (train_runs | test_runs) - available:
            raise ValueError("Run partitions contain unknown run names.")
        if train_runs & test_runs:
            raise ValueError("Run partitions require disjoint train/test run names.")
        result.append((_partition(data, train_runs), _partition(data, test_runs)))
    if not result:
        raise ValueError("The splitter produced no folds.")
    return result


def _queries(data, targets):
    query, entries = {}, []
    for subject, names in targets.items():
        for run, modalities in data.get(subject, {}).items():
            for modality in names:
                if modality in modalities:
                    truth = modalities[modality]
                    query.setdefault(subject, {}).setdefault(run, {})[modality] = truth.times
                    entries.append((subject, run, modality, truth))
    if not entries:
        raise ValueError("The evaluation partition has no observed requested targets.")
    return query, entries


def _warning_records(caught):
    return [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]


def _comparison(models, data, targets, sources, failures=None):
    query, entries = _queries(data, targets)
    failures = {} if failures is None else dict(failures)
    predictions, errors, warning_rows = {}, [], []
    for name, model in models.items():
        if name in failures:
            errors.append({"model": name, "source": None, **failures[name]})
            continue
        for source in sources:
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    output = model.predict(data, targets=targets, source=source, times=query)
                current = {}
                for subject, run, modality, truth in entries:
                    prediction = output[subject][run][modality]
                    if not np.array_equal(prediction.times, truth.times):
                        raise ValueError(
                            "Prediction timestamps must match native evaluation queries."
                        )
                    if prediction.values.shape != truth.values.shape:
                        raise ValueError("Prediction feature shape must match the target.")
                    valid = common_valid_mask(truth.mask, [prediction.valid])
                    if not np.isfinite(prediction.values[valid]).all():
                        raise ValueError("Valid prediction entries must be finite.")
                    current[subject, run, modality] = prediction
                predictions[name, source] = current
            except Exception as error:
                failure = {
                    "status": "prediction_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                failures[name] = failure
                errors.append({"model": name, "source": source, **failure})
            warning_rows.extend(
                {"model": name, "source": source, **row} for row in _warning_records(caught)
            )
    rows, records = [], []
    for subject, run, modality, truth in entries:
        key = subject, run, modality
        support = common_valid_mask(truth.mask, [p[key].valid for p in predictions.values()])
        for name in models:
            for source in sources:
                prediction = predictions.get((name, source), {}).get(key)
                identity = dict(
                    model=name,
                    subject=subject,
                    run=run,
                    modality=modality,
                    source=source,
                )
                failure = failures.get(name)
                mask = support if failure is None else np.zeros_like(truth.mask)
                values = (
                    prediction.values
                    if prediction is not None
                    else np.full_like(truth.values, np.nan)
                )
                rows.append(
                    {
                        **identity,
                        **score_prediction(truth.values, values, mask),
                        "status": "ok" if failure is None else failure["status"],
                        **({"error": failure["error"]} if failure else {}),
                    }
                )
                records.append(
                    {
                        **identity,
                        "truth": truth.values.copy(),
                        "times": truth.times.copy(),
                        "observation_mask": truth.mask.copy(),
                        "common_mask": support.copy(),
                        "values": values.copy(),
                        "valid": prediction.valid.copy()
                        if prediction is not None
                        else np.zeros(len(truth.times), bool),
                        "used_sources": deepcopy(prediction.metadata.get("used_sources", []))
                        if prediction is not None
                        else [],
                        "status": rows[-1]["status"],
                        **({"error": failure["error"]} if failure else {}),
                    }
                )
    return ModelComparisonResult(rows, records, errors, warning_rows)


def compare_models(models, data, *, targets, sources=("within", "across", "both")):
    """Compare fitted models on native observations and common valid support.

    The full requested target union is excluded from every conditional solve.
    A model with any failed source prediction has undefined scores for every
    source; successful raw predictions are retained for inspection.
    """
    _named(models, "predict")
    sources = _requests(targets, sources)
    return _comparison(models, normalize_data(data), targets, sources)


def _fit(estimator, train, affinity, **identity):
    model, caught = None, []
    record = dict(**identity, train_runs=sorted(_runs(train)), status="ok")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = clone(estimator)
            model.fit(train, affinity=affinity)
    except Exception as error:
        record.update(status="fit_failed", error=f"{type(error).__name__}: {error}")
    history = getattr(model, "objective_history_", [])
    record.update(
        converged=bool(getattr(model, "converged_", False)),
        objective=float(history[-1]) if len(history) else np.nan,
        restarts=deepcopy(getattr(model, "restart_diagnostics_", [])),
        warnings=_warning_records(caught),
        preprocessing=deepcopy(getattr(model, "preprocessing_", {})),
    )
    return (model if record["status"] == "ok" else None), record


def _aggregate(rows, metric, evaluation_weights):
    positive = [
        row for row in rows if evaluation_weights is None or evaluation_weights[row["modality"]] > 0
    ]
    if not positive or any(not np.isfinite(row[metric]) for row in positive):
        return np.nan
    if evaluation_weights is None:
        return float(np.mean([row[metric] for row in positive]))
    # Missing positive-weight modalities cannot silently renormalize the score.
    counts = {m: sum(row["modality"] == m for row in positive) for m in evaluation_weights}
    if any(weight > 0 and counts[m] == 0 for m, weight in evaluation_weights.items()):
        return np.nan
    return float(
        sum(
            evaluation_weights[row["modality"]] * row[metric] / counts[row["modality"]]
            for row in positive
        )
    )


def select_model(
    candidates,
    data,
    *,
    targets,
    source="both",
    metric="r2",
    cv=None,
    affinity=None,
    evaluation_weights=None,
    refit=True,
    n_jobs=1,
):
    """Select independent clones by strict inner run validation, then refit.

    Every positive-weight expected row must have a finite selected metric.
    Exact ties use candidate names. Finite unfinished optimization remains
    eligible, with warnings and stopping records separate from score validity.
    ``status`` is ok, no_eligible_candidate, or refit_failed. A failed refit
    retains the selected name but returns no deployable estimator.
    ``n_jobs`` parallelizes independent folds, preserving common support across
    candidates within each fold. Inner native restart workers are suppressed.
    """
    _named(candidates, "fit")
    _requests(targets, (source,), metric, evaluation_weights)
    data = normalize_data(data)
    partitions = _folds(data, cv)
    for _, test in partitions:
        _queries(test, targets)
    scores, folds, predictions = [], [], []
    results = ordered_tasks(
        _select_fold,
        [
            (candidates, train, test, fold, targets, source, affinity)
            for fold, (train, test) in enumerate(partitions)
        ],
        n_jobs,
    )
    for local_scores, local_folds, local_predictions in results:
        scores.extend(local_scores)
        folds.extend(local_folds)
        predictions.extend(local_predictions)
    summary = []
    for name in candidates:
        rows = [row for row in scores if row["candidate"] == name]
        diagnostics = [row for row in folds if row["candidate"] == name]
        score = _aggregate(rows, metric, evaluation_weights)
        failed = [row["fold"] for row in diagnostics if row["status"] != "ok"]
        summary.append(
            dict(
                candidate=name,
                eligible=bool(np.isfinite(score) and not failed),
                score=score,
                expected_rows=len(rows),
                defined_rows=sum(np.isfinite(row[metric]) for row in rows),
                failed_folds=failed,
                unfinished_folds=[
                    row["fold"]
                    for row in diagnostics
                    if row["status"] != "fit_failed" and not row["converged"]
                ],
            )
        )
    eligible = [row for row in summary if row["eligible"]]
    best = (
        min(
            eligible,
            key=lambda row: (
                row["score"] if metric == "mse" else -row["score"],
                row["candidate"],
            ),
        )
        if eligible
        else None
    )
    result = ModelSelectionResult(
        best["candidate"] if best else None,
        None,
        scores,
        summary,
        folds,
        predictions,
        source,
        metric,
        "ok" if best else "no_eligible_candidate",
    )
    if best is not None and refit:
        result.best_estimator, result.refit_diagnostics = _fit(
            candidates[result.best_name], data, affinity, candidate=result.best_name
        )
        if result.best_estimator is None:
            result.status = "refit_failed"
    return result


def _outer_summary(rows, sources, evaluation_weights):
    summary = []
    for source in sources:
        selected = [row for row in rows if row["source"] == source]
        record = dict(
            source=source,
            n_scores=len(selected),
            expected_rows=len(selected),
            failed_folds=sorted({row["fold"] for row in selected if row["status"] != "ok"}),
        )
        for metric in ("r2", "mse", "correlation"):
            record[metric] = _aggregate(selected, metric, evaluation_weights)
            record[f"{metric}_defined_scores"] = sum(np.isfinite(row[metric]) for row in selected)
        summary.append(record)
    return summary


def nested_cross_validate(
    candidates,
    data,
    *,
    targets,
    source="both",
    metric="r2",
    sources=("within", "across", "both"),
    outer_cv=None,
    inner_cv=None,
    affinity=None,
    evaluation_weights=None,
    n_jobs=1,
):
    """Select only inside each outer training partition and evaluate fresh runs.

    Models, folds and selections share outer-fold positions, including failures.
    Outer scores retain every expected target/source row; an incomplete metric
    has an undefined summary rather than a favorable available-case average.
    ``n_jobs`` parallelizes outer folds only; each worker selects/refits using
    serial native inner folds and restarts. All execution defaults to serial.
    """
    _named(candidates, "fit")
    sources = _requests(targets, sources, metric, evaluation_weights)
    _requests(targets, (source,), metric, evaluation_weights)
    data = normalize_data(data)
    if outer_cv is None and len(_runs(data)) < 3:
        raise ValueError("Default nested cross-validation requires at least three runs.")
    partitions = _folds(data, outer_cv)
    # Validate/materialize all nested input partitions before any numerical fits.
    inner_partitions = []
    for train, test in partitions:
        _queries(test, targets)
        inner = _folds(train, inner_cv)
        for _, validation in inner:
            _queries(validation, targets)
        inner_partitions.append(inner)
    scores, models, folds, selections, predictions = [], [], [], [], []
    results = ordered_tasks(
        _nested_fold,
        [
            (
                candidates,
                train,
                test,
                inner,
                fold,
                targets,
                source,
                metric,
                sources,
                affinity,
                evaluation_weights,
            )
            for fold, ((train, test), inner) in enumerate(zip(partitions, inner_partitions))
        ],
        n_jobs,
    )
    for local_scores, model, record, selection, local_predictions in results:
        scores.extend(local_scores)
        models.append(model)
        folds.append(record)
        selections.append(selection)
        predictions.extend(local_predictions)
    return NestedCrossValidationResult(
        scores,
        _outer_summary(scores, sources, evaluation_weights),
        models,
        folds,
        selections,
        predictions,
    )


class _MaterializedFolds:
    def __init__(self, folds):
        self.folds = folds

    def split(self, data):
        return iter(self.folds)


def _select_fold(candidates, train, test, fold, targets, source, affinity):
    scores, folds, predictions = [], [], []
    models, failures, diagnostics = {}, {}, {}
    for name, estimator in candidates.items():
        model, record = _fit(
            estimator,
            train,
            affinity,
            candidate=name,
            fold=fold,
            test_runs=sorted(_runs(test)),
        )
        models[name], diagnostics[name] = model, record
        if model is None:
            failures[name] = {key: record[key] for key in ("status", "error")}
    compared = _comparison(models, test, targets, (source,), failures)
    for error in compared.errors:
        diagnostics[error["model"]].update(status=error["status"], error=error["error"])
    for warning in compared.warnings:
        diagnostics[warning["model"]]["warnings"].append(warning)
    folds.extend(diagnostics.values())
    scores.extend({**row, "candidate": row["model"], "fold": fold} for row in compared.scores)
    predictions.extend(
        {**row, "candidate": row["model"], "fold": fold} for row in compared.predictions
    )
    return scores, folds, predictions


def _nested_fold(
    candidates,
    train,
    test,
    inner,
    fold,
    targets,
    source,
    metric,
    sources,
    affinity,
    evaluation_weights,
):
    scores, models, folds, selections, predictions = [], [], [], [], []
    selection = select_model(
        candidates,
        train,
        targets=targets,
        source=source,
        metric=metric,
        cv=_MaterializedFolds(inner),
        affinity=affinity,
        evaluation_weights=evaluation_weights,
    )
    model, name = selection.best_estimator, selection.best_name
    failure = (
        None
        if model is not None
        else {
            name: {
                "status": selection.status,
                "error": (selection.refit_diagnostics or {}).get(
                    "error", "No candidate has complete finite validation scores."
                ),
            }
        }
    )
    compared = _comparison({name: model}, test, targets, sources, failure)
    record = dict(
        fold=fold,
        candidate=name,
        train_runs=sorted(_runs(train)),
        test_runs=sorted(_runs(test)),
        status=selection.status,
        converged=bool(getattr(model, "converged_", False)),
        errors=compared.errors,
        warnings=compared.warnings,
        refit_diagnostics=selection.refit_diagnostics,
    )
    if model is not None and compared.errors:
        record["status"] = "prediction_failed"
    folds.append(record)
    models.append(model)
    selections.append(selection)
    scores.extend({**row, "fold": fold, "candidate": name} for row in compared.scores)
    predictions.extend({**row, "fold": fold, "candidate": name} for row in compared.predictions)
    return scores, models[0], folds[0], selections[0], predictions
