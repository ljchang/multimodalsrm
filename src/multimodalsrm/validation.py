"""External run-based evaluation with common-support, original-unit scores."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from ._parallel import ordered_tasks
from .data import normalize_data


class LeaveOneRunOut:
    """Hold out a named stimulus run across all participants.

    This splitter understands the nested multimodal contract, unlike generic
    sklearn row splitters. At least two distinct run identifiers are required.
    Participants without observations in a partition are omitted there.
    """

    def split(self, data):
        data = normalize_data(data)
        runs = list(dict.fromkeys(r for subject in data.values() for r in subject))
        if len(runs) < 2:
            raise ValueError("LeaveOneRunOut requires at least two distinct runs.")
        for held_out in runs:
            train, test = {}, {}
            for subject, subject_runs in data.items():
                kept = {r: modalities for r, modalities in subject_runs.items() if r != held_out}
                if kept:
                    train[subject] = kept
                if held_out in subject_runs:
                    test[subject] = {held_out: subject_runs[held_out]}
            yield train, test

    def get_n_splits(self, data):
        return sum(1 for _ in self.split(data))


@dataclass
class CrossValidationResult:
    """Unaggregated score rows, explicit summaries, and independent fold fits."""

    scores: list
    summary: list
    models: list
    folds: list


def common_valid_mask(observed, valid_masks):
    """Intersect feature masks with row- or feature-level prediction validity."""
    mask = np.asarray(observed, dtype=bool).copy()
    if mask.ndim != 2:
        raise ValueError("The observation mask must be time by feature.")
    for valid in valid_masks:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape == (len(mask),):
            valid = valid[:, None]
        if valid.shape not in (mask.shape, (len(mask), 1)):
            raise ValueError("Prediction validity does not match the target shape.")
        mask &= valid
    return mask


def score_prediction(observed, predicted, mask):
    """Feature-mean correlation/R2 and entry-mean MSE in original units.

    Correlation needs two nonconstant vectors. R2 needs at least two observations
    and a nonconstant target. Undefined features are counted, and a metric with
    no defined features is NaN, never zero. Masked entries are never evaluated.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    mask = np.asarray(mask, bool)
    if observed.ndim != 2 or predicted.shape != observed.shape or mask.shape != observed.shape:
        raise ValueError("Scores require matching time-by-feature arrays and mask.")
    if not (np.isfinite(observed[mask]).all() and np.isfinite(predicted[mask]).all()):
        raise ValueError("Valid scoring entries must be finite.")
    correlations, r2s = [], []
    undefined_features = {"correlation": 0, "r2": 0}
    for feature in range(observed.shape[1]):
        selected = mask[:, feature]
        y, p = observed[selected, feature], predicted[selected, feature]
        y_variance = float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0
        p_variance = float(np.sum((p - p.mean()) ** 2)) if len(p) else 0.0
        if len(y) >= 2 and y_variance > 0 and p_variance > 0:
            correlations.append(float(np.corrcoef(y, p)[0, 1]))
        else:
            undefined_features["correlation"] += 1
        if len(y) >= 2 and y_variance > 0:
            r2s.append(float(1 - np.sum((y - p) ** 2) / y_variance))
        else:
            undefined_features["r2"] += 1
    count = int(mask.sum())
    scores = {
        "correlation": float(np.mean(correlations)) if correlations else np.nan,
        "mse": float(np.mean((observed[mask] - predicted[mask]) ** 2)) if count else np.nan,
        "r2": float(np.mean(r2s)) if r2s else np.nan,
        "n_observations": count,
        "n_timestamps": int(mask.any(axis=1).sum()),
        "undefined_features": undefined_features,
    }
    scores["undefined"] = [
        name for name in ("correlation", "mse", "r2") if not np.isfinite(scores[name])
    ]
    return scores


def _summarize(rows, sources, evaluation_weights):
    result = []
    for source in sources:
        selected = [r for r in rows if r["source"] == source]
        summary = {"source": source, "n_scores": len(selected)}
        for metric in ("correlation", "mse", "r2"):
            usable = [r for r in selected if np.isfinite(r[metric])]
            if evaluation_weights is None:
                weights = np.ones(len(usable))
            else:
                # Assign modality proportions first; average equally across
                # that modality's available participant/run/fold score rows.
                counts = {m: sum(r["modality"] == m for r in usable) for m in evaluation_weights}
                weights = np.array(
                    [evaluation_weights[r["modality"]] / counts[r["modality"]] for r in usable]
                )
            summary[metric] = (
                float(np.average([r[metric] for r in usable], weights=weights))
                if weights.sum() > 0
                else np.nan
            )
            summary[f"{metric}_defined_scores"] = len(usable)
        result.append(summary)
    return result


def cross_validate(
    estimator,
    data,
    *,
    targets,
    sources=("within", "across", "both"),
    cv=None,
    affinity=None,
    evaluation_weights=None,
    n_jobs=1,
):
    """Clone and fit on training runs, then score explicit targets on test runs.

    All compared sources use the intersection of their valid prediction entries
    and the target observation mask. Target observations may be supplied to
    ``predict`` for convenience, but its exclusion boundary removes them before
    inference. Fit weights are never reused as evaluation weights. With no
    evaluation weights, summary metrics equally average available score rows.
    Unsupported folds raise a clear error; they are not silently discarded.
    ``n_jobs`` parallelizes folds in separate processes (1 by default; -1 uses
    available CPUs). Inner native restart parallelism is suppressed in workers.
    """
    data = normalize_data(data)
    sources = tuple(sources)
    if (
        not sources
        or len(set(sources)) != len(sources)
        or any(s not in ("within", "across", "both") for s in sources)
    ):
        raise ValueError("sources must be distinct within/across/both values.")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("targets must map participant identifiers to modalities.")
    modalities = set()
    for names in targets.values():
        if isinstance(names, str) or not names or len(set(names)) != len(names):
            raise ValueError(
                "Each target participant needs a nonempty list of distinct modalities."
            )
        modalities.update(names)
    if evaluation_weights is not None:
        if set(evaluation_weights) != modalities:
            raise ValueError("evaluation_weights must cover exactly the targeted modalities.")
        weights = np.array(list(evaluation_weights.values()), float)
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("evaluation_weights must be finite, nonnegative and sum to one.")
    rows, models, folds = [], [], []
    splitter = LeaveOneRunOut() if cv is None else cv
    results = ordered_tasks(
        _cross_validate_fold,
        [
            (estimator, train, test, fold, targets, sources, affinity)
            for fold, (train, test) in enumerate(splitter.split(data))
        ],
        n_jobs,
    )
    for local_rows, fitted, record in results:
        rows.extend(local_rows)
        models.append(fitted)
        folds.append(record)
    if not folds:
        raise ValueError("The splitter produced no folds.")
    return CrossValidationResult(rows, _summarize(rows, sources, evaluation_weights), models, folds)


def _cross_validate_fold(estimator, train, test, fold, targets, sources, affinity):
    rows, models, folds = [], [], []
    train, test = normalize_data(train), normalize_data(test)
    train_runs = {r for runs in train.values() for r in runs}
    test_runs = {r for runs in test.values() for r in runs}
    if train_runs & test_runs:
        raise ValueError("Run-based cross-validation requires disjoint train/test run names.")
    fitted = clone(estimator).fit(train, affinity=affinity)
    query, fold_targets = {}, {}
    for subject, names in targets.items():
        for run, available in test.get(subject, {}).items():
            for modality in names:
                if modality in available:
                    query.setdefault(subject, {}).setdefault(run, {})[modality] = available[
                        modality
                    ].times
                    if modality not in fold_targets.setdefault(subject, []):
                        fold_targets[subject].append(modality)
    if not query:
        raise ValueError(f"Fold {fold} has no observed evaluation targets.")
    # Exclude the full requested union, even when a target lacks evaluation
    # observations in this fold. Only times specify which outputs are scored.
    predictions = {
        source: fitted.predict(test, targets=targets, source=source, times=query)
        for source in sources
    }
    for subject, runs in query.items():
        for run, names in runs.items():
            for modality, times in names.items():
                truth = test[subject][run][modality]
                series = [predictions[s][subject][run][modality] for s in sources]
                for prediction in series:
                    if not np.array_equal(prediction.times, times):
                        raise ValueError(
                            "Prediction timestamps must match native evaluation queries."
                        )
                valid = common_valid_mask(truth.mask, [p.valid for p in series])
                for source, prediction in zip(sources, series):
                    rows.append(
                        {
                            "fold": fold,
                            "subject": subject,
                            "run": run,
                            "modality": modality,
                            "source": source,
                            **score_prediction(truth.values, prediction.values, valid),
                        }
                    )
    models.append(fitted)
    folds.append(
        {
            "fold": fold,
            "train_runs": sorted(train_runs),
            "test_runs": sorted(test_runs),
            "converged": bool(fitted.converged_),
        }
    )
    return rows, models[0], folds[0]
