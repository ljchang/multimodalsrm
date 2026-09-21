"""Descriptive held-out alignment scores, one run and common query clock at a time.

No fitting, sign selection, time warping, or inferential p-values occur here.
TimeSeries inputs must already be in comparable coordinates; the caller owns
their train/test provenance. Bayesian inputs must come from independent transform.
"""

from collections.abc import Mapping

import numpy as np

from .data import TimeSeries


def _inputs(series):
    from .bayesian.results import GaussianMixtureSeries

    if not isinstance(series, Mapping) or len(series) < 2:
        raise ValueError("supply at least two participants for one run")
    values, masks, kinds, conventions, runs = [], [], set(), [], set()
    references, calibrated, readouts = [], False, set()
    times, features = None, None
    for subject, item in series.items():
        if isinstance(item, TimeSeries):
            mask = item.mask
            kinds.add("raw")
        elif isinstance(item, GaussianMixtureSeries):
            meta = item.metadata
            if (
                meta.get("independent_subject_inference") is not True
                or meta.get("conditioning_subject") != subject
                or meta.get("parameter_source")
                not in (
                    "original_training_MAP",
                    "group_MAP_plus_participant_calibration_MAP",
                )
                or meta.get("parameter_refits") != 0
                or meta.get("quantity") != "unfiltered_shared_latent"
            ):
                raise ValueError(
                    "Bayesian alignment requires independent subject transform results"
                )
            mask = np.broadcast_to(item.valid[:, None], item.values.shape)
            kinds.add("independent_training_MAP")
            conventions.append(meta.get("reference_convention"))
            runs.add(meta.get("conditioning_run"))
            readouts.add(meta.get("alignment_readout", "gp"))
            calibrated |= (
                meta.get("parameter_source") == "group_MAP_plus_participant_calibration_MAP"
            )
            references.append(meta.get("training_reference_id"))
        else:
            raise ValueError("inputs must be TimeSeries or independent GaussianMixtureSeries")
        if times is not None and not np.array_equal(times, item.times):
            raise ValueError("participants must have the same query clock")
        if features is not None and features != item.values.shape[1]:
            raise ValueError("participants must have the same feature count")
        times, features = item.times, item.values.shape[1]
        values.append(item.values)
        masks.append(mask)
    if len(kinds) != 1:
        raise ValueError("do not mix raw and transformed coordinates in one evaluation")
    if len(runs) > 1 or any(c != conventions[0] for c in conventions):
        raise ValueError("transformed participants must share a run and reference convention")
    if len(readouts) > 1:
        raise ValueError("transformed participants must share the same alignment readout")
    if calibrated and (
        any(not isinstance(r, str) or not r for r in references) or len(set(references)) != 1
    ):
        raise ValueError("calibrated participants must share the same frozen group reference")
    valid = np.stack(masks).all(axis=0)
    # Masked values can be NaN or infinite. They must not enter any arithmetic.
    values = np.where(valid[None, :, :], np.stack(values), 0.0)
    source = "independent_group_and_calibration_MAP" if calibrated else kinds.pop()
    return list(series), values, valid, times, source


def _mean(values):
    """Average finite values without overflowing the sum across participants."""
    scale = np.max(np.abs(values), axis=0)
    divisor = np.where(scale == 0, 1, scale)
    return np.clip(np.mean(values / divisor, axis=0), -1, 1) * scale


def _unit(vector):
    # Scale first to avoid overflow. Near-constant signals are undefined, not noise wins.
    if not np.isfinite(vector).all():
        raise FloatingPointError("nonfinite alignment vector")
    scale = np.max(np.abs(vector))
    if scale == 0:
        return None
    centered = vector / scale - np.mean(vector / scale)
    norm = np.linalg.norm(centered)
    if norm <= 1e-12 * np.sqrt(len(vector)):
        return None
    return centered / norm


def temporal_isc(series):
    """Per-feature temporal Pearson ISC against the leave-one-participant-out mean.

    Use each feature's intersection of valid rows across ALL participants.
    At least three rows and nonconstant query/reference signals are required.
    Aggregate Fisher-z means are undefined if any participant is undefined;
    individual outcomes and support counts are always retained. This is temporal
    ISC, not correlation over features at each time (pattern ISC).
    """
    subjects, values, valid, times, source = _inputs(series)
    output = {}
    for s, subject in enumerate(subjects):
        correlations, reasons = [], []
        for f in range(values.shape[2]):
            rows = valid[:, f]
            reason, correlation = None, None
            if rows.sum() < 3:
                reason = "insufficient_support"
            else:
                query = _unit(values[s, rows, f])
                reference = _unit(_mean(np.delete(values, s, axis=0)[:, rows, f]))
                if query is None or reference is None:
                    reason = "constant_signal"
                else:
                    correlation = float(np.clip(query @ reference, -1, 1))
            correlations.append(correlation)
            reasons.append(reason)
        output[subject] = dict(correlations=correlations, reasons=reasons)
    means = []
    for f in range(values.shape[2]):
        scores = [output[s]["correlations"][f] for s in subjects]
        means.append(
            None
            if any(r is None for r in scores)
            else float(np.tanh(np.mean(np.arctanh(np.clip(scores, -1 + 1e-12, 1 - 1e-12)))))
        )
    return dict(
        metric="temporal_isc",
        reference="leave_one_subject_out_mean",
        input_source=source,
        n_subjects=len(subjects),
        n_times=len(times),
        n_features=values.shape[2],
        n_common_valid_per_feature=valid.sum(axis=0).tolist(),
        subjects=output,
        mean_fisher_z_correlation_per_feature=means,
        fisher_clip=1e-12,
    )


def _window_unit(window, centering):
    if centering == "global":
        return _unit(window.ravel())
    if not np.isfinite(window).all():
        raise FloatingPointError("nonfinite alignment window")
    scale = np.max(np.abs(window))
    if scale == 0:
        return None
    scaled = window / scale
    centered = (scaled - scaled.mean(axis=0)).ravel()
    norm = np.linalg.norm(centered)
    if norm <= 1e-12 * np.linalg.norm(scaled):
        return None
    return centered / norm


def time_segment_matching(series, *, window_size, centering="global"):
    """Match each participant's time windows to the other participants' mean.

    Query times must be regularly spaced. Fully valid windows are flattened
    over time and features, then compared with Pearson correlation. Competing
    windows that partially overlap the query are excluded; its true match stays.
    Ties within 1e-12 receive fractional credit. Chance is 1/candidate count per
    query, not 1/total windows. Constant vectors or fewer than two candidates
    make a query undefined. An undefined query makes aggregate accuracy None.
    No windows cross masked rows; do not concatenate independent runs upstream.

    ``centering="feature"`` is a companion score: subtract each feature's
    within-window temporal mean before the normalized dot product. It is
    invariant to a common orthogonal feature rotation (up to roundoff), but
    not arbitrary feature scaling. The historical scalar-centering metric
    remains the default; the companion has a distinct metric identifier.
    """
    if centering not in ("global", "feature"):
        raise ValueError("centering must be global or feature")
    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or window_size < 3
    ):
        raise ValueError("window_size must be an integer of at least three samples")
    subjects, values, valid, times, source = _inputs(series)
    delta = np.diff(times)
    if len(delta) and not np.allclose(delta, delta[0], rtol=1e-7, atol=1e-10):
        raise ValueError("time-segment matching requires a regular query clock")
    rows = valid.all(axis=1)
    possible = max(0, len(times) - window_size + 1)
    starts = np.array([i for i in range(possible) if rows[i : i + window_size].all()], dtype=int)
    output = {}
    for s, subject in enumerate(subjects):
        reference = _mean(np.delete(values, s, axis=0))
        queries = [_window_unit(values[s, i : i + window_size], centering) for i in starts]
        targets = [_window_unit(reference[i : i + window_size], centering) for i in starts]
        target_valid = np.array([t is not None for t in targets], bool)
        target_matrix = np.array(
            [t if t is not None else np.zeros(window_size * values.shape[2]) for t in targets]
        )
        records = []
        # One query at a time avoids a quadratic score matrix in memory.
        for index, (i, query) in enumerate(zip(starts, queries)):
            candidates = (np.abs(starts - i) >= window_size) | (starts == i)
            count = int(candidates.sum())
            reason, credit, ties = None, None, None
            if count < 2:
                reason = "insufficient_candidates"
            elif query is None or not target_valid[candidates].all():
                reason = "constant_signal"
            else:
                scores = target_matrix @ query
                winners = candidates & np.isclose(
                    scores, scores[candidates].max(), rtol=0, atol=1e-12
                )
                ties = int(winners.sum())
                credit = float(winners[index]) / ties
            records.append(
                dict(
                    start_index=int(i),
                    start_time=float(times[i]),
                    n_candidates=count,
                    credit=credit,
                    chance=1 / count if count >= 2 else None,
                    n_ties=ties,
                    reason=reason,
                )
            )
        complete = bool(records) and all(r["credit"] is not None for r in records)
        output[subject] = dict(
            accuracy=float(np.mean([r["credit"] for r in records])) if complete else None,
            chance=float(np.mean([r["chance"] for r in records]))
            if records and all(r["chance"] is not None for r in records)
            else None,
            n_queries=len(records),
            n_scored_queries=sum(r["credit"] is not None for r in records),
            reason=None if complete else ("undefined_queries" if records else "no_valid_windows"),
            queries=records,
        )
    return dict(
        metric="time_segment_matching"
        if centering == "global"
        else "time_segment_matching_feature_centered",
        reference="leave_one_subject_out_mean",
        input_source=source,
        n_subjects=len(subjects),
        n_times=len(times),
        n_features=values.shape[2],
        window_size=int(window_size),
        query_interval=float(delta[0]) if len(delta) else None,
        window_starts=starts.tolist(),
        n_common_valid_rows=int(rows.sum()),
        n_excluded_windows=possible - len(starts),
        subjects=output,
        tie_tolerance=1e-12,
    )
