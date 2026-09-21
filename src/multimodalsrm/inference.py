"""Fresh conditional trajectory solves with explicit observation boundaries."""

from collections.abc import Mapping

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .data import TimeSeries, normalize_data, validate_times
from .grouping import latent_groups
from .kernels import Identity
from .objective import build_grids, make_blocks, solve_latents
from .operators import observation_operator
from .results import SeriesResult


def filter_data(data, keep):
    """Filter identifiers before inspecting excluded series or normalizing features."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty subject mapping")
    result = {}
    styles = set()
    for s, entries in data.items():
        if not isinstance(entries, Mapping):
            raise ValueError("subject entries must be mappings")
        short = any(isinstance(v, TimeSeries) for v in entries.values())
        runs = {"run-01": entries} if short else entries
        for r, mods in runs.items():
            if not isinstance(mods, Mapping):
                raise ValueError("mixed nesting is not allowed")
            selected = {m: ts for m, ts in mods.items() if keep(s, r, m)}
            if selected:
                styles.add("short" if short else "explicit")
                result.setdefault(s, {})[r] = selected
    if len(styles) > 1:
        raise ValueError("mixed nesting is not allowed")
    return normalize_data(result) if result else {}


def condition(model, data):
    if not data:
        raise ValueError("no usable conditioning observations remain after source exclusion")
    for s, runs in data.items():
        for mods in runs.values():
            for m in mods:
                if m not in model.loadings_.get(s, {}):
                    raise ValueError(
                        f"conditioning mapping {s}/{m} is unavailable; calibrate it first"
                    )
    grids, domains = build_grids(data, model.latent_dt)
    blocks = make_blocks(
        data,
        grids,
        domains,
        model.preprocessing_,
        model.subject_kernels_,
        model.responses_,
        model.modality_weights_,
        model.gap_threshold,
    )
    if not blocks:
        raise ValueError("no usable conditioning observations within kernel support")
    z, diagnostics = solve_latents(
        blocks,
        grids,
        model.subjects_,
        model.loadings_,
        model.features,
        model.latent_pooling,
        model.latent_strength,
        model.latent_ridge,
        model.temporal_strength,
        model.affinity_,
    )
    return z, grids, domains, blocks, diagnostics


def require_support(model, subject, run, blocks):
    donors = {b.subject for b in blocks if b.run == run and b.valid.any()}
    if subject in donors:
        return
    if not donors:
        raise ValueError(f"no usable conditioning support for {subject}/{run}")
    if model.latent_pooling == "shared":
        return
    if model.latent_pooling == "components":
        group = next(
            g for g in latent_groups(model.subjects_, "components", model.affinity_) if subject in g
        )
        if not donors.intersection(group):
            raise ValueError(f"{subject}/{run} is disconnected from usable donor support")
        return
    if model.latent_strength <= 0:
        raise ValueError(f"{subject}/{run} needs positive latent coupling to donor support")
    if model.latent_pooling == "population":
        return
    reached, queue = {subject}, [subject]
    while queue:
        s = queue.pop()
        for t, weight in model.affinity_.get(s, {}).items():
            if weight > 0 and t not in reached:
                reached.add(t)
                queue.append(t)
    if not reached.intersection(donors):
        raise ValueError(f"{subject}/{run} is disconnected from usable donor support")


def query_times(times, subject, run, modality, runs, default):
    if times is None:
        return default
    if not isinstance(times, Mapping):
        if len(runs) != 1:
            raise ValueError(
                "common time vector is ambiguous for multiple runs; use run-keyed queries"
            )
        return validate_times(times)
    if subject in times and isinstance(times[subject], Mapping):
        value = times[subject].get(run)
        if isinstance(value, Mapping):
            value = value.get(modality)
    else:
        value = times.get(run)
    return None if value is None else validate_times(value)


def component_support(model, data, blocks, subject, run, domain):
    """Select output provenance and raw bounds without trimming response support.

    Numerical solve grids remain global. Only exact components restrict their
    output support to selected, positively weighted observations in that group.
    Raw masked timestamps matter here: block.valid already excludes response
    boundaries and would apply convolution support twice if used as a domain.
    """
    if model.latent_pooling != "components":
        return blocks, domain
    group = next(
        g for g in latent_groups(model.subjects_, "components", model.affinity_) if subject in g
    )
    local = [b for b in blocks if b.subject in group and b.run == run]
    raw = [
        ts.times[ts.mask.any(axis=1)]
        for member in group
        for modality, ts in data.get(member, {}).get(run, {}).items()
        if model.modality_weights_.get(modality, 0) > 0
    ]
    raw = [t for t in raw if len(t)]
    bounds = (min(t[0] for t in raw), max(t[-1] for t in raw)) if raw else (np.inf, -np.inf)
    return local, bounds


def coverage_metadata(blocks, run, query, domain, excluded=(), gap_threshold=None):
    local = [b for b in blocks if b.run == run]
    coverage = []
    observed = np.zeros(len(query), bool)
    for b in local:
        t = b.times[b.valid]
        if len(t):
            delta = np.diff(t)
            threshold = (
                gap_threshold
                if gap_threshold is not None
                else (2 * np.median(delta) if len(delta) else np.inf)
            )
            segments = np.split(t, np.flatnonzero(delta > threshold) + 1)
            intervals = [(float(seg[0]), float(seg[-1])) for seg in segments]
            coverage.append(
                {
                    "subject": b.subject,
                    "run": run,
                    "modality": b.modality,
                    "intervals": intervals,
                    "observed_times": t.copy(),
                }
            )
            observed |= np.isin(query, t)
    return {
        "domain": domain,
        "used_sources": [(b.subject, run, b.modality) for b in local],
        "excluded_sources": list(excluded),
        "observation_coverage": coverage,
        "observed_at_query": observed,
        "model_conditioned": True,
        "interpolation": "model-conditioned; gaps are not observed coverage",
    }


def infer_latent(model, data, *, times=None):
    check_is_fitted(model, "loadings_")
    selected = filter_data(data, lambda s, r, m: model.modality_weights_.get(m, 1) > 0)
    z, grids, domains, blocks, diagnostics = condition(model, selected)
    output_runs = {s: list(runs) for s, runs in selected.items()}
    if isinstance(times, Mapping):
        nested = any(isinstance(v, Mapping) for v in times.values())
        if nested and not all(isinstance(v, Mapping) for v in times.values()):
            raise ValueError("time queries must use consistent nesting")
        requested = {r for sruns in times.values() for r in sruns} if nested else set(times)
        if not requested <= set(grids):
            raise ValueError("requested runs have no usable conditioning support")
        if nested and not set(times) <= set(model.subjects_):
            raise ValueError("requested subjects have no fitted mappings; calibrate them first")
        output_runs = (
            {s: list(runs) for s, runs in times.items()}
            if nested
            else {s: list(times) for s in selected}
        )
    output = {}
    for s, runs in output_runs.items():
        for r in runs:
            require_support(model, s, r, blocks)
            query = query_times(times, s, r, None, grids, grids[r])
            if query is None:
                continue
            H, valid = observation_operator(grids[r], query, Identity())
            local, domain = component_support(model, selected, blocks, s, r, domains[r])
            valid &= (query >= domain[0]) & (query <= domain[1])
            meta = coverage_metadata(local, r, query, domain, gap_threshold=model.gap_threshold)
            meta["solver"] = diagnostics
            output.setdefault(s, {})[r] = SeriesResult(H @ z[s][r], query, valid, meta)
    return output


def predict(model, data, *, targets, source="within", times=None):
    check_is_fitted(model, "loadings_")
    if source not in ("within", "across", "both"):
        raise ValueError("source must be within, across, or both")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must name subjects and modality lists")
    union = set()
    for s, modalities in targets.items():
        if isinstance(modalities, str) or not modalities:
            raise ValueError("targets must contain nonempty modality lists")
        for m in modalities:
            if m not in model.loadings_.get(s, {}):
                raise ValueError(f"target mapping {s}/{m} is unavailable; calibrate it first")
            union.add((s, m))
    output = {}
    for target, modalities in targets.items():
        requested_runs = None
        if isinstance(times, Mapping):
            nested = any(isinstance(v, Mapping) for v in times.values())
            if nested:
                target_queries = times.get(target, {})
                requested_runs = {
                    r
                    for r, queries in target_queries.items()
                    if not isinstance(queries, Mapping) or any(m in queries for m in modalities)
                }
            else:
                requested_runs = set(times)
            if not requested_runs:
                continue
        selected = filter_data(
            data,
            lambda s, r, m: (
                (s, m) not in union
                and (source != "within" or s == target)
                and (source != "across" or s != target)
                and model.modality_weights_.get(m, 1) > 0
            ),
        )
        z, grids, domains, blocks, diagnostics = condition(model, selected)
        if requested_runs is not None and not requested_runs <= set(grids):
            raise ValueError("requested runs have no usable conditioning support")
        for r, grid in grids.items():
            if requested_runs is not None and r not in requested_runs:
                continue
            require_support(model, target, r, blocks)
            local, domain = component_support(model, selected, blocks, target, r, domains[r])
            for m in modalities:
                query = query_times(times, target, r, m, grids, grid)
                if query is None:
                    continue
                support = model.responses_[m].support_envelope()
                H, valid = observation_operator(
                    grid, query, model.subject_kernels_[target][m], support
                )
                valid &= (query - support[1] >= domain[0] - 1e-10) & (
                    query - support[0] <= domain[1] + 1e-10
                )
                stats = model.preprocessing_[target][m]
                values = (H @ z[target][r]) @ model.loadings_[target][m].T
                values = values * stats["scale"] + stats["mean"]
                meta = coverage_metadata(
                    local, r, query, domain, sorted(union), model.gap_threshold
                )
                meta.update(
                    source=source,
                    solver=diagnostics,
                    excluded_subjects=[
                        s
                        for s in model.subjects_
                        if (source == "within" and s != target)
                        or (source == "across" and s == target)
                    ],
                )
                output.setdefault(target, {}).setdefault(r, {})[m] = SeriesResult(
                    values, query, valid, meta
                )
    return output
