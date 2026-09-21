"""Joint new-participant calibration and isolated posterior readouts."""

from collections.abc import Mapping

import numpy as np

from ..data import normalize_data
from .posterior_updates import (
    finish_update,
    prepare_update,
    reference_state,
    restore_adapter,
)


def prepare_calibration(model, data, *, search=None, sampler=None, random_state=None):
    """Prepare one new mapping against frozen original support and statistics."""
    reference = reference_state(model)
    adapter = restore_adapter(reference["adapter"])
    data = normalize_data(data)
    if len(data) != 1:
        raise ValueError("calibrate exactly one new participant at a time")
    participant = next(iter(data))
    if participant in adapter._training_data:
        raise ValueError("calibration requires a new participant ID")
    runs = data[participant]
    for run, modalities in runs.items():
        if run not in adapter.domains_:
            raise ValueError("calibration run IDs must name the same training stimulus")
        lo, hi = adapter.domains_[run]
        for modality, series in modalities.items():
            if modality not in adapter.responses_:
                raise ValueError("calibration modality has no trained group filter")
            observed = series.times[series.mask.any(axis=1)]
            if observed.min() < lo or observed.max() > hi:
                raise ValueError("calibration observations extend outside the training run")
    statistics = {}
    for modality in dict.fromkeys(m for mods in runs.values() for m in mods):
        series = [(r, mods[modality]) for r, mods in runs.items() if modality in mods]
        values = np.concatenate([ts.values for _, ts in series])
        mask = np.concatenate(
            [
                ts.mask & adapter._support(ts.times, modality, adapter.domains_[r])[:, None]
                for r, ts in series
            ]
        )
        count = mask.sum(axis=0)
        if np.any(count < model.features + 1):
            raise ValueError(
                "each calibration feature needs at least features + 1 eligible observations"
            )
        mean = np.where(mask, values, 0).sum(axis=0) / count
        scale = np.sqrt(np.where(mask, (values - mean) ** 2, 0).sum(axis=0) / count)
        constant = scale <= np.finfo(float).eps
        scale[constant] = 1.0
        if not adapter.standardize:
            mean, scale = np.zeros_like(mean), np.ones_like(scale)
        statistics[modality] = dict(mean=mean, scale=scale, constant_features=constant)
    return prepare_update(
        model,
        data,
        kind="calibration",
        participant=participant,
        added_preprocessing={participant: statistics},
        search=search,
        sampler=sampler,
        random_state=random_state,
    )


def calibrate_posterior(
    model, data, *, search=None, sampler=None, random_state=None, progress=None
):
    """Sample the joint group and new mapping posterior on shared training runs."""
    return finish_update(
        prepare_calibration(model, data, search=search, sampler=sampler, random_state=random_state),
        progress=progress,
    )


def prepare_participant(model, data, *, participant, sampler=None, random_state=None):
    """Select a participant before validating, copying or preprocessing payloads."""
    reference = reference_state(model)
    adapter = restore_adapter(reference["adapter"])
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty participant mapping")
    if participant not in adapter._training_data:
        raise ValueError("participant mapping is unavailable; calibrate it first")
    if participant not in data:
        raise ValueError("participant is absent from the supplied observations")
    selected = normalize_data({participant: data[participant]})
    return prepare_update(
        model,
        selected,
        kind="participant",
        participant=participant,
        sampler=sampler,
        random_state=random_state,
    )


def condition_participants(model, data, *, sampler=None, random_state=None, progress=None):
    """Return separate joint updates, each with only its own new-run evidence."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty participant mapping")
    return {
        participant: finish_update(
            prepare_participant(
                model,
                data,
                participant=participant,
                sampler=sampler,
                random_state=random_state,
            ),
            progress=progress,
        )
        for participant in data
    }
