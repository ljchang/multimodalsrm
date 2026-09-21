"""Validated native-rate multimodal observations (time by feature)."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def readonly_array(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def validate_times(times, *, allow_empty=False):
    result = readonly_array(times)
    if result.ndim != 1 or (not allow_empty and not result.size):
        raise ValueError("times must be a nonempty one-dimensional vector")
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    return result


@dataclass(frozen=True, eq=False)
class TimeSeries:
    values: np.ndarray
    times: np.ndarray
    mask: np.ndarray | None = None

    def __post_init__(self):
        values = readonly_array(self.values)
        times = validate_times(self.times)
        if values.ndim != 2 or 0 in values.shape or len(values) != len(times):
            raise ValueError("values must be nonempty time-by-feature with one row per timestamp")
        mask = np.ones(values.shape, dtype=bool) if self.mask is None else np.asarray(self.mask)
        if mask.dtype != np.dtype(bool) or mask.shape != values.shape:
            raise ValueError("mask must be Boolean and match values shape")
        if not mask.any():
            raise ValueError("fully masked series must be omitted")
        if not np.isfinite(values[mask]).all():
            raise ValueError("observed values must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "mask", readonly_array(mask, bool))


def normalize_data(data):
    """Copy dictionaries, preserving immutable series; reject mixed nesting."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty subject mapping")
    styles = set()
    for subject, entries in data.items():
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"{subject}: subject entries must be nonempty mappings")
        for value in entries.values():
            styles.add(
                "short"
                if isinstance(value, TimeSeries)
                else "explicit"
                if isinstance(value, Mapping)
                else "invalid"
            )
    if len(styles) != 1 or "invalid" in styles:
        raise ValueError(
            "mixed nesting is not allowed; use consistent subject/run/modality mappings"
        )
    out, counts = {}, {}
    for subject, entries in data.items():
        runs = {"run-01": entries} if "short" in styles else entries
        out[subject] = {}
        for run, modalities in runs.items():
            if not isinstance(modalities, Mapping) or not modalities:
                raise ValueError("empty runs must be omitted")
            out[subject][run] = {}
            for modality, series in modalities.items():
                if not isinstance(series, TimeSeries):
                    raise ValueError("each modality must be a TimeSeries")
                key = (subject, modality)
                count = series.values.shape[1]
                if key in counts and counts[key] != count:
                    raise ValueError(f"feature count changes across runs for {key}")
                counts[key] = count
                out[subject][run][modality] = series
    return out
