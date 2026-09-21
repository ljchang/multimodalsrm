"""Optional preprocessing, capacity inspection and portable result archives.

These helpers never fit the probability model or change its assumptions.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import TimeSeries, normalize_data
from ..kernels import Identity, Response
from .prediction import donor_data
from .results import GaussianMixtureSeries


def _selected(data, targets):
    if targets is None:
        return normalize_data(data)
    if (
        not isinstance(targets, dict)
        or not targets
        or any(isinstance(ms, str) or not ms for ms in targets.values())
    ):
        raise ValueError("targets must map subjects to nonempty modality lists")
    return donor_data(data, {(s, m) for s, ms in targets.items() for m in ms}, "runs")


@dataclass
class TrainingStandardizer:
    """Training-only feature coordinates; uncertainty conditions on these stats.

    Constant or wholly unobserved features must be removed explicitly. This
    prevents arbitrary unit conventions from silently changing prior meaning.
    """

    statistics: dict

    @classmethod
    def fit(cls, training):
        groups = {}
        for s, runs in normalize_data(training).items():
            for mods in runs.values():
                for m, ts in mods.items():
                    groups.setdefault((s, m), []).append(ts)
        stats = {}
        for key, series in groups.items():
            values = np.concatenate([ts.values for ts in series])
            mask = np.concatenate([ts.mask for ts in series])
            if np.any(mask.sum(axis=0) < 2):
                raise ValueError(f"each feature needs two training observations: {key}")
            mean = np.array([values[mask[:, f], f].mean() for f in range(values.shape[1])])
            scale = np.array([values[mask[:, f], f].std() for f in range(values.shape[1])])
            if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
                raise ValueError(f"constant or invalid training feature: {key}")
            stats[key] = {"mean": mean, "scale": scale}
        return cls(stats)

    def transform(self, data, *, targets=None):
        """Exclude donor targets before inspecting or transforming their payloads."""
        result = {}
        for s, runs in _selected(data, targets).items():
            for r, mods in runs.items():
                for m, ts in mods.items():
                    if (s, m) not in self.statistics:
                        raise ValueError(f"unknown training mapping: {s}/{m}")
                    stats = self.statistics[s, m]
                    if ts.values.shape[1] != len(stats["mean"]):
                        raise ValueError(f"feature count differs: {s}/{m}")
                    values = (ts.values - stats["mean"]) / stats["scale"]
                    result.setdefault(s, {}).setdefault(r, {})[m] = TimeSeries(
                        values, ts.times, ts.mask
                    )
        return result

    def inverse_result(self, result, subject, modality):
        """Restore every mixture component, preserving non-Gaussian intervals."""
        stats = self.statistics[subject, modality]
        if result.values.shape[1] != len(stats["mean"]):
            raise ValueError("prediction feature count differs from preprocessing")
        return GaussianMixtureSeries(
            result.component_means * stats["scale"] + stats["mean"],
            result.component_variances * stats["scale"] ** 2,
            result.times,
            result.valid,
            {
                **result.metadata,
                "units": "original_observation_units",
                "preprocessing_uncertainty": "conditional_on_training_statistics",
            },
        )


def capacity_report(data, responses=None, *, max_observations=800, targets=None):
    """Count eligible feature scalars without allocating any covariance matrix.

    Complexity is linear in the input masks. A run pools every participant and
    modality; the limit is not per channel. Storage is for ONE float64 matrix,
    not an estimate of total JAX, gradient or sampling memory.
    """
    if (
        isinstance(max_observations, bool)
        or not isinstance(max_observations, int)
        or max_observations < 1
    ):
        raise ValueError("max_observations must be a positive integer")
    data = _selected(data, targets)
    responses = responses or {}
    domains = {}
    for runs in data.values():
        for r, mods in runs.items():
            for ts in mods.values():
                times = ts.times[ts.mask.any(axis=1)]
                lo, hi = domains.get(r, (np.inf, -np.inf))
                domains[r] = min(lo, times[0]), max(hi, times[-1])
    records = {
        r: {
            "domain": [float(a), float(b)],
            "observed_scalars": 0,
            "eligible_scalars": 0,
            "series": [],
        }
        for r, (a, b) in domains.items()
    }
    for s, runs in data.items():
        for r, mods in runs.items():
            a, b = domains[r]
            if b <= a:
                raise ValueError("each run requires positive observed duration")
            tolerance = 1e-10 * max(1.0, abs(a), abs(b))
            for m, ts in mods.items():
                response = responses.get(m, Response(Identity(), estimate=False, pooling="shared"))
                lo, hi = response.support_envelope()
                valid = (ts.times - hi >= a - tolerance) & (ts.times - lo <= b + tolerance)
                valid &= (ts.times >= a - tolerance) & (ts.times <= b + tolerance)
                observed = int(ts.mask.sum())
                count = int(ts.mask[valid].sum())
                records[r]["observed_scalars"] += observed
                records[r]["eligible_scalars"] += count
                records[r]["series"].append(
                    {
                        "subject": s,
                        "modality": m,
                        "rows": len(ts.times),
                        "features": ts.values.shape[1],
                        "observed_scalars": observed,
                        "eligible_scalars": count,
                    }
                )
    for record in records.values():
        n = record["eligible_scalars"]
        record.update(one_covariance_bytes=8 * n * n, within_limit=0 < n <= max_observations)
    return {
        "max_observations": max_observations,
        "runs": records,
        "within_limit": bool(records) and all(r["within_limit"] for r in records.values()),
        "storage_scope": "one_float64_covariance_not_total_process_memory",
    }


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save_results(path, results, *, metadata=None):
    """Write a new, non-pickle directory of marginal mixtures and provenance.

    Keys are (subject, run, modality) string tuples. This exports query results,
    not a fitted estimator; it does not support new queries without refitting.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    arrays, records = {}, []
    for i, (key, result) in enumerate(results.items()):
        if not isinstance(key, tuple) or len(key) != 3 or any(not isinstance(v, str) for v in key):
            raise ValueError("result keys must be three-string tuples")
        if not isinstance(result, GaussianMixtureSeries):
            raise ValueError("results must contain GaussianMixtureSeries")
        for field in ("component_means", "component_variances", "times", "valid"):
            arrays[f"r{i}_{field}"] = getattr(result, field)
        records.append({"key": list(key), "prefix": f"r{i}", "metadata": _json(result.metadata)})
    if not records:
        raise ValueError("at least one result is required")
    manifest = {
        "schema_version": 1,
        "results": records,
        "metadata": _json(metadata or {}),
    }
    json.dumps(manifest, allow_nan=False)  # Validate before creating the directory.
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "arrays.npz", **arrays)
    manifest["arrays_sha256"] = hashlib.sha256((path / "arrays.npz").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def load_results(path):
    """Verify an archive and recover the original marginal mixture objects."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported result schema version")
    payload = path / "arrays.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("result array hash mismatch")
    results = {}
    with np.load(payload, allow_pickle=False) as arrays:
        for record in manifest["results"]:
            key = tuple(record["key"])
            if len(key) != 3 or any(not isinstance(v, str) for v in key) or key in results:
                raise ValueError("invalid or duplicate result key")
            prefix = record["prefix"]
            results[key] = GaussianMixtureSeries(
                **{
                    field: arrays[f"{prefix}_{field}"]
                    for field in (
                        "component_means",
                        "component_variances",
                        "times",
                        "valid",
                    )
                },
                metadata=record["metadata"],
            )
    return results, manifest["metadata"]


def save_model(path, model, *, standardizer=None):
    """Save a training or updated posterior, MAP group, or ParticipantCalibration.

    Writes a new, non-pickle JSON/NPZ directory, including training observations
    needed to reconstruct the GP. Preserve empirical archives locally. Finite
    nonconverged fits retain their diagnostics; saving does not certify quality.
    Version 1 stores MAP group fits; version 2 keeps group and participant fits
    separate; version 3 preserves raw posterior training draws and diagnostics.
    Version 4 preserves explicit joint posterior updates and their evidence ledger.
    For a ParticipantCalibration, supply its new-only standardizer. Conditioned
    MAP copies and sampler continuation are unsupported. Optional posterior_provenance_
    is preserved as caller-supplied training provenance, separately from writer
    metadata; unknown training provenance remains unknown.
    """
    from .persistence import save_model as write

    return write(path, model, standardizer=standardizer)


def load_model(path):
    """Return ``(model, standardizer)`` for a training or participant archive.

    Rebuilds the training problem without fitting, optimizing or sampling.
    The standardizer is None if not exported. Apply it explicitly to new raw
    inputs as before fitting; estimator methods continue to consume supplied units.
    Original convergence flags and raw coordinates are retained. Posterior
    reporting reconstructs its draw-specific QR coordinates without refitting.
    """
    from .persistence import load_model as read

    return read(path)
