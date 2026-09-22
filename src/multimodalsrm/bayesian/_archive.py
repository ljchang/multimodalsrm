"""Versioned non-executable storage for built-in Bayesian model configurations."""

import hashlib
import json
import os
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ..data import TimeSeries, readonly_array
from ..kernels import (
    BatemanSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    Response,
)
from .fitting import SamplerConfig, SearchConfig
from .priors import BayesianPriors, Prior
from .spectral import SpectralConfig
from .workflow import TrainingStandardizer

CLASSES = {
    c.__name__: c
    for c in (
        TimeSeries,
        Identity,
        Gaussian,
        Gamma,
        DoubleGamma,
        BatemanSCR,
        Response,
        Prior,
        BayesianPriors,
        SearchConfig,
        SamplerConfig,
        SpectralConfig,
        TrainingStandardizer,
    )
}
FORMAT = "personalized-srm-bayesian-map"
VERSION = 1


def same(a, b):
    """Compare supported state without ambiguous array truth values."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b, equal_nan=True)
        )
    if is_dataclass(a) or is_dataclass(b):
        return type(a) is type(b) and all(
            same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict)
            and isinstance(b, dict)
            and a.keys() == b.keys()
            and all(same(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and np.isnan(a):
        return isinstance(b, (float, np.floating)) and np.isnan(b)
    return a == b


def encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("model arrays must have real numeric or Boolean dtype")
        name = f"a{len(arrays)}"
        arrays[name] = value.copy()
        return {"type": "array", "name": name}
    if isinstance(value, np.generic):
        return encode(value.item(), arrays)
    if type(value) in CLASSES.values():
        return {
            "type": "record",
            "class": type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name), arrays) for f in fields(value)},
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[encode(k, arrays), encode(v, arrays)] for k, v in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [encode(v, arrays) for v in value],
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": str(value)}
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"unsupported model archive object: {type(value).__name__}")


def decode(value, arrays, used):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid encoded model state")
    kind = value.get("type")
    if kind == "array" and set(value) == {"type", "name"}:
        name = value["name"]
        if name not in arrays.files:
            raise ValueError("missing model array")
        array = arrays[name]
        if array.dtype.kind not in "biuf":
            raise ValueError("invalid model array dtype")
        used.add(name)
        return readonly_array(array, array.dtype)
    if kind == "float" and set(value) == {"type", "value"}:
        if value["value"] not in ("inf", "-inf", "nan"):
            raise ValueError("invalid encoded float")
        return float(value["value"])
    if kind in ("tuple", "list", "dict") and set(value) == {"type", "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("invalid encoded container")
        if kind == "dict":
            result = {}
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid dictionary entry")
                key, item = [decode(v, arrays, used) for v in pair]
                if key in result:
                    raise ValueError("duplicate model state key")
                result[key] = item
            return result
        result = [decode(v, arrays, used) for v in items]
        return tuple(result) if kind == "tuple" else result
    if kind == "record" and set(value) == {"type", "class", "fields"}:
        if value["class"] == "BachSCR":
            raise ValueError(
                "BachSCR support has been removed. Reproduce this archive with its original "
                "Bach-supporting package version (Bach is supported in release 0.1.0), "
                "or configure BatemanSCR and refit from the original observations. "
                "Bach parameters and fitted models cannot be converted automatically."
            )
        cls = CLASSES.get(value["class"])
        record_fields = value["fields"]
        # Fill only additive fitting defaults in older archives; all other
        # missing or unknown fields remain errors. Keep provenance unchanged.
        if cls is SearchConfig and isinstance(record_fields, dict):
            expected = {f.name for f in fields(cls)}
            defaults = {"polish_max_parameters": 256, "n_jobs": 1, "conditioning": "none"}
            if expected - defaults.keys() <= record_fields.keys() <= expected:
                record_fields = {**defaults, **record_fields}
        if cls is SamplerConfig and isinstance(record_fields, dict):
            expected = {f.name for f in fields(cls)}
            defaults = {
                "mass_matrix": "dense",
                "max_dense_parameters": 1024,
                "orientation_refresh": "none",
            }
            if expected - defaults.keys() <= record_fields.keys() <= expected:
                record_fields = {**defaults, **record_fields}
        if (
            cls is None
            or not isinstance(record_fields, dict)
            or set(record_fields) != {f.name for f in fields(cls)}
        ):
            raise ValueError("unsupported model record or fields")
        return cls(**{k: decode(v, arrays, used) for k, v in record_fields.items()})
    raise ValueError("unsupported encoded model state")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, state, *, version=VERSION):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    arrays = {}
    manifest = dict(format=FORMAT, schema_version=version, state=encode(state, arrays))
    # All payload construction happens before claiming the destination. The final
    # checksum is the completion marker; incomplete publication cannot be loaded.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".srm-model-", dir=path.parent) as folder:
        temporary = Path(folder)
        np.savez_compressed(temporary / "arrays.npz", **arrays)
        manifest["arrays_sha256"] = digest(temporary / "arrays.npz")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2) + "\n"
        )
        (temporary / "manifest.sha256").write_text(digest(temporary / "manifest.json") + "\n")
        path.mkdir(exist_ok=False)
        for name in ("arrays.npz", "manifest.json", "manifest.sha256"):
            os.replace(temporary / name, path / name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in model archive")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON number: {value}")


def read(path):
    path = Path(path)
    try:
        if digest(path / "manifest.json") != (path / "manifest.sha256").read_text().strip():
            raise ValueError("model manifest hash mismatch")
        manifest = json.loads(
            (path / "manifest.json").read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            "format",
            "schema_version",
            "arrays_sha256",
            "state",
        }:
            raise ValueError("invalid model archive manifest")
        if (
            manifest["format"] != FORMAT
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] not in (1, 2, 3, 4)
        ):
            raise ValueError("unsupported model archive format or version")
        if digest(path / "arrays.npz") != manifest["arrays_sha256"]:
            raise ValueError("model array hash mismatch")
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            if len(set(arrays.files)) != len(arrays.files):
                raise ValueError("duplicate model array names")
            used = set()
            state = decode(manifest["state"], arrays, used)
            if used != set(arrays.files):
                raise ValueError("unreferenced model arrays")
        posterior = isinstance(state, dict) and state.get("kind") == "posterior_training"
        if (manifest["schema_version"] == 3) != posterior:
            raise ValueError("posterior archive kind requires schema version 3")
        updated = isinstance(state, dict) and state.get("kind") == "posterior_updated"
        if (manifest["schema_version"] == 4) != updated:
            raise ValueError("updated posterior archive kind requires schema version 4")
        return state
    except (
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        BadZipFile,
    ) as exc:
        raise ValueError(f"invalid or incomplete model archive: {exc}") from exc
