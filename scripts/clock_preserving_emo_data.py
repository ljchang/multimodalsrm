"""Research adapter keeping the first 252 brain volumes on their original clock.

The historical ignored loader removes splice-marked observations and renumbers
the remaining rows. Independent physiology pause markers support the original
row clock. Following the user's clarification, trim the eight extra volumes
from the end and treat splice flags as boundary markers. Retain the historical
nonzero-censor mask; do not separately mask post-splice contamination flags.

Brain values are raw parcel means. The research split applies training-only
standardization. Other streams retain the historical loader's values/clocks.

Use as --loader scripts/clock_preserving_emo_data.py in the empirical runners.
The original participant data and loader remain in ignored local storage.
"""

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from multimodalsrm import TimeSeries

LEGACY_PATH = (
    Path(__file__).resolve().parents[1] / "local_data/empirical-eval-2026-09-22/emo_data.py"
)
PHYSIO_COLUMNS = ("EDA - EDA100C-MRI", "Pulse Rate", "Respiration Rate")
SUBJECTS = ("s001", "s002", "s004", "s009", "s010")


def brain(legacy, subject):
    directory = legacy.ROOT / subject / "brain"
    image_path = next(directory.glob("*timecourse.nii.gz"))
    info_path = next(directory.glob("*info.csv"))
    info = pd.read_csv(info_path)
    labels = legacy.atlas_labels()
    image_stat = image_path.stat()
    provenance = json.dumps(
        dict(
            image_size=image_stat.st_size,
            image_mtime_ns=image_stat.st_mtime_ns,
            info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
            atlas_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        ),
        sort_keys=True,
    )
    cache = legacy.CACHE / f"brain_{subject}_first252_raw_parcels_v1.npz"
    if cache.exists():
        with np.load(cache) as saved:
            assert str(saved["provenance"]) == provenance, "Source changed; rebuild research cache"
            return TimeSeries(saved["values"], saved["times"], saved["mask"])
    image = nib.load(image_path)
    assert image.shape[-1] == len(info) == 260
    assert image.shape[:-1] == labels.shape
    # Slice before parcellation: the original eight tail volumes never enter analysis.
    data = np.asarray(image.dataobj, dtype=np.float32)[..., :252]
    values = np.stack([data[labels == label].mean(axis=0) for label in range(1, 101)], axis=1)
    mask = (info.censored.to_numpy()[:252, None] == 0) & np.isfinite(values)
    times = legacy.TR * np.arange(252)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
        staged = Path(temporary) / cache.name
        np.savez(staged, values=values, times=times, mask=mask, provenance=provenance)
        staged.replace(cache)
    return TimeSeries(values, times, mask)


def load(subjects=SUBJECTS, parcels=None, window=None):
    spec = importlib.util.spec_from_file_location("historical_emo_loader", LEGACY_PATH)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    assert tuple(legacy.PHYSIO_COLUMNS) == PHYSIO_COLUMNS
    result = {}
    for subject in subjects:
        streams = {
            "brain": legacy.clip(brain(legacy, subject), parcels=parcels, window=window),
            "face": legacy.clip(legacy.face(subject), window=window),
            "rating": legacy.clip(legacy.ratings(subject), window=window),
        }
        physio = legacy.physio(subject)
        if physio is not None:
            streams["physio"] = legacy.clip(physio, window=window)
        result[subject] = {legacy.RUN: streams}
    return result
