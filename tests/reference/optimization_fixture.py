"""Synthetic fixture extracted from frozen source; no research runner or evidence artifacts."""

import numpy as np
from scipy.integrate import trapezoid

from multimodalsrm import Identity, TimeSeries
from multimodalsrm.validation import common_valid_mask, score_prediction

RUNS = {"train-A": 1, "train-B": 2, "test-C": 3, "calibration-D": 4, "evaluation-E": 5}

DURATION = 32.0


def subset(data, runs):
    return {
        s: {r: mods for r, mods in sr.items() if r in runs}
        for s, sr in data.items()
        if any(r in runs for r in sr)
    }


def latent(t, seed, run, rank=1, cluster=0):
    """Independent run/cluster phases; frequencies integer-period over 32 seconds."""
    t = np.asarray(t)
    rng = np.random.default_rng(np.random.SeedSequence([seed, RUNS[run], 711, cluster]))
    phases = rng.uniform(0, 2 * np.pi, (rank, 3))
    frequencies = np.asarray([[2 + 2 * k, 5 + 2 * k, 9 + 2 * k] for k in range(rank)]) / DURATION
    z = [
        sum(
            a * np.sin(2 * np.pi * f * t + p)
            for a, f, p in zip([1, 0.35, 0.15], frequencies[k], phases[k])
        )
        for k in range(rank)
    ]
    return np.stack(z, axis=-1)


def convolved(t, kernel, seed, run, rank=1, cluster=0):
    if isinstance(kernel, Identity):
        return latent(t, seed, run, rank, cluster)
    lag = np.linspace(*kernel.support, 3001)
    return trapezoid(
        latent(np.asarray(t)[:, None] - lag[None, :], seed, run, rank, cluster)
        * kernel.evaluate(lag)[None, :, None],
        lag,
        axis=1,
    )


def generate(
    seed,
    maps,
    kernels=None,
    *,
    rank=1,
    noise=0.04,
    dt=None,
    clusters=None,
    runs=tuple(RUNS),
    mask=False,
):
    kernels = kernels or {s: {m: Identity() for m in mods} for s, mods in maps.items()}
    dt = dt or {}
    data = {}
    for si, (s, mods) in enumerate(maps.items()):
        data[s] = {}
        for run in runs:
            data[s][run] = {}
            for mi, (m, W) in enumerate(mods.items()):
                step = dt.get(m, 0.5)
                t = np.arange(0, DURATION + 1e-9, step)
                rng = np.random.default_rng(np.random.SeedSequence([seed, si, mi, RUNS[run], 123]))
                values = (
                    convolved(t, kernels[s][m], seed, run, rank, (clusters or {}).get(s, 0))
                    @ np.asarray(W).T
                )
                values += rng.normal(0, noise, size=values.shape)
                observed = np.ones(values.shape, bool)
                if mask:
                    observed[rng.uniform(size=values.shape) < 0.1] = False
                    observed[(t > 14) & (t < 18)] = False
                    values[~observed] = np.nan
                data[s][run][m] = TimeSeries(values, t, observed)
    return data, kernels


def predicted_scores(
    model,
    conditioning,
    target,
    *,
    subject="s0",
    modality="brain",
    run="test-C",
    sources=("within", "across", "both"),
    window=(3, 29),
):
    selected = (target.times >= window[0]) & (target.times <= window[1])
    ts = TimeSeries(target.values[selected], target.times[selected], target.mask[selected])
    predictions = {
        source: model.predict(
            conditioning, targets={subject: [modality]}, source=source, times=ts.times
        )[subject][run][modality]
        for source in sources
    }
    common = common_valid_mask(ts.mask, [p.valid for p in predictions.values()])
    rows = {
        source: score_prediction(ts.values, p.values, common) for source, p in predictions.items()
    }
    detail = {
        source: dict(
            used_sources=p.metadata["used_sources"],
            domain=p.metadata["domain"],
            solver=p.metadata["solver"],
        )
        for source, p in predictions.items()
    }
    return rows, detail
