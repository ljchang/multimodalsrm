"""Synthetic fixture extracted from frozen source; no research runner or evidence artifacts."""

import numpy as np

from multimodalsrm import Gaussian, Identity, TimeSeries

from .continuous_covariance import response_covariance

LENGTH_SCALE = 1.5

KERNEL = Gaussian(0.35, 0.5)


def synthetic(*, seed=20260911, duration=32, aux_hz=10):
    """Independent GP runs, common loadings, and explicit generator jitter."""
    if not np.isfinite(duration) or duration < 12 or duration != int(duration):
        raise ValueError("duration must be an integer number of seconds >= 12")
    if not np.isfinite(aux_hz) or aux_hz <= 0:
        raise ValueError("aux_hz must be positive and finite")
    brain_times = np.arange(float(duration))
    aux_times = np.arange(0.0, duration, 1.0 / aux_hz)
    clocks, kernels = (brain_times, aux_times), (Identity(), KERNEL)
    blocks = [
        [
            response_covariance(a, b, ka, kb, LENGTH_SCALE, tolerance=1e-10)
            for b, kb in zip(clocks, kernels)
        ]
        for a, ka in zip(clocks, kernels)
    ]
    covariance = np.block(blocks)
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(len(covariance)))
    weights = {}
    base = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    for i, subject in enumerate(("s1", "s2", "s3")):
        angle = i * 2 * np.pi / 3
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        weights[subject, "brain"] = base @ rotation
        if subject != "s3":
            weights[subject, "aux"] = np.array([[0.8, -0.4]]) @ rotation
    truth = dict(weights=weights, latent={}, times=brain_times, generator_diagonal_jitter=1e-9)
    output = []
    for run, stream in zip(("train", "test"), np.random.SeedSequence(seed).spawn(2)):
        rng = np.random.default_rng(stream)
        responses = factor @ rng.normal(size=(len(covariance), 2))
        truth["latent"][run] = responses[: len(brain_times)]
        data = {}
        for (subject, modality), W in weights.items():
            latent = (
                responses[: len(brain_times)]
                if modality == "brain"
                else responses[len(brain_times) :]
            )
            t = brain_times if modality == "brain" else aux_times
            values = latent @ W.T + rng.normal(scale=0.1, size=(len(t), len(W)))
            mask = np.ones_like(values, dtype=bool)
            if subject == "s2" and modality == "brain":
                mask[5, 1] = False
            if subject == "s1" and modality == "aux":
                mask[::19, 0] = False
            data.setdefault(subject, {}).setdefault(run, {})[modality] = TimeSeries(values, t, mask)
        output.append(data)
    return *output, truth
