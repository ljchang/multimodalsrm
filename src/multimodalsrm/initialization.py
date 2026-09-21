"""Deterministic informed starts from preprocessed native training blocks only."""

import numpy as np

from .grouping import latent_groups


def spectral_latents(blocks, grids, subjects, features, pooling, affinity=None):
    """Backproject native blocks and estimate a common basis across run rows.

    Individual soft modes receive a global seed to align initial map orientations.
    Unsupported cells are zero. Absolute operator mass normalizes signed kernels
    without cancellation or division by tiny squared operator values. This is a
    starting heuristic, not inverse filtering or a change to scored observations.
    On insufficient numerical rank, return None for the caller's existing random
    path; no padded zero dimensions are represented as learned components.
    """
    groups = latent_groups(subjects, pooling, affinity)
    if pooling in ("population", "neighborhood"):
        groups = [list(subjects)]
    diagnostics = dict(method="spectral", fallback_reason=None, groups=[])
    slices = {}
    offset = 0
    for run, grid in grids.items():
        slices[run] = slice(offset, offset + len(grid))
        offset += len(grid)
    result = {s: {} for s in subjects}
    for group in groups:
        local = [b for b in blocks if b.subject in group]
        columns = list(
            dict.fromkeys(
                (b.subject, b.modality, j)
                for s in group
                for b in local
                if b.subject == s
                for j in range(b.values.shape[1])
            )
        )
        lookup = {key: j for j, key in enumerate(columns)}
        numerator = np.zeros((offset, len(columns)))
        support = np.zeros_like(numerator)
        for b in local:
            weight = np.where(b.mask & b.valid[:, None], b.coefficients, 0.0)
            values = np.where(weight > 0, b.values, 0.0)
            ix = [lookup[b.subject, b.modality, j] for j in range(b.values.shape[1])]
            numerator[slices[b.run], ix] += b.H.T @ (weight * values)
            support[slices[b.run], ix] += abs(b.H).T @ weight
        threshold = max(float(np.max(support, initial=0.0)) * 1e-12, np.finfo(float).tiny)
        supported = support > threshold
        matrix = np.divide(numerator, support, out=np.zeros_like(numerator), where=supported)
        info = dict(
            subjects=list(group),
            supported_cells=int(supported.sum()),
            total_cells=int(supported.size),
            numerical_rank=0,
        )
        diagnostics["groups"].append(info)
        if np.isfinite(matrix).all() and matrix.size:
            try:
                u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            except np.linalg.LinAlgError:
                singular = np.array([])
            cutoff = singular[0] * max(matrix.shape) * np.finfo(float).eps if singular.size else 0.0
            info["numerical_rank"] = int(np.count_nonzero(singular > cutoff))
        if info["numerical_rank"] < features:
            diagnostics.update(
                method="random",
                fallback_reason=f"insufficient numerical rank for group {group}: {info['numerical_rank']} < {features}",
            )
            return None, diagnostics
        basis = u[:, :features] * np.sqrt(offset)
        for j in range(features):
            if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
                basis[:, j] *= -1
        for s in group:
            result[s] = {run: basis[sl].copy() for run, sl in slices.items()}
    return result, diagnostics
