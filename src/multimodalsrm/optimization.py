"""Original-training objective diagnostics and a reconstruction-preserving scale step.

These functions do not support calibration or frozen population references.
All kernel parameters and native observation operators are held fixed here.
"""

import numpy as np

from .grouping import latent_groups
from .objective import (
    _loading_penalty_divisor,
    _penalty,
    _validate_loading_penalty_scaling,
    complete_objective,
)


def objective_components(
    blocks,
    grids,
    latents,
    loadings,
    pooling,
    strength,
    ridge,
    temporal,
    loading_ridge,
    affinity=None,
    kernel_penalty=0.0,
    loading_penalty_scaling="pair",
):
    """Decompose the exact finite-grid training objective (no reference anchor)."""
    loading_penalty_scaling = _validate_loading_penalty_scaling(loading_penalty_scaling)
    reconstruction = {}
    for b in blocks:
        residual = b.values - (b.H @ latents[b.subject][b.run]) @ loadings[b.subject][b.modality].T
        reconstruction[b.modality] = reconstruction.get(b.modality, 0.0) + float(
            np.sum(b.coefficients * residual**2)
        )
    pairs = [w for mods in loadings.values() for w in mods.values()]
    loading = float(
        loading_ridge
        * sum(np.sum(w * w) / _loading_penalty_divisor(w, loading_penalty_scaling) for w in pairs)
        / len(pairs)
    )
    subjects = list(latents)
    penalties = _penalty(grids, subjects, pooling, strength, ridge, temporal, affinity)
    latent = 0.0
    for r, P in penalties.items():
        z = np.concatenate([latents[g[0]][r] for g in latent_groups(subjects, pooling, affinity)])
        latent += float(np.sum(z * (P @ z)))
    rec = float(sum(reconstruction.values()))
    return dict(
        reconstruction_by_modality=reconstruction,
        reconstruction=rec,
        loading_penalty=loading,
        latent_penalty=latent,
        kernel_penalty=float(kernel_penalty),
        total=rec + loading + latent + float(kernel_penalty),
    )


def _imbalance(loading, latent):
    if not np.isfinite([loading, latent]).all() or loading < 0 or latent < 0:
        return float("nan")
    scale = max(loading, latent)
    if scale == 0:
        return 0.0
    return float(abs(loading / scale - latent / scale) / (loading / scale + latent / scale))


def balance_global_scale(
    blocks,
    grids,
    latents,
    loadings,
    pooling,
    strength,
    ridge,
    temporal,
    loading_ridge,
    affinity=None,
    kernel_penalty=0.0,
    loading_penalty_scaling="pair",
):
    """Return (latents, loadings, diagnostic), never modifying input arrays.

    Minimize A/c**2+B*c**2 with c=(A/B)**.25. Use log arithmetic
    to avoid overflowing A/B, with no clipping of successful candidate scales.
    Reject any nonfinite state or complete-objective increase. No-op returns
    the original mappings; accepted outputs own new arrays. Shared input must
    already represent one equal trajectory per run, as in the training solver.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        comp = objective_components(
            blocks,
            grids,
            latents,
            loadings,
            pooling,
            strength,
            ridge,
            temporal,
            loading_ridge,
            affinity,
            kernel_penalty,
            loading_penalty_scaling,
        )
        before = complete_objective(
            blocks,
            grids,
            latents,
            loadings,
            pooling,
            strength,
            ridge,
            temporal,
            loading_ridge,
            affinity,
            kernel_penalty,
            loading_penalty_scaling=loading_penalty_scaling,
        )
    A, B = comp["loading_penalty"], comp["latent_penalty"]
    d = dict(
        accepted=False,
        reason="degenerate_penalty",
        scale=1.0,
        before=before,
        after=before,
        relative_scale_imbalance=_imbalance(A, B),
    )
    if not np.isfinite([before, A, B]).all():
        d["reason"] = "nonfinite_objective"
        return latents, loadings, d
    if A <= 0 or B <= 0:
        return latents, loadings, d
    for group in latent_groups(list(latents), pooling, affinity):
        first = group[0]
        if any(
            not np.array_equal(v, latents[first][r]) for s in group for r, v in latents[s].items()
        ):
            d["reason"] = "unequal_shared_latents"
            return latents, loadings, d
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        c = float(np.exp((np.log(A) - np.log(B)) / 4))
        if not np.isfinite(c) or c <= 0:
            d["reason"] = "nonfinite_scale"
            return latents, loadings, d
        z = {s: {r: v * c for r, v in runs.items()} for s, runs in latents.items()}
        w = {s: {m: v / c for m, v in mods.items()} for s, mods in loadings.items()}
        if not all(
            np.isfinite(v).all()
            for mapping in [z, w]
            for entries in mapping.values()
            for v in entries.values()
        ):
            d["reason"] = "nonfinite_candidate"
            return latents, loadings, d
        after = complete_objective(
            blocks,
            grids,
            z,
            w,
            pooling,
            strength,
            ridge,
            temporal,
            loading_ridge,
            affinity,
            kernel_penalty,
            loading_penalty_scaling=loading_penalty_scaling,
        )
        if not np.isfinite(after) or after > before:
            d["reason"] = "objective_increase" if np.isfinite(after) else "nonfinite_candidate"
            return latents, loadings, d
        balanced = objective_components(
            blocks,
            grids,
            z,
            w,
            pooling,
            strength,
            ridge,
            temporal,
            loading_ridge,
            affinity,
            kernel_penalty,
            loading_penalty_scaling,
        )
    d.update(
        accepted=True,
        reason="balanced",
        scale=c,
        after=after,
        relative_scale_imbalance=_imbalance(
            balanced["loading_penalty"], balanced["latent_penalty"]
        ),
    )
    return z, w, d


def _gradient_terms(
    blocks,
    grids,
    latents,
    loadings,
    pooling,
    strength,
    ridge,
    temporal,
    loading_ridge,
    affinity,
    loading_penalty_scaling="pair",
):
    loading_penalty_scaling = _validate_loading_penalty_scaling(loading_penalty_scaling)
    subjects = list(latents)
    pairs = sum(len(mods) for mods in loadings.values())
    gw = {
        s: {
            m: 2
            * loading_ridge
            * w
            / (pairs * _loading_penalty_divisor(w, loading_penalty_scaling))
            for m, w in mods.items()
        }
        for s, mods in loadings.items()
    }
    bw = {s: {m: np.zeros_like(w) for m, w in mods.items()} for s, mods in loadings.items()}
    groups = latent_groups(subjects, pooling, affinity)
    keys = [g[0] for g in groups]
    representatives = {s: g[0] for g in groups for s in g}
    gz = {s: {r: np.zeros_like(z) for r, z in latents[s].items()} for s in keys}
    bz = {s: {r: np.zeros_like(z) for r, z in latents[s].items()} for s in keys}
    for b in blocks:
        w = loadings[b.subject][b.modality]
        hz = b.H @ latents[b.subject][b.run]
        weighted_residual = b.coefficients * (hz @ w.T - b.values)
        gw[b.subject][b.modality] += 2 * weighted_residual.T @ hz
        bw[b.subject][b.modality] += (b.coefficients * b.values).T @ hz
        s = representatives[b.subject]
        gz[s][b.run] += 2 * (b.H.T @ (weighted_residual @ w))
        bz[s][b.run] += b.H.T @ ((b.coefficients * b.values) @ w)
    penalties = _penalty(grids, subjects, pooling, strength, ridge, temporal, affinity)
    for r, P in penalties.items():
        z = np.concatenate([latents[s][r] for s in keys])
        # P is symmetric for valid training graphs; this form is also the exact
        # derivative for an arbitrary supplied matrix.
        pg = (P + P.T) @ z
        for i, s in enumerate(keys):
            gz[s][r] += pg[i * len(grids[r]) : (i + 1) * len(grids[r])]
    return gw, gz, bw, bz


def training_gradients(
    blocks,
    grids,
    latents,
    loadings,
    pooling,
    strength,
    ridge,
    temporal,
    loading_ridge,
    affinity=None,
    kernel_penalty=0.0,
    loading_penalty_scaling="pair",
):
    """Return analytic (loading, latent) derivatives including factor two.

    Shared latent derivatives aggregate every subject's reconstruction term;
    only the first subject is returned, representing each free variable once.
    Kernel penalty is fixed and has no loading/latent derivative.
    """
    return _gradient_terms(
        blocks,
        grids,
        latents,
        loadings,
        pooling,
        strength,
        ridge,
        temporal,
        loading_ridge,
        affinity,
        loading_penalty_scaling,
    )[:2]


def stationarity_diagnostics(
    blocks,
    grids,
    latents,
    loadings,
    pooling,
    strength,
    ridge,
    temporal,
    loading_ridge,
    affinity=None,
    kernel_penalty=0.0,
    loading_penalty_scaling="pair",
):
    """Final fixed-kernel quadratic stationarity, independent of solver history.

    Each relative residual is ||gradient||_2 / max(2*||rhs||_2, 1e-15),
    where rhs belongs to that coordinate's normal equations. Norms aggregate
    all free coordinates, counting a shared trajectory once. This is not a
    convergence declaration or a diagnostic of kernel-parameter stationarity.
    """
    gw, gz, bw, bz = _gradient_terms(
        blocks,
        grids,
        latents,
        loadings,
        pooling,
        strength,
        ridge,
        temporal,
        loading_ridge,
        affinity,
        loading_penalty_scaling,
    )
    result = dict(scope="fixed_kernel_training_quadratic")
    for name, g, b in [("loading", gw, bw), ("latent", gz, bz)]:
        gv = np.concatenate([v.ravel() for entries in g.values() for v in entries.values()])
        bv = np.concatenate([v.ravel() for entries in b.values() for v in entries.values()])
        result[name + "_relative_residual"] = float(
            np.linalg.norm(gv) / max(2 * np.linalg.norm(bv), 1e-15)
        )
        result[name + "_max_abs_gradient"] = float(np.max(np.abs(gv)))
        result[name + "_gradient_norm"] = float(np.linalg.norm(gv))
        result[name + "_variable_count"] = int(gv.size)
    comp = objective_components(
        blocks,
        grids,
        latents,
        loadings,
        pooling,
        strength,
        ridge,
        temporal,
        loading_ridge,
        affinity,
        kernel_penalty,
        loading_penalty_scaling,
    )
    result["relative_scale_imbalance"] = _imbalance(comp["loading_penalty"], comp["latent_penalty"])
    return result
