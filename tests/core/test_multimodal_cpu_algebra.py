"""Numerical and workspace contracts for feature-exact CPU coordinate updates."""

import numpy as np
import pytest
from scipy import sparse

from multimodalsrm import objective
from multimodalsrm.grouping import latent_groups


def quadratic_case(common_weights=False, features=3):
    rng = np.random.default_rng(923)
    subjects = ["a", "b", "c"]
    grids = {"r0": np.arange(7.0), "r1": np.arange(6.0)}
    loadings = {
        s: {m: rng.normal(size=(f, features)) for m, f in [("brain", 11), ("face", 2)]}
        for s in subjects
    }
    blocks = []
    for si, s in enumerate(subjects):
        for r, grid in grids.items():
            for m, w in loadings[s].items():
                if s == "b" and m == "face":
                    continue
                t = len(grid) + si
                h = np.zeros((t, len(grid)))
                for row in h:
                    ix = rng.choice(len(grid), 2, replace=False)
                    row[ix] = [0.35, 0.65]
                c = rng.uniform(0.1, 1.0, size=(t, 1 if common_weights else len(w)))
                c[rng.random(c.shape) < 0.2] = 0
                c = np.broadcast_to(c, (t, len(w))).copy()
                c[0] = 0
                x = rng.normal(size=c.shape)
                blocks.append(
                    objective.ObservationBlock(
                        s,
                        r,
                        m,
                        x,
                        c > 0,
                        np.linspace(0, grid[-1], t),
                        sparse.csr_matrix(h),
                        c,
                        c.any(axis=1),
                    )
                )
    affinity = {"a": {"b": 1.0}, "b": {"a": 1.0}, "c": {}}
    return dict(
        blocks=blocks,
        grids=grids,
        subjects=subjects,
        loadings=loadings,
        features=features,
        strength=0.2,
        ridge=0.13,
        temporal=0.07,
        affinity=affinity,
    )


def dense_latent_oracle(a):
    """QR/SVD least squares on explicit observations, independent of contractions."""
    groups = latent_groups(a["subjects"], a["pooling"], a["affinity"])
    indices = {s: i for i, group in enumerate(groups) for s in group}
    penalties = objective._penalty(
        a["grids"],
        a["subjects"],
        a["pooling"],
        a["strength"],
        a["ridge"],
        a["temporal"],
        a["affinity"],
        a.get("population_reference"),
    )
    out = {s: {} for s in a["subjects"]}
    for r, grid in a["grids"].items():
        size = len(grid) * a["features"]
        p = np.kron(penalties[r].toarray(), np.eye(a["features"]))
        design = [np.linalg.cholesky(p).T]
        target = [np.zeros(len(p))]
        if a.get("population_reference") is not None:
            q = np.ones(len(grid)) * (grid[1] - grid[0]) / (grid[-1] - grid[0])
            q[[0, -1]] *= 0.5
            rhs = np.tile(
                (
                    a["strength"]
                    * q[:, None]
                    * a["population_reference"][r]
                    / (len(groups) * len(a["grids"]))
                ).ravel(),
                len(groups),
            )
            target[0] = np.linalg.solve(design[0].T, rhs)
        for b in a["blocks"]:
            if b.run != r:
                continue
            local = np.kron(b.H.toarray(), a["loadings"][b.subject][b.modality])
            full = np.zeros((len(local), len(p)))
            offset = indices[b.subject] * size
            full[:, offset : offset + size] = local
            root = np.sqrt(b.coefficients.ravel())
            design.append(root[:, None] * full)
            target.append(root * b.values.ravel())
        design, target = np.vstack(design), np.concatenate(target)
        z = np.zeros(len(p))
        fixed = []
        for s, runs in a.get("fixed_latents", {}).items():
            if r in runs:
                ix = np.arange(indices[s] * size, (indices[s] + 1) * size)
                z[ix] = runs[r].ravel()
                fixed.extend(ix)
        free = np.setdiff1d(np.arange(len(p)), fixed)
        z[free] = np.linalg.lstsq(design[:, free], target - design @ z, rcond=None)[0]
        for s in out:
            out[s][r] = z[indices[s] * size : (indices[s] + 1) * size].reshape(-1, a["features"])
    return out


@pytest.mark.parametrize("pooling", ["shared", "population", "neighborhood", "components"])
@pytest.mark.parametrize("common_weights", [False, True])
def test_latent_update_matches_observation_space_least_squares(pooling, common_weights):
    """Catches dropped masks, component transposes, group offsets and run mixing."""
    a = quadratic_case(common_weights) | {"pooling": pooling}
    a["fixed_latents"] = {"a": {"r0": np.arange(21.0).reshape(7, 3) / 20}}
    if pooling == "population":
        a["population_reference"] = {r: np.ones((len(g), 3)) * 0.3 for r, g in a["grids"].items()}
    expected = dense_latent_oracle(a)
    actual, diagnostics = objective.solve_latents(**a)
    assert all(d["success"] for d in diagnostics)
    for s, runs in expected.items():
        for r, z in runs.items():
            np.testing.assert_allclose(actual[s][r], z, atol=3e-12, rtol=3e-12)


def test_latent_update_avoids_observation_by_latent_workspace(monkeypatch):
    """Keep the sparse workspace bounded by latent dimension, not voxel count."""
    a = quadratic_case() | {"pooling": "shared"}
    expected = dense_latent_oracle(a)
    native_kron = sparse.kron
    limit = max(len(g) for g in a["grids"].values()) * a["features"]

    def bounded_kron(left, right, *args, **kwargs):
        assert left.shape[0] * right.shape[0] <= limit, (
            "expanded observation-space design exceeds latent workspace"
        )
        return native_kron(left, right, *args, **kwargs)

    monkeypatch.setattr(sparse, "kron", bounded_kron)
    actual, _ = objective.solve_latents(**a)
    for s, runs in expected.items():
        for r, z in runs.items():
            np.testing.assert_allclose(actual[s][r], z, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("common_weights", [False, True])
@pytest.mark.parametrize("scaling", ["pair", "feature"])
def test_loading_update_batches_features_and_preserves_weighted_fit(
    monkeypatch, common_weights, scaling
):
    """Catch per-feature solve dispatch and pooling or feature-mask errors."""
    rng = np.random.default_rng(431)
    f, k, ridge = 513, 3, 0.2
    latents, blocks, designs = {"a": {}}, [], []
    for r, t in [("r0", 21), ("r1", 17)]:
        z = rng.normal(size=(t, k))
        latents["a"][r] = z
        c = rng.uniform(0.2, 1.0, size=(t, 1 if common_weights else f))
        c[rng.random(c.shape) < 0.2] = 0
        c = np.broadcast_to(c, (t, f)).copy()
        x = rng.normal(size=(t, f))
        blocks.append(
            objective.ObservationBlock(
                "a",
                r,
                "brain",
                x,
                c > 0,
                np.arange(t),
                sparse.eye(t, format="csr"),
                c,
                c.any(axis=1),
            )
        )
        designs.append(z)
    expected = []
    penalty = np.eye(k) * np.sqrt(ridge / (f if scaling == "feature" else 1))
    for feature in range(f):
        matrix, target = [penalty], [np.zeros(k)]
        for b, z in zip(blocks, designs):
            root = np.sqrt(b.coefficients[:, feature])
            matrix.append(root[:, None] * z)
            target.append(root * b.values[:, feature])
        expected.append(np.linalg.lstsq(np.vstack(matrix), np.concatenate(target), rcond=None)[0])
    native_solve = np.linalg.solve
    calls = 0

    def bounded_dispatch(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= 4, "per-feature dispatch exceeds batched loading budget"
        return native_solve(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", bounded_dispatch)
    actual = objective.solve_loadings(blocks, latents, ridge, scaling)["a"]["brain"]
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
