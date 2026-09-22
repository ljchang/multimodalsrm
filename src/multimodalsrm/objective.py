"""One balanced objective and its quadratic coordinate solvers.

Ridges average over fitted pairs / participant-runs. Loading ridges optionally
normalize each pair by its fitted feature count. Temporal penalties are duration
averages on the grid with trapezoid nodes.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .grouping import latent_groups
from .operators import observation_operator, response_operator_derivatives


def _validate_loading_penalty_scaling(value):
    if value not in ("pair", "feature"):
        raise ValueError("loading_penalty_scaling must be 'pair' or 'feature'")
    return value


def _loading_penalty_divisor(loading, scaling):
    return loading.shape[0] if scaling == "feature" else 1


@dataclass
class ObservationBlock:
    subject: str
    run: str
    modality: str
    values: np.ndarray
    mask: np.ndarray
    times: np.ndarray
    H: object
    coefficients: np.ndarray
    valid: np.ndarray


class PreparedBlocks:
    """Fit-local fixed preprocessing with a bounded response-operator cache.

    Fixed support envelopes keep scored rows and all normalization independent
    of kernel candidates. Construct anew for each fit; never reuse across folds
    or datasets. Only H changes, while feature arrays are shared read-only by
    convention with the original blocks.
    """

    def __init__(self, blocks, grids, responses):
        self.blocks = blocks
        self.grids = grids
        self.supports = {m: r.support_envelope() for m, r in responses.items()}
        self.pairs = {(b.subject, b.modality) for b in blocks}
        self.cache = {}

    def operator(self, block, kernel, derivatives=False):
        key = (block.subject, block.run, block.modality, tuple(kernel.parameters.items()))
        cached = self.cache.get(key)
        if cached is None or (derivatives and cached[1] is None):
            args = (self.grids[block.run], block.times, kernel, self.supports[block.modality])
            if derivatives:
                h, gradients, _ = response_operator_derivatives(*args)
            else:
                h, _ = observation_operator(*args)
                gradients = None
            cached = (h, gradients)
            if key not in self.cache and len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = cached
        return cached

    def __call__(self, kernels):
        if not self.pairs.issubset({(s, m) for s, mods in kernels.items() for m in mods}):
            raise ValueError("kernel candidates must retain all fitted pairs")
        return [
            replace(b, H=self.operator(b, kernels[b.subject][b.modality])[0]) for b in self.blocks
        ]


def build_grids(data, dt):
    domains = {}
    for runs in data.values():
        for r, mods in runs.items():
            for ts in mods.values():
                t = ts.times[np.any(ts.mask, axis=1)]
                a, b = domains.get(r, (np.inf, -np.inf))
                domains[r] = (min(a, t[0]), max(b, t[-1]))
    grids = {}
    for r, (a, b) in domains.items():
        if b <= a:
            raise ValueError(f"run {r} needs positive observed duration")
        grids[r] = a + dt * np.arange(int(np.ceil((b - a) / dt)) + 1)
    return grids, domains


def fit_preprocessing(data, weights):
    stats = {}
    for s, runs in data.items():
        stats[s] = {}
        for m in dict.fromkeys(m for mods in runs.values() for m in mods if weights[m] > 0):
            series = [mods[m] for mods in runs.values() if m in mods]
            x = np.concatenate([ts.values for ts in series])
            mask = np.concatenate([ts.mask for ts in series])
            count = mask.sum(0)
            if np.any(count == 0):
                raise ValueError(f"no observed training values for feature in {s}/{m}")
            mean = np.where(mask, x, 0).sum(0) / count
            scale = np.sqrt(np.where(mask, (x - mean) ** 2, 0).sum(0) / count)
            constant = scale <= np.finfo(float).eps
            scale[constant] = 1
            stats[s][m] = dict(mean=mean, scale=scale, constant_features=constant)
    return stats


def timestamp_weights(times, gap_threshold=None):
    """Trapezoid cells per segment; singleton cell uses median native interval."""
    if len(times) == 1:
        return np.ones(1)
    delta = np.diff(times)
    median = np.median(delta)
    threshold = 2 * median if gap_threshold is None else gap_threshold
    segments = np.split(np.arange(len(times)), np.flatnonzero(delta > threshold) + 1)
    q = np.zeros(len(times))
    for seg in segments:
        if len(seg) == 1:
            q[seg] = median
        else:
            d = np.diff(times[seg])
            q[seg] = np.r_[d[0] / 2, (d[:-1] + d[1:]) / 2, d[-1] / 2]
    return q / q.sum()


def make_blocks(
    data,
    grids,
    domains,
    preprocessing,
    kernels,
    responses,
    weights,
    gap_threshold=None,
    operator_cache=None,
):
    """Build scored blocks. Optional cache must be scoped to fixed data/grids/responses.

    Cache holds at most 256 native observation operators; it never changes scores.
    """
    envelopes = {m: response.support_envelope() for m, response in responses.items()}
    blocks = []
    for s, runs in data.items():
        for r, mods in runs.items():
            local = []
            for m, ts in mods.items():
                if weights[m] <= 0 or m not in kernels.get(s, {}):
                    continue
                support = envelopes[m]
                key = (s, r, m, tuple(kernels[s][m].parameters.items()))
                cached = None if operator_cache is None else operator_cache.get(key)
                if cached is None:
                    H, valid = observation_operator(grids[r], ts.times, kernels[s][m], support)
                    if operator_cache is not None:
                        if len(operator_cache) >= 256:
                            operator_cache.pop(next(iter(operator_cache)))
                        operator_cache[key] = (H, valid.copy())
                else:
                    H, valid = cached
                    valid = valid.copy()
                valid &= (ts.times - support[1] >= domains[r][0] - 1e-10) & (
                    ts.times - support[0] <= domains[r][1] + 1e-10
                )
                valid &= ts.mask.any(1)
                if not valid.any():
                    continue
                stats = preprocessing[s][m]
                if len(stats["mean"]) != ts.values.shape[1]:
                    raise ValueError("feature count differs from fitted mapping")
                x = np.where(ts.mask, (ts.values - stats["mean"]) / stats["scale"], 0)
                q = np.zeros(len(ts.times))
                q[valid] = timestamp_weights(ts.times[valid], gap_threshold)
                c = weights[m] * q[:, None] * ts.mask / np.maximum(ts.mask.sum(1), 1)[:, None]
                local.append(ObservationBlock(s, r, m, x, ts.mask, ts.times, H, c, valid))
            if not local:
                raise ValueError(f"no positively weighted usable observations in {s}/{r}")
            mass = sum(weights[b.modality] for b in local)
            for b in local:
                b.coefficients /= mass
            blocks.extend(local)
    nblocks = len({(b.subject, b.run) for b in blocks})
    for b in blocks:
        b.coefficients /= nblocks
    return blocks


def _penalty(
    grids,
    subjects,
    pooling,
    strength,
    ridge,
    temporal,
    affinity,
    population_reference=None,
):
    """Run matrices over participant x time, component identity is implicit."""
    result = {}
    n = len(subjects)
    for r, grid in grids.items():
        t = len(grid)
        dt = grid[1] - grid[0]
        duration = grid[-1] - grid[0]
        q = np.ones(t) * dt / duration
        q[[0, -1]] *= 0.5
        Q = sparse.diags(q)
        D = sparse.diags([-np.ones(t - 1), np.ones(t - 1)], [0, 1], shape=(t - 1, t))
        base = ridge * Q + temporal * (D.T @ D) / (dt * duration)
        if pooling in ("shared", "components"):
            groups = latent_groups(subjects, pooling, affinity)
            result[r] = sparse.block_diag([len(g) / n * base / len(grids) for g in groups]).tocsr()
            continue
        P = sparse.kron(sparse.eye(n), base) / (n * len(grids))
        if pooling == "population":
            L = np.eye(n) if population_reference is not None else np.eye(n) - np.ones((n, n)) / n
            P += strength * sparse.kron(L, Q) / (n * len(grids))
        elif pooling == "neighborhood":
            A = np.array([[affinity.get(i, {}).get(j, 0.0) for j in subjects] for i in subjects])
            L = np.diag(A.sum(1)) - A
            P += strength * sparse.kron(L, Q) / (max(n, 1) * len(grids))
        result[r] = P.tocsr()
    return result


def _latent_block_normal(block, loading):
    """Contract observed features before assembling the latent normal matrix.

    At row t the sufficient statistics are W.T diag(c_t) W and
    W.T (c_t * x_t). This is exactly the observation-space weighted quadratic,
    including feature-specific missingness, without forming kron(H, W).
    """
    h, c = block.H, block.coefficients
    k = loading.shape[1]
    rhs = np.asarray(h.T @ ((c * block.values) @ loading)).ravel()
    if np.all(c == c[:, :1]):
        # Complete feature rows share a temporal weight: one temporal Gram.
        temporal = h.T @ h.multiply(c[:, :1])
        return sparse.kron(temporal, loading.T @ loading, format="csr"), rhs

    gram = np.einsum("tf,fi,fj->tij", c, loading, loading, optimize=True)
    rows, cols, values = [], [], []
    for i in range(k):
        for j in range(k):
            temporal = (h.T @ h.multiply(gram[:, i, j, None])).tocoo()
            rows.append(temporal.row * k + i)
            cols.append(temporal.col * k + j)
            values.append(temporal.data)
    size = h.shape[1] * k
    normal = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    ).tocsr()
    return normal, rhs


def solve_latents(
    blocks,
    grids,
    subjects,
    loadings,
    features,
    pooling,
    strength,
    ridge,
    temporal,
    affinity=None,
    fixed_latents=None,
    population_reference=None,
):
    penalties = _penalty(
        grids,
        subjects,
        pooling,
        strength,
        ridge,
        temporal,
        affinity,
        population_reference,
    )
    groups = latent_groups(subjects, pooling, affinity)
    group_index = {s: i for i, group in enumerate(groups) for s in group}
    output = {s: {} for s in subjects}
    diagnostics = []
    for r, grid in grids.items():
        n = len(groups)
        size = len(grid) * features
        normal = sparse.kron(penalties[r], sparse.eye(features)).tocsr()
        rhs = np.zeros(n * size)
        if pooling == "population" and population_reference is not None:
            q = np.ones(len(grid)) * (grid[1] - grid[0]) / (grid[-1] - grid[0])
            q[[0, -1]] *= 0.5
            anchor_rhs = strength * q[:, None] * population_reference[r] / (n * len(grids))
            rhs += np.tile(anchor_rhs.ravel(), n)
        for b in blocks:
            if b.run != r:
                continue
            local_normal, local_rhs = _latent_block_normal(b, loadings[b.subject][b.modality])
            offset = group_index[b.subject] * size
            coo = local_normal.tocoo()
            normal += sparse.csr_matrix(
                (coo.data, (coo.row + offset, coo.col + offset)),
                shape=(n * size, n * size),
            )
            rhs[offset : offset + size] += local_rhs
        fixed_latents = fixed_latents or {}
        fixed_indices, fixed_values = [], []
        for i, group in enumerate(groups):
            fixed = [
                np.asarray(fixed_latents[s][r]) for s in group if r in fixed_latents.get(s, {})
            ]
            if fixed:
                if any(not np.array_equal(fixed[0], value) for value in fixed[1:]):
                    raise ValueError("fixed latent donors in a shared group must agree")
                fixed_indices.extend(range(i * size, (i + 1) * size))
                fixed_values.extend(fixed[0].ravel())
        fixed_indices = np.asarray(fixed_indices, dtype=int)
        free = np.setdiff1d(np.arange(n * size), fixed_indices)
        z = np.zeros(n * size)
        z[fixed_indices] = fixed_values
        conditional_rhs = rhs[free] - normal[free][:, fixed_indices] @ z[fixed_indices]
        if len(free):
            z[free] = spsolve(normal[free][:, free].tocsc(), conditional_rhs)
        residual = np.linalg.norm((normal @ z - rhs)[free]) / max(
            np.linalg.norm(conditional_rhs), 1e-15
        )
        success = bool(np.isfinite(z).all() and residual < 1e-7)
        diagnostics.append(
            dict(
                run=r,
                success=success,
                relative_residual=float(residual),
                solver="spsolve",
            )
        )
        if not success:
            raise RuntimeError(f"latent sparse solve failed in {r}: residual {residual}")
        for i, s in enumerate(subjects):
            index = group_index[s]
            output[s][r] = z[index * size : (index + 1) * size].reshape(len(grid), features).copy()
    return output, diagnostics


def solve_loadings(blocks, latents, ridge, loading_penalty_scaling="pair"):
    loading_penalty_scaling = _validate_loading_penalty_scaling(loading_penalty_scaling)
    pairs = list(dict.fromkeys((b.subject, b.modality) for b in blocks))
    output = {}
    for s, m in pairs:
        local = [b for b in blocks if (b.subject, b.modality) == (s, m)]
        designs = [b.H @ latents[s][b.run] for b in local]
        k = designs[0].shape[1]
        W = np.zeros((local[0].values.shape[1], k))
        penalty = (
            np.eye(k) * ridge / (len(pairs) * _loading_penalty_divisor(W, loading_penalty_scaling))
        )
        if all(np.all(b.coefficients == b.coefficients[:, :1]) for b in local):
            # A single factorization serves all features with common row weights.
            normal, rhs = penalty.copy(), np.zeros((k, len(W)))
            for b, z in zip(local, designs):
                weighted = z.T * b.coefficients[:, 0]
                normal += weighted @ z
                rhs += weighted @ b.values
            W[:] = np.linalg.solve(normal, rhs).T
        else:
            # Arbitrary feature masks retain separate systems, solved in batches.
            products = [np.einsum("ti,tj->tij", z, z).reshape(len(z), k * k) for z in designs]
            for start in range(0, len(W), 256):
                stop = min(start + 256, len(W))
                normal = np.broadcast_to(penalty, (stop - start, k, k)).copy()
                rhs = np.zeros((stop - start, k))
                for b, z, zz in zip(local, designs, products):
                    c = b.coefficients[:, start:stop]
                    normal += (c.T @ zz).reshape(-1, k, k)
                    rhs += (c * b.values[:, start:stop]).T @ z
                W[start:stop] = np.linalg.solve(normal, rhs[..., None])[..., 0]
        output.setdefault(s, {})[m] = W
    return output


def complete_objective(
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
    population_reference=None,
    loading_penalty_scaling="pair",
):
    loading_penalty_scaling = _validate_loading_penalty_scaling(loading_penalty_scaling)
    value = float(kernel_penalty)
    for b in blocks:
        residual = b.values - (b.H @ latents[b.subject][b.run]) @ loadings[b.subject][b.modality].T
        value += np.sum(b.coefficients * residual**2)
    pairs = [w for mods in loadings.values() for w in mods.values()]
    value += (
        loading_ridge
        * sum(np.sum(w * w) / _loading_penalty_divisor(w, loading_penalty_scaling) for w in pairs)
        / len(pairs)
    )
    subjects = list(latents)
    penalties = _penalty(
        grids,
        subjects,
        pooling,
        strength,
        ridge,
        temporal,
        affinity,
        population_reference,
    )
    for r, P in penalties.items():
        z = np.concatenate([latents[g[0]][r] for g in latent_groups(subjects, pooling, affinity)])
        value += np.sum(z * (P @ z))
        if pooling == "population" and population_reference is not None:
            grid = grids[r]
            q = np.ones(len(grid)) * (grid[1] - grid[0]) / (grid[-1] - grid[0])
            q[[0, -1]] *= 0.5
            center = population_reference[r]
            factor = strength / (len(subjects) * len(grids))
            for subject in subjects:
                value += factor * np.sum(
                    q[:, None] * (center**2 - 2 * latents[subject][r] * center)
                )
    return float(value)
