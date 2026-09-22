"""Vectorized SCR cell integration with complex-step physical derivatives.

Eight-point Gaussian quadrature is subdivided to resolve the faster stage.
All parameter dependence, including moving support endpoints and the exact
finite L2 normalizer, is differentiated. No optional JAX dependency is needed.
"""

import numpy as np
from scipy.sparse import csr_matrix

from . import _bateman


def bateman_operator(grid, times, kernel, valid, derivatives=False):
    rows, cols, values = [], [], []
    gradients = {p: [] for p in kernel.parameters}
    nodes, weights = np.polynomial.legendre.leggauss(8)
    dt = grid[1] - grid[0]
    subdivisions = max(1, int(np.ceil(dt / (2 * min(kernel.rise, kernel.decay)))))
    nodes = ((np.arange(subdivisions)[:, None] + (nodes + 1) / 2) / subdivisions).ravel()
    weights = np.tile(weights / (2 * subdivisions), subdivisions)
    indices = np.flatnonzero(valid)
    for start in range(0, len(indices), 32):
        selected = indices[start : start + 32]
        lower = times[selected, None] - grid[None, 1:] - kernel.lag
        upper = times[selected, None] - grid[None, :-1] - kernel.lag
        r, c = np.nonzero((upper > 0) & (lower < _bateman.SCR_DURATION))
        t = times[selected[r]]

        def integrate(p):
            lo = t - grid[c + 1] - p["lag"]
            hi = t - grid[c] - p["lag"]
            lo = np.where(lo.real > 0, lo, 0)
            hi = np.where(hi.real < _bateman.SCR_DURATION, hi, _bateman.SCR_DURATION)
            u = lo[:, None] + (hi - lo)[:, None] * nodes
            mass = (hi - lo)[:, None] * weights * _bateman.raw(u, p["rise"], p["decay"])
            mass /= _bateman.energy(p["rise"], p["decay"])
            fraction = (grid[c + 1, None] - t[:, None] + p["lag"] + u) / dt
            left = np.sum(mass * fraction, axis=1)
            right = np.sum(mass * (1 - fraction), axis=1)
            return np.column_stack([left, right]).ravel()

        rows.append(np.repeat(selected[r], 2))
        cols.append(np.column_stack([c, c + 1]).ravel())
        values.append(integrate(kernel.parameters))
        if derivatives:
            for p in gradients:
                parameters = dict(kernel.parameters)
                parameters[p] += 1e-25j
                gradients[p].append(integrate(parameters).imag / 1e-25)
    shape = (len(times), len(grid))
    if not rows:
        empty = csr_matrix(shape)
        return (
            (empty, {p: empty.copy() for p in gradients}, valid) if derivatives else (empty, valid)
        )
    locations = (np.concatenate(rows), np.concatenate(cols))
    op = csr_matrix((np.concatenate(values), locations), shape=shape)
    op.eliminate_zeros()
    if derivatives:
        return (
            op,
            {
                p: csr_matrix((np.concatenate(v), locations), shape=shape)
                for p, v in gradients.items()
            },
            valid,
        )
    return op, valid
