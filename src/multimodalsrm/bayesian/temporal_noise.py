"""Fixed OU measurement covariance and exact observed-time Markov precision."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..data import readonly_array
from ._backend import runtime


def validate(value, modalities):
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not set(value) <= set(modalities):
        raise ValueError("noise_timescales must map known modalities to positive seconds")
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, float, np.integer, np.floating))
        or not np.isfinite(v)
        or v <= 0
        for v in value.values()
    ):
        raise ValueError("noise_timescales must contain finite positive seconds")
    return {m: float(v) for m, v in value.items()}


@dataclass(frozen=True)
class NoiseSystem:
    times: np.ndarray
    keys: np.ndarray
    lengths: np.ndarray
    diagonal: np.ndarray
    current: np.ndarray
    previous: np.ndarray
    off_diagonal: np.ndarray
    logdet: float

    @classmethod
    def prepare(cls, times, keys, all_keys, timescales):
        times = np.asarray(times, float)
        indices = np.array([all_keys.index(k) for k in keys])
        lengths = np.array([timescales.get(k[1], 0.0) for k in keys])
        diagonal = np.ones(len(times))
        current, previous, off, determinant = [], [], [], 0.0
        for key in np.unique(indices):
            rows = np.flatnonzero(indices == key)
            rows = rows[np.argsort(times[rows])]
            delta = np.diff(times[rows])
            if np.any(delta <= 0) or not np.isfinite(delta).all():
                raise ValueError("noise covariance requires distinct finite times per feature/run")
            tau = lengths[rows[0]]
            if not tau:
                continue
            rho = np.exp(-delta / tau)
            innovation = -np.expm1(-2 * delta / tau)
            i, j = rows[1:], rows[:-1]
            diagonal[i] += rho**2 / innovation
            diagonal[j] += rho**2 / innovation
            current.extend(i)
            previous.extend(j)
            off.extend(-rho / innovation)
            determinant += float(np.log(innovation).sum())
        if not np.isfinite([*diagonal, *off, determinant]).all():
            raise ValueError("nonfinite temporal-noise precision; no jitter is applied")
        return cls(
            *(
                readonly_array(a, dtype=a.dtype)
                for a in (
                    times,
                    indices,
                    lengths,
                    diagonal,
                    np.array(current, dtype=int),
                    np.array(previous, dtype=int),
                    np.array(off, dtype=float),
                )
            ),
            determinant,
        )

    def precision(self, rhs, variance):
        """Apply N^-1 without allocating any observation covariance matrix."""
        _, jnp, _, _ = runtime()
        rhs = jnp.asarray(rhs)
        expand = (slice(None),) + (None,) * (rhs.ndim - 1)
        answer = (self.diagonal / variance)[expand] * rhs
        coefficient = (self.off_diagonal / variance[self.current])[expand]
        answer = answer.at[self.current].add(coefficient * rhs[self.previous])
        return answer.at[self.previous].add(coefficient * rhs[self.current])

    def correlation(self):
        same = self.keys[:, None] == self.keys
        delta = abs(self.times[:, None] - self.times)
        safe = np.where(self.lengths > 0, self.lengths, 1.0)
        return same * np.where(
            self.lengths[:, None] > 0,
            np.exp(-delta / safe[:, None]),
            np.eye(len(delta)),
        )

    def cross(self, query, key, variance):
        """OU residual covariance with the same process, never white replicas."""
        _, jnp, _, _ = runtime()
        safe = np.where(self.lengths > 0, self.lengths, 1.0)
        return (
            jnp.exp(-jnp.abs(query[:, None] - self.times) / safe)
            * ((self.keys == key) & (self.lengths > 0))[None, :]
            * variance
        )


def design_transpose(rhs, weights, nodes, count):
    jax, jnp, _, _ = runtime()
    rhs = jnp.asarray(rhs)
    extra = rhs.shape[1:]
    product = weights[(slice(None), slice(None)) + (None,) * len(extra)] * rhs[:, None]
    return jax.ops.segment_sum(product, nodes, num_segments=count).reshape(
        count * weights.shape[1], *extra
    )


def design_product(rhs, weights, nodes, count):
    _, jnp, _, _ = runtime()
    extra = rhs.shape[1:]
    latent = rhs.reshape(count, weights.shape[1], *extra)
    return jnp.sum(
        weights[(slice(None), slice(None)) + (None,) * len(extra)] * latent[nodes],
        axis=1,
    )


def operands(problem, x, run):
    from .grouped import operands as shared_operands

    _, jnp, _, _ = runtime()
    temporal, weights, residual, variance, nodes = shared_operands(problem, x, run)
    weights = weights.reshape(len(weights), problem.features)
    return (
        jnp.kron(temporal, jnp.eye(problem.features)),
        weights,
        residual,
        variance,
        nodes,
    )


def factor(problem, x, run):
    """Woodbury sufficient statistics with non-diagonal A.T N^-1 A."""
    _, jnp, jsp, _ = runtime()
    K, w, residual, variance, nodes = operands(problem, x, run)
    system = problem.noise_systems[run]
    count = len(K) // problem.features
    locations = nodes[:, None] * problem.features + np.arange(problem.features)
    D = (
        jnp.zeros_like(K)
        .at[locations[:, :, None], locations[:, None, :]]
        .add(w[:, :, None] * w[:, None, :] * (system.diagonal / variance)[:, None, None])
    )
    i, j = system.current, system.previous
    blocks = w[i, :, None] * w[j, None, :] * (system.off_diagonal / variance[i])[:, None, None]
    D = D.at[locations[i, :, None], locations[j, None, :]].add(blocks)
    D = D.at[locations[j, :, None], locations[i, None, :]].add(jnp.swapaxes(blocks, 1, 2))
    score = design_transpose(system.precision(residual, variance), w, nodes, count)
    lu, pivots = jsp.linalg.lu_factor(jnp.eye(len(K)) + K @ D)
    diagonal = jnp.diag(lu)
    sign = jnp.prod(jnp.sign(diagonal)) * (-1.0) ** jnp.sum(pivots != jnp.arange(len(K)))
    logdet = jnp.where(sign > 0, jnp.log(jnp.abs(diagonal)).sum(), jnp.inf)
    q = jsp.linalg.lu_solve((lu, pivots), score, trans=1)
    mean = K @ q
    error = residual - design_product(mean, w, nodes, count)
    quadratic = error @ system.precision(error, variance) + q @ mean
    value = 0.5 * (
        quadratic
        + jnp.log(variance).sum()
        + system.logdet
        + logdet
        + len(residual) * np.log(2 * np.pi)
    )
    return value, q, D, (lu, pivots)


def solve(problem, x, run, rhs, lu):
    """Apply the full observation precision to vectors or narrow RHS batches."""
    _, _, jsp, _ = runtime()
    K, weights, _, variance, nodes = operands(problem, x, run)
    system = problem.noise_systems[run]
    count = len(K) // problem.features
    first = system.precision(rhs, variance)
    score = design_transpose(first, weights, nodes, count)
    q = jsp.linalg.lu_solve(lu, score, trans=1)
    return first - system.precision(design_product(K @ q, weights, nodes, count), variance)
