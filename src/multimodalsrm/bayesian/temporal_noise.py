"""OU measurement covariance and exact observed-time Markov precision.

Timescales are either fixed numbers, prepared once with NumPy, or members of
the parameter vector, evaluated with JAX at every objective call.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..data import readonly_array
from ._backend import runtime
from .priors import Prior

# The constructor default of ``estimate_noise_timescales``.
ESTIMATE_BY_DEFAULT = False
NOISE_TIMESCALE = "noise_timescale"
# Default prior and bounds, in units of each modality's median sampling
# interval dt. Below dt / 20 the lag-one autocorrelation is under 3e-9 and the
# noise is white at this sampling rate; above the longest run the residual is
# an offset for that run, which the model cannot tell from the feature offset.
# The lognormal has median 2 dt (autocorrelation 0.61) and a 95% range of
# 0.17 dt to 23 dt (autocorrelation 0.003 to 0.96). MAP uses its density in
# seconds, whose mode exp(mu - sd^2) is 0.42 dt: white data, whose likelihood
# is flat below about 0.15 dt, settle between there and the mode, well inside
# the lower bound, instead of resting on it.
LOWER_INTERVALS = 0.05
PRIOR_MEDIAN_INTERVALS = 2.0
PRIOR_LOG_SD = 1.25


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


def validate_estimation(flag):
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError("estimate_noise_timescales must be Boolean")
    return bool(flag)


def sampling(systems, modalities):
    """Median sampling interval and longest run span per modality, in seconds."""
    intervals, spans = {m: [] for m in modalities}, {m: 0.0 for m in modalities}
    for system in systems.values():
        times, rows = np.asarray(system.times, float), {}
        for i, key in enumerate(system.keys):
            if key[1] in intervals:
                rows.setdefault(key, []).append(i)
        for key, index in rows.items():
            observed = np.sort(times[index])
            intervals[key[1]].append(np.diff(observed))
            spans[key[1]] = max(spans[key[1]], float(observed[-1] - observed[0]))
    output = {}
    for m in modalities:
        delta = np.concatenate(intervals[m]) if intervals[m] else np.empty(0)
        delta = delta[delta > 0]
        if not len(delta):
            raise ValueError(
                f"estimate_noise_timescales needs at least two distinct times per run "
                f"for modality {m!r}"
            )
        output[m] = float(np.median(delta)), spans[m]
    return output


def timescale_priors(supplied, sampled):
    """Per-modality priors truncated to [dt / 20, longest run]."""
    priors = {}
    for m, (interval, span) in sampled.items():
        prior = supplied.get(m) if isinstance(supplied, Mapping) else supplied
        if prior is None:
            prior = Prior.lognormal(float(np.log(PRIOR_MEDIAN_INTERVALS * interval)), PRIOR_LOG_SD)
        lower, upper = LOWER_INTERVALS * interval, span
        if max(prior.lower, lower) >= min(prior.upper, upper):
            raise ValueError(
                f"noise_timescale prior for modality {m!r} has no support between "
                f"{lower:g} and {upper:g} seconds (a twentieth of the sampling interval and "
                "the longest run)"
            )
        priors[m] = prior.bounded(lower, upper)
    return priors


def starting_values(supplied, priors):
    """Supplied starting timescales, else each prior's median."""
    output = {}
    for m, prior in priors.items():
        value = supplied.get(m)
        if value is None:
            value = float(prior.ppf(0.5))
        elif not prior.lower < value < prior.upper:
            raise ValueError(
                f"noise_timescales starting value for modality {m!r} must lie strictly "
                f"between {prior.lower:g} and {prior.upper:g} seconds"
            )
        output[m] = value
    return output


def white_start(problem, config, seed):
    """Start the search from the white-noise MAP of the same observations.

    Releasing the timescales from a point that already fits the shared signal
    keeps the first start out of optima where slow noise stands in for it. The
    white model has the same parameters without the timescales, which start at
    their supplied or prior-median values. Returns the point, or None when the
    white search found no finite fit, and a record for the restart diagnostics.
    """
    import time

    from .fitting import search
    from .problem import BayesianProblem

    full = problem._full_training_target
    began = time.perf_counter()
    white = BayesianProblem(
        problem.adapter,
        problem.priors,
        anchor=problem.anchor,
        reference_modality=problem.reference_convention["reference_modality"],
        systems=None if full else problem.systems,
        linear_algebra=problem.linear_algebra,
        response_quadrature_order=problem.response_quadrature_order,
        conventions_from=None if full else problem,
    )
    count = len(white.names)
    if white.names != problem.names[:count]:
        raise ValueError("white-noise start has different parameter coordinates")
    best, records = search(white, config, seed)
    record = dict(
        method="white_noise_map",
        starts=len(records),
        available=best is not None,
        seconds=time.perf_counter() - began,
    )
    if best is None:
        return None, record
    record.update(
        objective=best["objective"],
        meets_gradient_tolerance=best["meets_gradient_tolerance"],
    )
    point = problem.initial.copy()
    point[:count] = best["parameters"]
    # A white-noise parameter on a bound starts just inside it, close enough
    # that the search still reports it as resting on that bound.
    lower, upper = np.asarray(problem.bounds, dtype=float).T
    margin = 1e-10 * np.where(
        np.isfinite(upper - lower), upper - lower, np.maximum(1.0, abs(point))
    )
    point = np.where(np.isfinite(lower), np.maximum(point, lower + margin), point)
    point = np.where(np.isfinite(upper), np.minimum(point, upper - margin), point)
    return point, record


def autocorrelation(timescale, interval):
    """Lag-one autocorrelation of an OU process sampled every ``interval``."""
    return float(np.exp(-interval / timescale))


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
        # ``key`` is one feature index, or one per query time.
        return (
            jnp.exp(-jnp.abs(query[:, None] - self.times) / safe)
            * ((jnp.asarray(key)[..., None] == self.keys) & (self.lengths > 0))
            * variance
        )


class EvaluatedNoiseSystem(NoiseSystem):
    """A NoiseSystem at one parameter draw: every modality is OU, in JAX."""

    def correlation(self):
        _, jnp, _, _ = runtime()
        same = self.keys[:, None] == self.keys
        delta = abs(self.times[:, None] - self.times)
        return same * jnp.exp(-delta / self.lengths[:, None])

    def cross(self, query, key, variance):
        _, jnp, _, _ = runtime()
        return (
            jnp.exp(-jnp.abs(query[:, None] - self.times) / self.lengths)
            * (jnp.asarray(key)[..., None] == self.keys)
            * variance
        )


@dataclass(frozen=True)
class EstimatedNoiseSystem:
    """Observed-time structure of the OU precision; timescales arrive later."""

    times: np.ndarray
    keys: np.ndarray
    modalities: np.ndarray
    current: np.ndarray
    previous: np.ndarray
    delta: np.ndarray

    @classmethod
    def prepare(cls, times, keys, all_keys, modalities):
        times = np.asarray(times, float)
        indices = np.array([all_keys.index(k) for k in keys])
        rows_modality = np.array([modalities.index(k[1]) for k in keys])
        current, previous, deltas = [], [], []
        for key in np.unique(indices):
            rows = np.flatnonzero(indices == key)
            rows = rows[np.argsort(times[rows])]
            delta = np.diff(times[rows])
            if np.any(delta <= 0) or not np.isfinite(delta).all():
                raise ValueError("noise covariance requires distinct finite times per feature/run")
            current.extend(rows[1:])
            previous.extend(rows[:-1])
            deltas.extend(delta)
        return cls(
            *(
                readonly_array(a, dtype=a.dtype)
                for a in (
                    times,
                    indices,
                    rows_modality,
                    np.array(current, dtype=int),
                    np.array(previous, dtype=int),
                    np.array(deltas, dtype=float),
                )
            )
        )

    def at(self, timescales):
        """The exact precision for per-modality ``timescales`` (a JAX vector)."""
        _, jnp, _, _ = runtime()
        lengths = jnp.asarray(timescales)[self.modalities]
        tau = lengths[self.current]
        rho = jnp.exp(-self.delta / tau)
        innovation = -jnp.expm1(-2 * self.delta / tau)
        weight = rho**2 / innovation
        diagonal = (
            jnp.ones(len(self.times)).at[self.current].add(weight).at[self.previous].add(weight)
        )
        return EvaluatedNoiseSystem(
            self.times,
            self.keys,
            lengths,
            diagonal,
            self.current,
            self.previous,
            -rho / innovation,
            jnp.log(innovation).sum(),
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
    """Woodbury sufficient statistics with non-diagonal A.T N^-1 A.

    D couples nodes here, so it has no block root and I + K D has no cheap
    symmetric form. Householder QR factors it stably; LU does not.
    """
    from .precision_factor import factorize_general, logdet, score_solve

    _, jnp, _, _ = runtime()
    K, w, residual, variance, nodes = operands(problem, x, run)
    system = problem.noise_system(run, x)
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
    factored = factorize_general(K, D)
    q = score_solve(factored, score)
    mean = K @ q
    error = residual - design_product(mean, w, nodes, count)
    quadratic = error @ system.precision(error, variance) + q @ mean
    value = 0.5 * (
        quadratic
        + jnp.log(variance).sum()
        + system.logdet
        + logdet(factored)
        + len(residual) * np.log(2 * np.pi)
    )
    return value, q, D, factored


def solve(problem, x, run, rhs, factored):
    """Apply the full observation precision to vectors or narrow RHS batches."""
    from .precision_factor import score_solve

    K, weights, _, variance, nodes = operands(problem, x, run)
    system = problem.noise_system(run, x)
    count = len(K) // problem.features
    first = system.precision(rhs, variance)
    score = design_transpose(first, weights, nodes, count)
    q = score_solve(factored, score)
    return first - system.precision(design_product(K @ q, weights, nodes, count), variance)
