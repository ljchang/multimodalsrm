"""Exact Gaussian reduction over repeated shared response functionals."""

from dataclasses import dataclass

import numpy as np

from ._backend import runtime, temporal
from .covariance_pairs import CovariancePairs


@dataclass(frozen=True)
class GroupedSystem:
    times: np.ndarray
    modalities: np.ndarray
    observation_nodes: np.ndarray
    covariance_pairs: CovariancePairs | None

    @classmethod
    def prepare(cls, times, modalities, *, covariance_lookup=True):
        # Exact floating-point equality only: no time bins or modality merging.
        lookup = {}
        nodes = [
            lookup.setdefault((int(m), float(t)), len(lookup)) for m, t in zip(modalities, times)
        ]
        times = np.array([t for _, t in lookup])
        modalities = np.array([m for m, _ in lookup], dtype=int)
        return cls(
            times,
            modalities,
            np.array(nodes, dtype=int),
            CovariancePairs.prepare(times, modalities) if covariance_lookup else None,
        )


def operands(problem, x, run):
    _, jnp, _, _ = runtime()
    nodes = problem.grouped_systems[run]
    ki, gi, _ = problem._packed[run]
    weights, offsets, noise, widths, lags = problem.arrays(x)
    w, variance = weights[ki], noise[gi]
    residual = jnp.asarray(problem.systems[run].values) - offsets[ki]
    mi = nodes.modalities
    if problem.response_quadrature is not None:
        K = problem.temporal_covariance(x, nodes.times, mi, nodes.times, mi)
    elif nodes.covariance_pairs is None:
        K = temporal(
            nodes.times,
            nodes.times,
            widths[mi],
            widths[mi],
            lags[mi],
            lags[mi],
            problem.gp_length_scale(x),
        )
    else:
        K = nodes.covariance_pairs.evaluate(widths, lags, problem.gp_length_scale(x))
    return K, w, residual, variance, nodes.observation_nodes


def factor_arrays(K, w, residual, variance, observation_nodes):
    """Factor I + K D with no divisions by possibly zero loadings.

    K is covariance over shared functionals; D=A.T N^-1 A is diagonal.
    The precision-weighted score is t=A.T N^-1 (y-offset). B = I + K D is
    not symmetric and LU of it is unstable for smooth K, so the similar
    symmetric positive-definite I + D^1/2 K D^1/2 is Cholesky factored.
    """
    from .precision_factor import factorize, logdet, score_solve

    jax, jnp, _, _ = runtime()
    count = len(K)
    precision = jax.ops.segment_sum(w**2 / variance, observation_nodes, num_segments=count)
    score = jax.ops.segment_sum(w * residual / variance, observation_nodes, num_segments=count)
    factor = factorize(K, precision)
    q = score_solve(factor, score)
    mean = K @ q
    # Equivalent to r.T N^-1 r - t.T K q, without subtracting large terms.
    error = residual - w * mean[observation_nodes]
    quadratic = jnp.sum(error**2 / variance) + q @ mean
    nll = 0.5 * (
        quadratic + jnp.log(variance).sum() + logdet(factor) + len(residual) * np.log(2 * np.pi)
    )
    return nll, q, precision, factor


def factor(problem, x, run):
    if problem.noise_timescales:
        from .temporal_noise import factor as correlated_factor

        return correlated_factor(problem, x, run)
    if problem.features > 1:
        from .multifactor import factor as block_factor

        return block_factor(problem, x, run)
    return factor_arrays(*operands(problem, x, run))


def nll(problem, x, run):
    if problem.noise_timescales:
        return factor(problem, x, run)[0]
    if problem.features > 1:
        from .multifactor_score import gaussian_score
    else:
        from .grouped_score import gaussian_score

    return gaussian_score()(*operands(problem, x, run))
