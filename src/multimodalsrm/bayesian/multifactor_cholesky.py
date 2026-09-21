"""Guarded symmetric factorization of the exact multifactor GP likelihood."""

import numpy as np

from ._backend import runtime
from .factorization import symmetric_solve
from .multifactor import block_statistics, factor_statistics


def gaussian_terms(temporal, weights, residual, variance, nodes):
    """Return density, q, posterior diagonal blocks and precision factor trace.

    For positive-definite precision blocks D, S=K+D^-1 is SPD even if K is
    singular. Then q=S^-1 D^-1 score, D(I+KD)^-1=S^-1, and
    C=(I+KD)^-1 K=D^-1 S^-1 K. All block contractions are exact.
    Ill-conditioned or singular D uses the existing LU identities unchanged.
    The guard selects algebra only: no loading truncation, jitter or ridge.
    """
    jax, jnp, jsp, _ = runtime()
    count, factors = len(temporal), weights.shape[1]
    size = count * factors
    blocks, score = block_statistics(weights, residual, variance, nodes, count)

    def reference(_):
        value, q, _, lu = factor_statistics(
            temporal, weights, residual, variance, nodes, blocks, score
        )
        inverse = jsp.linalg.lu_solve(lu, jnp.eye(size)).reshape(count, factors, count, factors)
        diagonal = jnp.einsum("iajb,ji->iab", inverse, temporal)
        trace = jnp.einsum("iab,ibja->ij", blocks, inverse)
        return value, q.reshape(count, factors), diagonal, trace

    def cholesky(_):
        inverse_blocks = jnp.linalg.inv(blocks)
        inverse_blocks = 0.5 * (inverse_blocks + inverse_blocks.swapaxes(-1, -2))
        symmetric = jnp.kron(temporal, jnp.eye(factors)) + jnp.einsum(
            "ij,iab->iajb", jnp.eye(count), inverse_blocks
        ).reshape(size, size)
        rhs = jnp.einsum("iab,ib->ia", inverse_blocks, score.reshape(count, factors)).reshape(-1)
        precision, q, logdet_s = symmetric_solve()(symmetric, rhs)
        solved = (
            jnp.all(jnp.isfinite(precision)) & jnp.all(jnp.isfinite(q)) & jnp.isfinite(logdet_s)
        )
        # Only finite solve outputs enter calculations preceding the fallback
        # conditional, so reverse derivatives of an unselected path stay finite.
        precision = jnp.where(solved, precision, jnp.zeros_like(precision))
        q = jnp.where(solved, q, jnp.zeros_like(q))
        logdet_s = jnp.where(solved, logdet_s, 0.0)
        q = q.reshape(count, factors)
        precision = precision.reshape(count, factors, count, factors)
        diagonal = jnp.einsum(
            "iab,ibc->iac",
            inverse_blocks,
            jnp.einsum("iajb,ji->iab", precision, temporal),
        )
        trace = jnp.einsum("iaja->ij", precision)
        mean = temporal @ q
        error = residual - jnp.sum(weights * mean[nodes], axis=1)
        logdet_d = jnp.linalg.slogdet(blocks)[1].sum()
        value = 0.5 * (
            jnp.sum(error**2 / variance)
            + jnp.sum(q * mean)
            + jnp.log(variance).sum()
            + logdet_s
            + logdet_d
            + len(residual) * np.log(2 * np.pi)
        )
        result = value, q, diagonal, trace
        valid = (
            solved
            & jnp.isfinite(value)
            & jnp.all(jnp.isfinite(q))
            & jnp.all(jnp.isfinite(diagonal))
            & jnp.all(jnp.isfinite(trace))
        )
        return jax.lax.cond(valid, lambda _: result, reference, operand=None)

    eigenvalues = jnp.linalg.eigvalsh(blocks)
    # Do not invert nearly singular or uniformly tiny loading directions:
    # derivatives of D^-1 can overflow even when D has a good condition ratio.
    # The absolute floor is a solver choice, not a change to precision or density.
    usable = jnp.all(jnp.isfinite(eigenvalues)) & jnp.all(
        eigenvalues[:, 0] > np.sqrt(np.finfo(float).eps) * jnp.maximum(eigenvalues[:, -1], 1.0)
    )
    return jax.lax.cond(usable, cholesky, reference, operand=None)
