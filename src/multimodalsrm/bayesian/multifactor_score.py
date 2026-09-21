"""Exact Gaussian score using block sufficient statistics for shared factors."""

from functools import lru_cache

from ._backend import runtime
from .multifactor import factor_arrays
from .multifactor_cholesky import gaussian_terms


@lru_cache(maxsize=1)
def gaussian_score():
    """Differentiate on the symmetric temporal-covariance domain.

    Write K = temporal x I, D = blockdiag(sum w w'/variance), B = I+KD,
    C = B^-1 K, and q = B^-T score. The exact Gaussian scores need only
    the diagonal blocks of C and the factor trace of D B^-1 - q q'.
    Guarded Cholesky inverts well-conditioned precision blocks; singular or
    ill-conditioned blocks retain the LU identities, with no inverse of K or
    D. Thus zero/rank-deficient loadings and singular K remain valid. Both
    paths obtain the scores without cubic dense products or differentiating LU.

    The primal uses the stable residual-form density. This is a custom JVP
    (rather than a stopped derivative) to retain higher-order derivatives.
    """
    jax, jnp, jsp, _ = runtime()

    @jax.custom_jvp
    def nll(temporal, weights, residual, variance, nodes):
        return factor_arrays(temporal, weights, residual, variance, nodes)[0]

    @nll.defjvp
    def differential(primals, tangents):
        temporal, weights, residual, variance, nodes = primals
        dK, dw, dr, dv, _ = tangents
        value, q, diagonal, trace_precision = gaussian_terms(*primals)
        count, factors = len(temporal), weights.shape[1]
        # Keep node-major, factor-minor coordinates throughout.
        q = q.reshape(count, factors)
        mean = temporal @ q
        error = residual - jnp.sum(weights * mean[nodes], axis=1)
        gK = 0.5 * (0.5 * (trace_precision + trace_precision.T) - q @ q.T)
        cw = jnp.einsum("nab,nb->na", diagonal[nodes], weights)
        gw = (cw - error[:, None] * mean[nodes]) / variance[:, None]
        gr = error / variance
        gv = 0.5 * (1 / variance - (error**2 + jnp.sum(weights * cw, axis=1)) / variance**2)
        tangent = jnp.sum(gK * dK) + jnp.sum(gw * dw) + gr @ dr + gv @ dv
        return value, tangent

    return nll
