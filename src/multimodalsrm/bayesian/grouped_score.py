"""Exact Gaussian score for the symmetric grouped covariance domain."""

from functools import lru_cache

from ._backend import runtime
from .grouped import factor_arrays


@lru_cache(maxsize=1)
def gaussian_score():
    """Use one inverse solve instead of differentiating the factorization.

    This rule assumes symmetric covariance K and its symmetric tangents. With
    C=(I+KD)^-1 K and q=(I+DK)^-1 score, dL/dK=(D(I+KD)^-1-qq')/2.
    The observation scores use the posterior mean Kq and diagonal of C.
    No inverse of K or division by D is needed, including at zero loadings
    or singular latent covariance. D(I+KD)^-1 is the symmetric R'M^-1R of
    the Cholesky factor, and the density is the stable residual form.
    """
    from .precision_factor import posterior_terms

    jax, jnp, _, _ = runtime()

    @jax.custom_jvp
    def nll(K, w, residual, variance, nodes):
        return factor_arrays(K, w, residual, variance, nodes)[0]

    @nll.defjvp
    def differential(primals, tangents):
        K, w, residual, variance, nodes = primals
        dK, dw, dr, dv, _ = tangents
        value, q, _, factor = factor_arrays(*primals)
        mean = K @ q
        error = residual - w * mean[nodes]
        G, diagonal = posterior_terms(factor)
        cdiag = diagonal[:, 0, 0][nodes]
        gK = 0.5 * (0.5 * (G + G.T) - jnp.outer(q, q))
        gw = (w * cdiag - error * mean[nodes]) / variance
        gr = error / variance
        gv = 0.5 * (1 / variance - (error**2 + w**2 * cdiag) / variance**2)
        tangent = jnp.sum(gK * dK) + gw @ dw + gr @ dr + gv @ dv
        return value, tangent

    return nll
