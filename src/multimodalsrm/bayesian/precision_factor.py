"""Backward-stable factorizations of I + K D for the grouped Gaussian solves.

K is the covariance over shared functionals and D = A.T N^-1 A the positive
semidefinite precision of the observations about them. B = I + K D is not
symmetric, and LU with partial pivoting on it has unbounded element growth
when K is smooth, so B is never LU factored.

White noise makes D block diagonal, one block per node, and K = temporal x I.
With any root D = R.T R,

    M = I + R K R.T              symmetric, eigenvalues >= 1
    log det B = log det M
    (I + D K)^-1 R.T u = R.T M^-1 u
    D (I + K D)^-1 = R.T M^-1 R

so one Cholesky factor of M gives every grouped quantity. Nothing is divided
by a loading and K is never inverted: zero or rank-deficient blocks and a
singular K stay valid. R is not a smooth function of D where a block is
singular, so the Cholesky factor and root are a derivative-free cache.
Derivatives follow from the identities in K and D alone, which keeps every
order exact at zero loadings.

Correlated noise couples the nodes in D. No root of that D is available
below cubic cost in its full size, so B itself is factored by Householder QR,
which is backward stable without pivoting growth.
"""

from functools import lru_cache
from typing import NamedTuple

from ._backend import runtime


class PrecisionFactor(NamedTuple):
    """Cholesky cache for block-diagonal D, with the operands it was built from."""

    cholesky: object  # lower factor of M, node-major then factor-minor
    root: object  # (nodes, factors, factors) blocks R with D = R.T R
    coroot: object  # blocks T with R.T T v = v for every v in range(D)
    temporal: object
    blocks: object


class OrthogonalFactor(NamedTuple):
    """Householder QR of B for a precision that is not block diagonal."""

    orthogonal: object
    triangular: object
    precision: object


def factorize(temporal, blocks):
    """Factor M = I + R (temporal x I) R.T for PSD blocks, or a diagonal D."""
    jax, jnp, _, _ = runtime()
    if blocks.ndim == 1:
        blocks = blocks[:, None, None]
    K, D = jax.lax.stop_gradient((temporal, blocks))
    count, factors = D.shape[:2]
    if factors == 1:
        eigenvalues, vectors = D[:, :, 0], jnp.ones_like(D)
    else:
        eigenvalues, vectors = jnp.linalg.eigh(0.5 * (D + D.swapaxes(-1, -2)))
    # A PSD block has no negative eigenvalue; eigh may return one at roundoff.
    positive = eigenvalues > 0
    scale = jnp.sqrt(jnp.where(positive, eigenvalues, 0.0))
    inverse = jnp.where(positive, 1 / jnp.where(positive, scale, 1.0), 0.0)
    transposed = vectors.swapaxes(-1, -2)
    root = scale[:, :, None] * transposed
    coroot = inverse[:, :, None] * transposed
    size = count * factors
    if factors == 1:
        M = scale * K * scale.T
    else:
        # M[ia, jb] = K[i, j] (R[i] R[j].T)[a, b]: a Gram matrix times repeated K.
        flat = root.reshape(size, factors)
        M = (flat @ flat.T) * jnp.repeat(jnp.repeat(K, factors, axis=0), factors, axis=1)
    M = 0.5 * (M + M.T) + jnp.eye(size, dtype=M.dtype)
    return PrecisionFactor(jnp.linalg.cholesky(M), root, coroot, temporal, blocks)


def factorize_general(K, D):
    """Factor B = I + K D by QR for a dense symmetric PSD precision D."""
    _, jnp, _, _ = runtime()
    orthogonal, triangular = jnp.linalg.qr(jnp.eye(len(K), dtype=K.dtype) + K @ D)
    return OrthogonalFactor(orthogonal, triangular, D)


def _blocks(matrices, rhs):
    """Apply one block to each node of a node-major vector or matrix."""
    _, jnp, _, _ = runtime()
    count, factors = matrices.shape[:2]
    if factors == 1:
        return matrices.reshape((count,) + (1,) * (rhs.ndim - 1)) * rhs
    return jnp.einsum("iab,ibm->iam", matrices, rhs.reshape(count, factors, -1)).reshape(rhs.shape)


def _covariance(temporal, rhs):
    """Apply temporal x I without forming the Kronecker product."""
    return (temporal @ rhs.reshape(len(temporal), -1)).reshape(rhs.shape)


def _cholesky_solve(factor, rhs):
    _, _, jsp, _ = runtime()
    return jsp.linalg.cho_solve((factor.cholesky, True), rhs.reshape(len(rhs), -1)).reshape(
        rhs.shape
    )


def _weighted(factor, rhs):
    """R.T M^-1 R rhs from the cache alone."""
    transposed = factor.root.swapaxes(-1, -2)
    return _blocks(transposed, _cholesky_solve(factor, _blocks(factor.root, rhs)))


def _transpose_system(factor, rhs):
    """Solve (I + D K) x = rhs for the tangent of a grouped solve.

    custom_linear_solve keeps the implicit derivative in K and D through
    every later transposition and differentiation. The cache only applies
    the inverse, (I + D K)^-1 = I - R.T M^-1 R K.
    """
    jax, _, _, _ = runtime()
    temporal, blocks = factor.temporal, factor.blocks

    def matvec(v):
        return v + _blocks(blocks, _covariance(temporal, v))

    def solve(_, b):
        return b - _weighted(factor, _covariance(temporal, b))

    def transpose_solve(_, b):
        return b - _covariance(temporal, _weighted(factor, b))

    return jax.lax.custom_linear_solve(matvec, rhs, solve, transpose_solve)


@lru_cache(maxsize=1)
def _operations():
    jax, jnp, _, _ = runtime()

    @jax.custom_jvp
    def precision_solve(factor, rhs):
        return _weighted(factor, rhs)

    @precision_solve.defjvp
    def precision_solve_jvp(primals, tangents):
        factor, rhs = primals
        (_, _, _, dK, dD), drhs = tangents
        answer = precision_solve(factor, rhs)
        # y = D (I + K D)^-1 rhs solves (I + D K) y = D rhs.
        change = _blocks(dD, rhs - _covariance(factor.temporal, answer)) + _blocks(
            factor.blocks, drhs - _covariance(dK, answer)
        )
        return answer, _transpose_system(factor, change)

    @jax.custom_jvp
    def score_solve(factor, score):
        transposed = factor.root.swapaxes(-1, -2)
        return _blocks(transposed, _cholesky_solve(factor, _blocks(factor.coroot, score)))

    @score_solve.defjvp
    def score_solve_jvp(primals, tangents):
        factor, score = primals
        (_, _, _, dK, dD), dscore = tangents
        answer = score_solve(factor, score)
        # q solves (I + D K) q = score.
        change = (
            dscore
            - _blocks(dD, _covariance(factor.temporal, answer))
            - _blocks(factor.blocks, _covariance(dK, answer))
        )
        return answer, _transpose_system(factor, change)

    @jax.custom_jvp
    def logdet(factor):
        return 2 * jnp.log(jnp.diag(factor.cholesky)).sum()

    @logdet.defjvp
    def logdet_jvp(primals, tangents):
        (factor,) = primals
        ((_, _, _, dK, dD),) = tangents
        # d log det B = tr(D B^-1 dK) + tr(B^-1 K dD).
        trace, diagonal = posterior_terms(factor)
        return logdet(factor), jnp.sum(trace * dK) + jnp.sum(diagonal * dD)

    return precision_solve, score_solve, logdet


def precision_solve(factor, rhs):
    """Return D (I + K D)^-1 rhs, the symmetric R.T M^-1 R rhs, for any rhs."""
    if isinstance(factor, OrthogonalFactor):
        _, _, jsp, _ = runtime()
        solved = jsp.linalg.solve_triangular(factor.triangular, factor.orthogonal.T @ rhs)
        return factor.precision @ solved
    return _operations()[0](factor, rhs)


def score_solve(factor, score):
    """Return (I + D K)^-1 score for a precision-weighted score A.T N^-1 r.

    Every such score lies in the range of D, so the Cholesky form
    R.T M^-1 T score involves no subtraction and no division by a loading.
    """
    if isinstance(factor, OrthogonalFactor):
        _, _, jsp, _ = runtime()
        return factor.orthogonal @ jsp.linalg.solve_triangular(factor.triangular, score, trans="T")
    return _operations()[1](factor, score)


def logdet(factor):
    """Return log det (I + K D), which is positive for PSD K and D."""
    if isinstance(factor, OrthogonalFactor):
        _, jnp, _, _ = runtime()
        return jnp.log(jnp.abs(jnp.diag(factor.triangular))).sum()
    return _operations()[2](factor)


def posterior_terms(factor):
    """Return the factor trace of D (I + K D)^-1 and diagonal blocks of (I + K D)^-1 K.

    These are the contractions the exact Gaussian scores need. The first is
    (nodes, nodes), the second (nodes, factors, factors).
    """
    _, jnp, _, _ = runtime()
    temporal = factor.temporal
    count, factors = factor.blocks.shape[:2]
    weighted = precision_solve(factor, jnp.eye(count * factors, dtype=temporal.dtype))
    if factors == 1:
        reduction = jnp.sum(temporal.T * (weighted @ temporal), axis=0)
        return weighted, (jnp.diag(temporal) - reduction)[:, None, None]
    weighted = weighted.reshape(count, factors, count, factors)
    trace = jnp.einsum("iaja->ij", weighted)
    product = (temporal @ weighted.reshape(count, -1)).reshape(count, factors, count, factors)
    prior = jnp.diag(temporal)[:, None, None] * jnp.eye(factors, dtype=temporal.dtype)
    diagonal = prior - jnp.einsum("iajb,ji->iab", product, temporal)
    return trace, 0.5 * (diagonal + diagonal.swapaxes(-1, -2))
