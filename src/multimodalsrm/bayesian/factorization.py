"""Differentiable SPD inverse/solve with a CPU LAPACK implementation."""

from functools import lru_cache

import numpy as np

from ._backend import runtime


def _cpu_symmetric_solve(matrix, rhs):
    """Pure numerical callback; POTRI avoids solving against a dense identity."""
    from scipy.linalg import cho_solve
    from scipy.linalg.lapack import get_lapack_funcs

    # Callback inputs may alias read-only JAX buffers. Only overwrite our copy.
    matrix = np.array(matrix, order="F", copy=True)
    rhs = np.asarray(rhs)
    failure = (
        np.full_like(matrix, np.nan),
        np.full_like(rhs, np.nan),
        np.asarray(np.nan, dtype=matrix.dtype),
    )
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        return failure
    potrf, potri = get_lapack_funcs(("potrf", "potri"), (matrix,))
    chol, info = potrf(matrix, lower=True, overwrite_a=True, clean=False)
    if info != 0:
        return failure
    logdet = 2 * np.log(np.diag(chol)).sum()
    solution = cho_solve((chol, True), rhs, check_finite=False)
    inverse, info = potri(chol, lower=True, overwrite_c=True)
    if info != 0:
        return failure
    inverse = np.tril(inverse) + np.tril(inverse, -1).T
    return inverse, solution, np.asarray(logdet, dtype=matrix.dtype)


@lru_cache(maxsize=1)
def symmetric_solve():
    """Return inverse, solution and log determinant on the symmetric domain.

    CPU uses POTRF/POTRI on sufficiently large matrices; other devices and
    small matrices use native JAX Cholesky. Selection is made at lowering,
    not by inspecting the default device. Implicit derivatives preserve JVP,
    reverse-mode and higher-order differentiation across the pure callback.
    Non-SPD or nonfinite inputs yield NaNs for the caller's guarded fallback.
    """
    jax, jnp, jsp, _ = runtime()

    def native(matrix, rhs):
        chol = jnp.linalg.cholesky(matrix)
        return (
            jsp.linalg.cho_solve((chol, True), jnp.eye(len(matrix), dtype=matrix.dtype)),
            jsp.linalg.cho_solve((chol, True), rhs),
            2 * jnp.log(jnp.diag(chol)).sum(),
        )

    def cpu(matrix, rhs):
        shapes = (
            jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            jax.ShapeDtypeStruct((), matrix.dtype),
        )
        return jax.pure_callback(
            _cpu_symmetric_solve, shapes, matrix, rhs, vmap_method="sequential"
        )

    @jax.custom_jvp
    def solve(matrix, rhs):
        matrix = (matrix + matrix.T) * 0.5
        if len(matrix) < 128:
            return native(matrix, rhs)
        return jax.lax.platform_dependent(matrix, rhs, cpu=cpu, default=native)

    @solve.defjvp
    def differential(primals, tangents):
        matrix, rhs = primals
        dmatrix, drhs = tangents
        dmatrix = (dmatrix + dmatrix.T) * 0.5
        inverse, solution, logdet = solve(matrix, rhs)
        valid = (
            jnp.all(jnp.isfinite(inverse)) & jnp.all(jnp.isfinite(solution)) & jnp.isfinite(logdet)
        )

        # Invalid solves have no derivative on the SPD domain. Their caller
        # discards them; an explicit zero branch prevents 0*NaN contamination
        # when reverse-mode differentiates through that discarded computation.
        def finite(_):
            return (
                -inverse @ dmatrix @ inverse,
                inverse @ (drhs - dmatrix @ solution),
                jnp.sum(inverse.T * dmatrix),
            )

        derivative = jax.lax.cond(
            valid,
            finite,
            lambda _: (
                jnp.zeros_like(inverse),
                jnp.zeros_like(solution),
                jnp.zeros_like(logdet),
            ),
            operand=None,
        )
        return (inverse, solution, logdet), derivative

    return solve
