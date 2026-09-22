"""Differentiable SPD inverse/solve with a CPU LAPACK implementation."""

import warnings
from functools import lru_cache

import numpy as np

from ._backend import runtime


def _blas_threads():
    """Largest BLAS pool visible to threadpoolctl, or None when none is loaded."""
    from threadpoolctl import threadpool_info

    pools = [pool["num_threads"] for pool in threadpool_info() if pool.get("user_api") == "blas"]
    return max(pools) if pools else None


def _physical_cores():
    from joblib import cpu_count

    return cpu_count(only_physical_cores=True)


def blas_oversubscription(blas_threads=None, physical_cores=None):
    """Describe a BLAS pool larger than half the physical cores, else return None.

    The CPU factorization runs LAPACK inside an XLA host callback while XLA's
    own pool already occupies the cores. On a 64-core workstation the default
    of one BLAS thread per core slowed a 2,727-dimension factorization by up
    to 10x relative to 16 threads. The threshold is a heuristic warning level,
    not a tuned optimum; the package never changes thread limits itself.
    """
    if blas_threads is None:
        blas_threads = _blas_threads()
    if blas_threads is None:
        return None
    if physical_cores is None:
        physical_cores = _physical_cores()
    if blas_threads <= max(1, physical_cores // 2):
        return None
    return (
        f"CPU symmetric factorization runs LAPACK inside a JAX callback with {blas_threads} "
        f"BLAS threads on {physical_cores} physical cores; JAX already uses its own thread "
        "pool, and this oversubscription can slow the factorization several-fold. Set "
        "OPENBLAS_NUM_THREADS (or the equivalent for your BLAS) before starting Python: 1 "
        "reproduces CI, 8 to 16 was the measured range on a 64-core workstation."
    )


_thread_warning_issued = False


def _warn_thread_oversubscription_once():
    global _thread_warning_issued
    if _thread_warning_issued:
        return
    _thread_warning_issued = True
    message = blas_oversubscription()
    if message is not None:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _cpu_symmetric_solve(matrix, rhs):
    """Pure numerical callback; POTRI avoids solving against a dense identity."""
    from scipy.linalg import cho_solve
    from scipy.linalg.lapack import get_lapack_funcs

    _warn_thread_oversubscription_once()
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
