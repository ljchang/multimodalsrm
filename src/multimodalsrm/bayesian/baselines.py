"""Fixed-scale Gaussian run offsets, independent of the exact shared GP."""

from collections.abc import Mapping

import numpy as np

from ._backend import runtime


def validate(config, modalities):
    if config is None:
        return {}
    if not isinstance(config, Mapping) or not set(config) <= set(modalities):
        raise ValueError("run baseline SD must map known modalities to finite nonnegative scales")
    result = {}
    for modality, sd in config.items():
        if isinstance(sd, (bool, np.bool_)) or not np.isscalar(sd):
            raise ValueError("run baseline SD must be a finite nonnegative scalar")
        try:
            sd = float(sd)
        except (TypeError, ValueError):
            raise ValueError("run baseline SD must be numeric") from None
        if not np.isfinite(sd) or sd < 0 or sd > np.sqrt(np.finfo(float).max):
            raise ValueError("run baseline SD and variance must be finite and nonnegative")
        if sd:
            result[modality] = sd
    return result


def design(system, scales):
    keys = list(dict.fromkeys(k for k in system.keys if k[1] in scales))
    U = np.zeros((len(system.times), len(keys)))
    for i, key in enumerate(system.keys):
        if key[1] in scales:
            U[i, keys.index(key)] = scales[key[1]]
    return U, keys


def query_cross(problem, run, key):
    """Constant query/donor baseline covariance for this feature and run."""
    _, jnp, _, _ = runtime()
    sd = problem.run_baseline_sd.get(key[1], 0.0) if key else 0.0
    return jnp.asarray([sd**2 if k == key else 0.0 for k in problem.systems[run].keys])


def grouped_factor(problem, x, run):
    """Woodbury update of the exact grouped solve, without a dense observation C.

    U uses fixed prior SDs. Its independent standard-normal coefficients are
    integrated out, adding U U.T separately in every run. The stable energy
    form avoids subtracting large nearly equal quadratic forms.
    """
    from .grouped import factor_arrays, operands

    jax, jnp, jsp, _ = runtime()
    K, w, r, variance, nodes = operands(problem, x, run)
    _, _, precision, lu = factor_arrays(K, w, r, variance, nodes)
    U = jnp.asarray(problem.baseline_designs[run][0])

    def first_solve(rhs):
        score = jax.ops.segment_sum(
            w[:, None] * rhs / variance[:, None], nodes, num_segments=len(K)
        )
        q = jsp.linalg.lu_solve(lu, score, trans=1)
        return (rhs - w[:, None] * (K @ q)[nodes]) / variance[:, None]

    rhs = jnp.column_stack((U, r))
    solved = first_solve(rhs)
    # Refine the few right-hand sides using C0 products at grouped dimension.
    # Low noise otherwise amplifies cancellation in N^-1(rhs - A K q).
    for _ in range(2):
        node_score = jax.ops.segment_sum(w[:, None] * solved, nodes, num_segments=len(K))
        error = rhs - variance[:, None] * solved - w[:, None] * (K @ node_score)[nodes]
        solved = solved + first_solve(error)
    V, alpha0 = solved[:, :-1], solved[:, -1]
    # Derive GP coefficients from the refined observation inverse. Accumulating
    # LU coefficients separately leaves them inconsistent at high signal/noise.
    coefficients = jax.ops.segment_sum(w[:, None] * solved, nodes, num_segments=len(K))
    qU, q0 = coefficients[:, :-1], coefficients[:, -1]
    small = jnp.eye(U.shape[1]) + U.T @ V
    L = jnp.linalg.cholesky(0.5 * (small + small.T))
    beta = jsp.linalg.cho_solve((L, True), U.T @ alpha0)

    adjusted = r - U @ beta
    score = jax.ops.segment_sum(w * adjusted / variance, nodes, num_segments=len(K))
    q = jsp.linalg.lu_solve(lu, score, trans=1)
    mean = K @ q
    error = adjusted - w * mean[nodes]
    quadratic = jnp.sum(error**2 / variance) + q @ mean + beta @ beta
    diagonal = jnp.diag(lu[0])
    sign = jnp.prod(jnp.sign(diagonal)) * (-1.0) ** jnp.sum(lu[1] != jnp.arange(len(K)))
    logdet = jnp.where(sign > 0, jnp.log(jnp.abs(diagonal)).sum(), jnp.inf)
    nll = (
        0.5 * (quadratic + jnp.log(variance).sum() + logdet + len(r) * np.log(2 * np.pi))
        + jnp.log(jnp.diag(L)).sum()
    )
    return nll, beta, L, q0, qU, precision, lu
