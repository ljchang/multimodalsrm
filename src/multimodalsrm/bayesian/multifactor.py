"""Exact block sufficient statistics for a vector of shared GP factors."""

import numpy as np

from ._backend import runtime


def validate_anchors(problem, anchors):
    """Resolve declared training feature keys before any optimization."""
    if anchors is None:
        return None
    if (
        not isinstance(anchors, (tuple, list))
        or len(anchors) != problem.features
        or any(not isinstance(a, (tuple, list)) or len(a) != 3 for a in anchors)
    ):
        raise ValueError("factor_anchors must contain one training feature key per factor")
    keys = [tuple(a) for a in anchors]
    if any(any(isinstance(v, (list, dict, set)) for v in a) for a in keys):
        raise ValueError("factor_anchors must contain observed feature keys")
    if len(set(keys)) != len(keys):
        raise ValueError("factor_anchors must be distinct")
    if keys[0] != problem.anchor:
        raise ValueError("factor_anchors must start with anchor")
    systems = problem.adapter._systems(problem.adapter._training_data, problem.adapter.domains_)[0]
    observed = {key for s in systems.values() for key in s.keys}
    if any(key not in observed for key in keys):
        raise ValueError("factor_anchors must name observed training features")
    return keys


def orientation(problem, x, anchors):
    """Fixed training QR coordinates; MAP parameter records remain untouched."""
    keys = validate_anchors(problem, anchors)
    if keys is None:
        return dict(
            method="optimizer_coordinates",
            anchors=[],
            rotation=np.eye(problem.features).tolist(),
            parameter_coordinates="raw_optimizer",
            reported_loadings="W_raw",
            reported_latent="Z_raw",
            estimated_from="training_MAP_only",
            physiological_identification=False,
        )
    weights = np.asarray(problem.arrays(x)[0])
    block = weights[[problem.keys.index(key) for key in keys]]
    singular = np.linalg.svd(block, compute_uv=False)
    if not np.isfinite(block).all() or singular[0] == 0 or singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("factor anchor loading block is rank deficient; cannot orient factors")
    Q, R = np.linalg.qr(block.T)
    Q *= np.where(np.diag(R) >= 0, 1.0, -1.0)[None, :]
    return dict(
        method="training_anchor_QR_positive_diagonal",
        anchors=[list(k) for k in keys],
        rotation=Q.tolist(),
        anchor_singular_values=singular.tolist(),
        relative_rank_tolerance=1e-10,
        parameter_coordinates="raw_optimizer",
        reported_loadings="W_raw @ rotation",
        reported_latent="Z_raw @ rotation",
        estimated_from="training_MAP_only",
        physiological_identification=False,
    )


def factor(problem, x, run):
    """Retain the full precision-matrix contract for conditional prediction."""
    from .grouped import operands

    _, jnp, _, _ = runtime()
    value, q, blocks, lu = factor_arrays(*operands(problem, x, run))
    count, features = blocks.shape[:2]
    precision = jnp.einsum("ij,iab->iajb", jnp.eye(count), blocks).reshape(
        count * features, count * features
    )
    return value, q, precision, lu


def factor_arrays(temporal, weights, residual, variance, nodes):
    """Use block precision without inverting a possibly singular GP covariance."""
    blocks, score = block_statistics(weights, residual, variance, nodes, len(temporal))
    return factor_statistics(temporal, weights, residual, variance, nodes, blocks, score)


def block_statistics(weights, residual, variance, nodes, count):
    jax, jnp, jsp, _ = runtime()
    blocks = jax.ops.segment_sum(
        weights[:, :, None] * weights[:, None, :] / variance[:, None, None],
        nodes,
        num_segments=count,
    )
    score = jax.ops.segment_sum(
        weights * (residual / variance)[:, None], nodes, num_segments=count
    ).reshape(-1)
    return blocks, score


def factor_statistics(temporal, weights, residual, variance, nodes, blocks, score):
    """LU reference using already constructed exact sufficient statistics."""
    _, jnp, jsp, _ = runtime()
    count, features = len(temporal), weights.shape[1]
    # Coordinates are ordered node-major, then factor within each node.
    size = count * features
    # (K_time x I) D has block (i,j) = K_time[i,j] D[j]. Construct
    # those blocks directly instead of a cubic dense matrix multiplication.
    B = jnp.eye(size) + jnp.einsum("ij,jab->iajb", temporal, blocks).reshape(size, size)
    lu, pivots = jsp.linalg.lu_factor(B)
    diagonal = jnp.diag(lu)
    sign = jnp.prod(jnp.sign(diagonal)) * (-1.0) ** jnp.sum(pivots != jnp.arange(size))
    logdet = jnp.where(sign > 0, jnp.log(jnp.abs(diagonal)).sum(), jnp.inf)
    q = jsp.linalg.lu_solve((lu, pivots), score, trans=1)
    mean = temporal @ q.reshape(count, features)
    error = residual - jnp.sum(weights * mean[nodes], axis=1)
    quadratic = jnp.sum(error**2 / variance) + jnp.sum(q.reshape(count, features) * mean)
    nll = 0.5 * (quadratic + jnp.log(variance).sum() + logdet + len(residual) * np.log(2 * np.pi))
    return nll, q, blocks, (lu, pivots)
