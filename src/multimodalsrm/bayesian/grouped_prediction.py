"""Grouped marginal prediction with one factorization per parameter draw.

Reuse native-time covariance across output features within a result. Retain
only one draw's conditioning state and narrow query batches on the device;
no cache survives the call or can outlive its conditioning observations.
"""

import numpy as np

from ._backend import runtime
from .grouped import factor


def project(problem, draws, run, times, *, keys, include_noise, directions=None):
    """Return draw/time/feature moments for one modality or latent directions."""
    if problem.linear_algebra != "grouped" or problem.run_baseline_sd:
        raise ValueError("Expected grouped algebra without run baselines")
    if not keys or any(key is not None and key not in problem.keys for key in keys):
        raise ValueError("query mapping is unavailable")
    modalities = {None if key is None else key[1] for key in keys}
    if len(modalities) != 1:
        raise ValueError("grouped prediction requires one query modality")
    modality = next(iter(modalities))
    latent = modality is None
    if latent:
        directions = np.broadcast_to(
            np.eye(problem.features)[: len(keys)] if directions is None else directions,
            (len(draws), len(keys), problem.features),
        )
        if not np.isfinite(directions).all():
            raise ValueError("latent directions must be finite")
    jax, jnp, jsp, _ = runtime()
    system = problem.systems[run]
    nodes = problem.grouped_systems[run]
    count = len(nodes.times)
    indices = nodes.observation_nodes
    target = np.zeros(len(keys), int) if latent else np.array([problem.keys.index(k) for k in keys])
    groups = jnp.array([problem.groups.index(key[:2]) for key in problem.keys])
    query_modality = -1 if latent else problem.modalities.index(modality)
    # White-noise queries are replicas. OU queries share private residuals only
    # with observations of the same feature in this conditioning run.
    observed_keys = set(system.keys)
    correlated = (
        include_noise
        and modality in problem.noise_timescales
        and any(key in observed_keys for key in keys)
    )

    @jax.jit
    def prepare(x):
        _, q, precision, lu = factor(problem, x, run)
        weights, offsets, noise, _, _ = problem.arrays(x)
        common = (
            q,
            precision,
            lu,
            weights.reshape(len(problem.keys), problem.features),
            offsets,
            noise,
        )
        if not correlated:
            return common, ()
        from .temporal_noise import design_product, operands

        K, w, residual, variance, _ = operands(problem, x, run)
        noise_system = problem.noise_systems[run]
        alpha = noise_system.precision(
            residual - design_product(K @ q, w, indices, count), variance
        )
        return common, (K, w, variance, alpha)

    @jax.jit
    def temporal(x, query):
        qm = np.full(query.shape, query_modality, dtype=int)
        cross = problem.temporal_covariance(x, query, qm, nodes.times, nodes.modalities)
        prior = problem.temporal_covariance(x, jnp.zeros(1), qm[:1], jnp.zeros(1), qm[:1])[0, 0]
        return cross, prior

    @jax.jit
    def batch(prepared, cross, prior, query, output_keys, loading_directions):
        (q, precision, lu, weights, offsets, noise), private = prepared
        loading = loading_directions if latent else weights[output_keys]
        offset = 0.0 if latent else offsets[output_keys]
        measurement = noise[groups[output_keys]] if include_noise and not latent else 0.0
        if correlated:
            from .temporal_noise import design_product, design_transpose

            K, w, variance, alpha = private
            noise_system = problem.noise_systems[run]
            observed_cross = cross[:, indices] * (loading @ w.T)
            safe = np.where(noise_system.lengths > 0, noise_system.lengths, 1.0)
            observed_cross += (
                jnp.exp(-jnp.abs(query[:, None] - system.times) / safe)
                * ((output_keys[:, None] == noise_system.keys) & (noise_system.lengths > 0))
                * variance
            )
            rhs = observed_cross.T
            first = noise_system.precision(rhs, variance)
            score = design_transpose(first, w, indices, count)
            coefficient = jsp.linalg.lu_solve(lu, score, trans=1)
            inverse = first - noise_system.precision(
                design_product(K @ coefficient, w, indices, count), variance
            )
            mean = observed_cross @ alpha + offset
            reduction = jnp.sum(rhs * inverse, axis=0)
        else:
            shared_cross = (cross[:, :, None] * loading[:, None, :]).reshape(len(query), -1)
            mean = shared_cross @ q + offset
            solved = jsp.linalg.lu_solve(lu, shared_cross.T)
            weighted = precision[:, None] * solved if precision.ndim == 1 else precision @ solved
            reduction = jnp.sum(shared_cross.T * weighted, axis=0)
        scale = prior * jnp.sum(loading**2, axis=1) + measurement
        return mean, scale - reduction, scale

    means = np.empty((len(draws), len(times), len(keys)))
    variances = np.empty_like(means)
    if not len(times) or not len(draws):
        return means, variances
    unique, lookup = np.unique(times, return_inverse=True)
    time_batch = min(128, len(unique))
    query_batch = min(32, time_batch * len(keys))
    for d, x in enumerate(draws):
        prepared = prepare(jnp.asarray(x))
        unique_mean = np.empty((len(unique), len(keys)))
        unique_variance = np.empty_like(unique_mean)
        for start in range(0, len(unique), time_batch):
            query = unique[start : start + time_batch]
            padded_query = np.pad(query, (0, time_batch - len(query)), mode="edge")
            cross, prior = temporal(jnp.asarray(x), jnp.asarray(padded_query))
            for first in range(0, len(query) * len(keys), query_batch):
                flat = np.arange(first, min(first + query_batch, len(query) * len(keys)))
                padded = np.pad(flat, (0, query_batch - len(flat)), mode="edge")
                ti, fi = padded // len(keys), padded % len(keys)
                direction = (
                    directions[d, fi] if latent else np.zeros((query_batch, problem.features))
                )
                mean, variance, scale = (
                    np.asarray(v)
                    for v in batch(prepared, cross[ti], prior, query[ti], target[fi], direction)
                )
                if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                    raise FloatingPointError("nonfinite posterior prediction")
                if np.any(variance < -1e-9 * np.maximum(1.0, np.abs(scale))):
                    raise FloatingPointError("materially negative conditional variance")
                rows, columns = start + ti[: len(flat)], fi[: len(flat)]
                unique_mean[rows, columns] = mean[: len(flat)]
                unique_variance[rows, columns] = np.maximum(variance[: len(flat)], 0.0)
        means[d], variances[d] = unique_mean[lookup], unique_variance[lookup]
        del prepared
    return means, variances
