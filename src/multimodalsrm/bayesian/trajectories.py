"""Conditional joint latent sampling from retained physical parameter draws."""

import numpy as np

from ._backend import runtime
from .posterior_coordinates import rotations
from .prediction import queries, result_metadata, selected_draws
from .response_scope import validate_posterior_response_target
from .results import TrajectorySamples


def joint_moments(problem, run, times):
    """Build a draw-wise evaluator in time-major, then reported-factor order.

    The grouped expression is C = K** - K*g D (I + Kg D)^-1 Kg*,
    retaining every cross-time/factor entry without inverting Kg. Rotations
    act on the query cross-covariance; the isotropic latent prior is invariant.
    """
    jax, jnp, jsp, _ = runtime()
    features = problem.features
    query_modality = np.full(len(times), -1, dtype=int)
    size = len(times) * features
    system = problem.systems[run]
    ki, _, mi = problem._packed[run]

    def evaluate(x, rotation):
        prior = jnp.kron(
            problem.temporal_covariance(x, times, query_modality, times, query_modality),
            jnp.eye(features),
        )
        if problem.linear_algebra == "grouped":
            from .grouped import factor

            _, q, precision, lu = factor(problem, x, run)
            nodes = problem.grouped_systems[run]
            temporal = problem.temporal_covariance(
                x,
                times,
                query_modality,
                nodes.times,
                nodes.modalities,
            )
            cross = jnp.kron(temporal, rotation.T)
            mean = cross @ q
            solved = jsp.linalg.lu_solve(lu, cross.T)
            weighted = precision[:, None] * solved if features == 1 else precision @ solved
            covariance = prior - cross @ weighted
        else:
            weights, offsets, _, _, _ = problem.arrays(x)
            reported = weights.reshape(-1, features) @ rotation
            temporal = problem.temporal_covariance(
                x,
                times,
                query_modality,
                system.times,
                mi,
            )
            cross = (temporal[:, None, :] * reported[ki].T[None, :, :]).reshape(size, -1)
            L = jnp.linalg.cholesky(problem.covariance(x, run))
            whitened = jsp.linalg.solve_triangular(
                L,
                jnp.asarray(system.values) - offsets[ki],
                lower=True,
            )
            projected = jsp.linalg.solve_triangular(L, cross.T, lower=True)
            mean = projected.T @ whitened
            covariance = prior - projected.T @ projected
        return mean.reshape(len(times), features), covariance

    return jax.jit(evaluate)


def covariance_root(covariance):
    """Square root allowing roundoff-sized negative modes, never added jitter.

    Unit prior variance sets the cancellation scale. Reject material asymmetry
    or negative eigenvalues rather than silently changing the distribution.
    """
    covariance = np.asarray(covariance, float)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("nonfinite joint latent covariance")
    tolerance = 64 * np.finfo(float).eps * len(covariance)
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise FloatingPointError("materially asymmetric joint latent covariance")
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError("materially negative joint latent covariance")
    return vectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]


def sample_latent(
    model,
    *,
    times,
    max_draws,
    draws_per_parameter,
    random_state,
    max_joint_size,
):
    problem = model.problem_
    if model.configuration_["inference"] != "posterior":
        raise ValueError("sample_latent requires a fitted parameter posterior")
    if (
        problem.linear_algebra not in ("dense", "grouped")
        or problem.run_baseline_sd
        or problem.noise_timescales
    ):
        raise ValueError(
            "sample_latent supports dense/grouped "
            "posteriors with independent noise and no run baselines"
        )
    validate_posterior_response_target(problem)
    for name, value in (
        ("draws_per_parameter", draws_per_parameter),
        ("max_joint_size", max_joint_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    requested = queries(model, times)
    features = problem.features
    for run, query in requested.items():
        if len(query) * features > max_joint_size:
            raise ValueError(
                f"joint query for run {run!r} exceeds max_joint_size={max_joint_size}; "
                "use fewer query points or explicitly raise the limit"
            )
    draws, indices = selected_draws(model, max_draws, return_indices=True)
    orientations = (
        rotations(problem, draws, model._factor_anchor_keys_, indices)
        if features > 1
        else np.ones((len(draws), 1, 1))
    )
    rng = np.random.default_rng(random_state)
    results = {}
    for run, query in requested.items():
        a, b = model.prediction_runs_[run]
        valid = (query >= a) & (query <= b)
        samples = np.full((len(draws), draws_per_parameter, len(query), features), np.nan)
        if valid.any():
            evaluate = joint_moments(problem, run, query[valid])
            for i, (x, rotation) in enumerate(zip(draws, orientations)):
                try:
                    mean, covariance = (np.asarray(v) for v in evaluate(x, rotation))
                    if not np.isfinite(mean).all():
                        raise FloatingPointError("nonfinite joint latent mean")
                    root = covariance_root(covariance)
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    chain, draw = indices[i]
                    raise FloatingPointError(
                        f"{error} at run {run!r}, chain {chain}, draw {draw}"
                    ) from error
                standard = rng.standard_normal((draws_per_parameter, root.shape[0]))
                paths = (standard @ root.T + mean.ravel()).reshape(
                    draws_per_parameter,
                    int(valid.sum()),
                    features,
                )
                samples[i][:, valid] = paths
        metadata = result_metadata(
            model,
            run,
            query,
            draws,
            [None],
            include_noise=False,
            draw_indices=indices,
        )
        metadata.update(
            moment_scope="joint_time_factor_samples",
            sample_axes=["parameter_draw", "conditional_draw", "time", "factor"],
            draws_per_parameter=int(draws_per_parameter),
            joint_across_runs="shared_parameter_draw; conditionally_independent_latents",
            covariance_factorization="symmetric_eigendecomposition_roundoff_only",
            max_joint_size=int(max_joint_size),
        )
        results[run] = TrajectorySamples(samples, query, valid, metadata)
    return results
