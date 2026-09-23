"""Conditional Gaussian components using the same covariance as fitting."""

from collections.abc import Mapping

import numpy as np

from ..data import normalize_data, validate_times
from ._backend import runtime
from .results import GaussianMixtureSeries


def donor_data(data, excluded, layout):
    """Remove targets using an explicit layout, never target payload evidence.

    In particular, new run IDs may equal excluded modality names. Inferring
    layout from those entries would require opening an excluded payload.
    """
    if layout not in ("runs", "shorthand"):
        raise ValueError("donor_layout must be runs or shorthand")
    if not isinstance(data, Mapping) or not data:
        raise ValueError("donors must be a nonempty subject mapping")
    selected = {}
    for subject, entries in data.items():
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError("donor subject entries must be nonempty mappings")
        runs = {"run-01": entries} if layout == "shorthand" else entries
        for run, modalities in runs.items():
            if not isinstance(modalities, Mapping):
                raise ValueError("donor_layout='runs' requires subject/run/modality mappings")
            kept = {m: modalities[m] for m in modalities if (subject, m) not in excluded}
            if kept:
                selected.setdefault(subject, {})[run] = kept
    return normalize_data(selected) if selected else {}


def selected_draws(model, max_draws, *, return_indices=False):
    """Balance selected draws across chains, optionally preserving source IDs."""
    xs = model.parameter_draws_
    chains, draws, dimension = xs.shape
    if max_draws is None:
        idx = np.arange(draws)
    elif (
        isinstance(max_draws, bool)
        or not isinstance(max_draws, (int, np.integer))
        or max_draws < chains
    ):
        raise ValueError("max_draws must be None or an integer at least the number of chains")
    else:
        count = min(draws, max_draws // chains)
        idx = np.linspace(0, draws - 1, count, dtype=int)
    selected = xs.reshape(-1, dimension) if max_draws is None else xs[:, idx].reshape(-1, dimension)
    if return_indices:
        return selected, [(c, int(d)) for c in range(chains) for d in idx]
    return selected


def project(problem, draws, run, times, *, key, include_noise, latent_loading=None):
    """Return draw/time means and variances for one feature, or shared latent.

    Both query points and parameter draws are chunked to bound dense temporary
    arrays. Only roundoff-sized negative conditional variances may be clipped.
    ``latent_loading`` accepts one direction or a draw-aligned array; each
    direction is conditioned with its raw draw, including cross-factor terms.
    """
    jax, jnp, jsp, _ = runtime()
    system = problem.systems[run]
    ki, _, mi = problem._packed[run]
    if key is not None and key not in problem.keys:
        raise ValueError("query mapping is unavailable")
    if latent_loading is not None:
        latent_loading = np.asarray(latent_loading, float)
        if (
            key is not None
            or problem.features == 1
            or latent_loading.shape not in ((problem.features,), (len(draws), problem.features))
            or not np.isfinite(latent_loading).all()
        ):
            raise ValueError(
                "latent_loading must be a finite vector or draw/vector array for a multiple-factor latent query"
            )

    query_modality = -1 if key is None else problem.modalities.index(key[1])
    if problem.linear_algebra == "state_space":
        from .state_space import project as state_project

        return state_project(
            problem,
            draws,
            run,
            times,
            key=key,
            include_noise=include_noise,
            latent_loading=latent_loading,
        )

    directions = np.broadcast_to(
        np.eye(problem.features)[0] if latent_loading is None else latent_loading,
        (len(draws), problem.features),
    )
    if problem.linear_algebra == "grouped" and not problem.run_baseline_sd:
        from .grouped_prediction import project as grouped_project

        means, variances = grouped_project(
            problem,
            draws,
            run,
            times,
            keys=[key],
            include_noise=include_noise,
            directions=directions[:, None, :] if key is None else None,
        )
        return means[..., 0], variances[..., 0]

    def one(x, query, direction):
        weights, offsets, noise, widths, lags = problem.arrays(x)
        if key is None:
            width, lag, loading, offset, measurement = 0.0, 0.0, 1.0, 0.0, 0.0
            if problem.features > 1:
                loading = direction
        else:
            target = problem.keys.index(key)
            modality = problem.modalities.index(key[1])
            width, lag, loading, offset = (
                widths[modality],
                lags[modality],
                weights[target],
                offsets[target],
            )
            measurement = noise[problem.groups.index(key[:2])] if include_noise else 0.0
        baseline_variance = problem.run_baseline_sd.get(key[1], 0.0) ** 2 if key else 0.0
        baseline_uncertainty = 0.0
        qa, la = jnp.full(query.shape, width), jnp.full(query.shape, lag)
        qm = np.full(query.shape, query_modality, dtype=int)
        if problem.linear_algebra == "spectral":
            from .spectral import factor

            _, coefficient_mean, L = factor(problem, x, run)
            F = loading * problem.spectral_bases[run].features(query, qa, la)
            mean = F @ coefficient_mean + offset
            projected = jsp.linalg.solve_triangular(L, F.T, lower=True)
            variance = jnp.sum(projected**2, axis=0) + measurement
            # A scalar upper scale keeps the existing variance roundoff check
            # compatible with both stationary and finite-domain prior diagonals.
            prior_scale = jnp.max(jnp.sum(F**2, axis=1)) + measurement
            return mean, variance, prior_scale
        if problem.noise_timescales:
            # Noise belongs to the queried measurement process only when noisy
            # observations are requested. Shared/filtered signal excludes it.
            cross = problem.temporal_covariance(x, query, qm, system.times, mi) * (
                loading * weights[ki][None, :]
                if problem.features == 1
                else (weights[ki] @ loading)[None, :]
            )
            if key is not None and include_noise:
                cross = cross + problem.noise_systems[run].cross(
                    query, problem.keys.index(key), noise[problem._packed[run][1]]
                )
            residual = jnp.asarray(system.values) - offsets[ki]
            L = jnp.linalg.cholesky(problem.covariance(x, run))
            alpha = jsp.linalg.cho_solve((L, True), residual)
            inverse_cross = jsp.linalg.cho_solve((L, True), cross.T)
            mean = cross @ alpha + offset
            reduction = jnp.sum(cross.T * inverse_cross, axis=0)
        elif problem.linear_algebra == "grouped" and problem.run_baseline_sd:
            from .baselines import grouped_factor

            _, beta, L, q, qU, precision, lu = grouped_factor(problem, x, run)
            nodes = problem.grouped_systems[run]
            nm = nodes.modalities
            cross = loading * problem.temporal_covariance(x, query, qm, nodes.times, nm)
            baseline_keys = problem.baseline_designs[run][1]
            query_baseline = jnp.zeros((len(query), len(baseline_keys)))
            if key in baseline_keys:
                query_baseline = query_baseline.at[:, baseline_keys.index(key)].set(
                    problem.run_baseline_sd[key[1]]
                )
                baseline_variance = 0.0  # Posterior coefficient variance below.
            delta = query_baseline - cross @ qU
            mean = cross @ q + offset + delta @ beta
            solved = jsp.linalg.lu_solve(lu, cross.T)
            reduction = jnp.sum(cross.T * precision[:, None] * solved, axis=0)
            # Law of total variance avoids subtracting large baseline terms.
            projected_baseline = jsp.linalg.solve_triangular(L, delta.T, lower=True)
            baseline_uncertainty = jnp.sum(projected_baseline**2, axis=0)
        else:
            L = jnp.linalg.cholesky(problem.covariance(x, run))
            residual = jnp.asarray(system.values) - offsets[ki]
            whitened = jsp.linalg.solve_triangular(L, residual, lower=True)
            cross = problem.temporal_covariance(x, query, qm, system.times, mi) * (
                loading * weights[ki][None, :]
                if problem.features == 1
                else (weights[ki] @ loading)[None, :]
            )
            if problem.run_baseline_sd:
                from .baselines import query_cross

                cross = cross + query_cross(problem, run, key)[None, :]
            projected = jsp.linalg.solve_triangular(L, cross.T, lower=True)
            mean = projected.T @ whitened + offset
            reduction = jnp.sum(projected**2, axis=0)
        prior = (
            problem.temporal_covariance(
                x,
                jnp.zeros(1),
                np.array([query_modality]),
                jnp.zeros(1),
                np.array([query_modality]),
            )[0, 0]
            * jnp.sum(loading**2)
            + baseline_variance
        )
        variance = prior - reduction + measurement + baseline_uncertainty
        return mean, variance, prior + measurement

    evaluate = jax.jit(jax.vmap(one, in_axes=(0, None, 0)))
    means = np.empty((len(draws), len(times)))
    variances = np.empty_like(means)
    for start in range(0, len(draws), 16):
        ds = slice(start, start + 16)
        for first in range(0, len(times), 128):
            ts = slice(first, first + 128)
            mean, variance, prior = (
                np.asarray(v)
                for v in evaluate(
                    jnp.asarray(draws[ds]),
                    jnp.asarray(times[ts]),
                    jnp.asarray(directions[ds]),
                )
            )
            if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                raise FloatingPointError("nonfinite posterior prediction")
            if np.any(variance < -1e-9 * np.maximum(1.0, np.abs(prior[:, None]))):
                raise FloatingPointError("materially negative conditional variance")
            means[ds, ts] = mean
            variances[ds, ts] = np.maximum(variance, 0.0)
    return means, variances


def queries(model, times):
    if times is None:
        raise ValueError("supply explicit native query times")
    requested = times if isinstance(times, Mapping) else {r: times for r in model.prediction_runs_}
    if not requested or not set(requested) <= set(model.prediction_runs_):
        raise ValueError("query runs must have conditioning observations")
    return {r: validate_times(t) for r, t in requested.items()}


def result(model, run, times, draws, keys, *, include_noise, draw_indices=None):
    features = model.problem_.features
    rotation = (
        np.asarray(model.configuration_["factor_orientation"]["rotation"])
        if features > 1 and model.configuration_["inference"] == "map"
        else None
    )
    if features > 1 and keys[0] is None and model.configuration_["inference"] == "posterior":
        from .posterior_coordinates import rotations

        rotation = rotations(model.problem_, draws, model._factor_anchor_keys_, draw_indices)
    if model.problem_.linear_algebra == "grouped" and not model.problem_.run_baseline_sd:
        from .grouped_prediction import project as grouped_project

        means, variances = grouped_project(
            model.problem_,
            draws,
            run,
            times,
            keys=keys,
            include_noise=include_noise,
            directions=np.swapaxes(rotation, -1, -2)
            if rotation is not None and keys[0] is None
            else None,
        )
    else:
        components = [
            project(
                model.problem_,
                draws,
                run,
                times,
                key=key,
                include_noise=include_noise,
                latent_loading=rotation[..., i] if key is None and rotation is not None else None,
            )
            for i, key in enumerate(keys)
        ]
        means = np.stack([c[0] for c in components], axis=-1)
        variances = np.stack([c[1] for c in components], axis=-1)
    a, b = model.prediction_runs_[run]
    modality = keys[0][1] if keys[0] is not None else None
    valid = (
        ((times >= a) & (times <= b))
        if modality is None
        else model.adapter_._support(times, modality, (a, b))
    )
    metadata = result_metadata(
        model,
        run,
        times,
        draws,
        keys,
        include_noise=include_noise,
        draw_indices=draw_indices,
    )
    return GaussianMixtureSeries(means, variances, times, valid, metadata)


def result_metadata(model, run, times, draws, keys, *, include_noise, draw_indices=None):
    """Conditioning and provenance shared by marginal and joint latent queries."""
    features = model.problem_.features
    a, b = model.prediction_runs_[run]
    modality = keys[0][1] if keys[0] is not None else None
    approximation = (
        model.problem_.response_quadrature.metadata()
        if model.problem_.response_quadrature is not None
        else None
    )
    if (
        model.problem_.linear_algebra == "state_space"
        and not model.problem_.response_state_space.identity_only
    ):
        approximation = model.problem_.response_state_space.metadata(model.features)
    if run in model.problem_.spectral_bases:
        basis = model.problem_.spectral_bases[run]
        approximation = basis.metadata()
        approximation["query_outside_basis_domain"] = (
            (times < basis.left) | (times > basis.right)
        ).tolist()
    metadata = dict(
        quantity="unfiltered_shared_latent"
        if modality is None
        else ("noisy_observation" if include_noise else "filtered_signal"),
        uncertainty=model.configuration_["uncertainty"],
        reference_convention=model.configuration_["reference_convention"],
        sample_blocks=model.configuration_.get("sample_blocks"),
        fixed_parameters=model.configuration_.get("fixed_parameters", []),
        fixed_parameter_coordinates="raw_physical",
        conditioning_mode=model.configuration_.get("conditioning_mode", "training"),
        parameter_source=model.configuration_.get("parameter_source", "training"),
        parameter_refits=model.configuration_.get("parameter_refits"),
        parameter_conditioning=model.configuration_.get(
            "parameter_source",
            "training_plus_donors" if model.targets_ is not None else "training",
        ),
        target_exclusion=model.targets_,
        include_noise=include_noise,
        sampling_diagnostics_passed=None
        if model.sampling_diagnostics_ is None
        else model.sampling_diagnostics_["passes"],
        calibration_established=False,
        n_parameter_draws=len(draws),
        available_parameter_draws=int(np.prod(model.parameter_draws_.shape[:2])),
        draw_selection="evenly_spaced_with_equal_count_per_chain",
        n_conditioning_observations=len(model.problem_.systems[run].times),
        support_domain=(a, b),
        temporal_covariance_error_bound=model.problem_.covariance_error_bound,
        covariance_approximation=approximation,
    )
    if draw_indices is not None:
        metadata["parameter_draw_indices"] = [list(index) for index in draw_indices]
    if model.configuration_.get("gp_hyperparameters"):
        import copy

        metadata["gp_hyperparameters"] = copy.deepcopy(model.configuration_["gp_hyperparameters"])
    if features > 1:
        import copy

        metadata["factor_orientation"] = copy.deepcopy(model.configuration_["factor_orientation"])
        metadata["moment_scope"] = (
            "marginal_time_factor; joint_covariance_not_returned"
            if modality is None
            else "marginal_time_feature; joint_covariance_not_returned"
        )
    if model.configuration_.get("temporal_noise"):
        import copy

        metadata["temporal_noise"] = copy.deepcopy(model.configuration_["temporal_noise"])
    if model.configuration_.get("run_baseline"):
        import copy

        metadata["run_baseline"] = copy.deepcopy(model.configuration_["run_baseline"])
        if modality in model.problem_.run_baseline_sd and not include_noise:
            metadata["quantity"] = "filtered_signal_with_run_baseline"
    return metadata
