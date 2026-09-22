"""Independent mode search and diagnostic-rich sampling of a common density."""

import time
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from ._backend import runtime
from .blocks import ParameterSubspace
from .diagnostics import diagnostic_summary
from .execution import execution_info
from .problem import _group_priors
from .rank_diagnostics import bfmi as energy_bfmi
from .timing import PhaseTimings


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SearchConfig:
    """MAP budgets; n_jobs runs independent restarts in shared-memory threads.

    The default is serial. Set BLAS/XLA thread settings before starting Python;
    search never changes process-global thread limits. Refinement is serial.
    Optional diagonal conditioning uses staged physical coordinates for full
    multifactor dense/grouped fits with independent noise; the default is none.
    """

    starts: int = 16
    maxiter: int = 1200
    ftol: float = 1e-12
    gtol: float = 1e-6
    physical_gradient_tolerance: float = 1e-3
    refine_maxiter: int = 0
    polish_max_parameters: int = 256
    n_jobs: int = 1
    conditioning: str = "none"

    def __post_init__(self):
        if self.conditioning not in ("none", "diagonal"):
            raise ValueError("conditioning must be none or diagonal")
        _positive_integer(self.starts, "starts")
        _positive_integer(self.n_jobs, "n_jobs")
        _positive_integer(self.maxiter, "maxiter")
        _positive_integer(self.refine_maxiter, "refine_maxiter", minimum=0)
        _positive_integer(self.polish_max_parameters, "polish_max_parameters", minimum=0)
        if (
            not np.isfinite([self.ftol, self.gtol, self.physical_gradient_tolerance]).all()
            or min(self.ftol, self.gtol, self.physical_gradient_tolerance) <= 0
        ):
            raise ValueError("search tolerances must be positive finite")


@dataclass(frozen=True)
class SamplerConfig:
    """NUTS settings; optional Haar refresh supports full multifactor targets.

    ``orientation_refresh='haar'`` refreshes common rotations and reflections
    after retained transitions. Warmup and the default ``'none'`` are unchanged.
    Supports dense/grouped Identity/Gaussian or quadrature Gamma/DoubleGamma/BachSCR
    targets with all parameters active and isotropic unbounded Gaussian
    loading priors.
    """

    chains: int = 4
    warmup: int = 1000
    draws: int = 1200
    target_accept: float = 0.99
    max_tree_depth: int = 10
    chain_method: str = "sequential"
    start_objective_window: float = 5.0
    start_jitter: float = 0.08
    mass_matrix: str = "dense"
    max_dense_parameters: int = 1024
    orientation_refresh: str = "none"

    def __post_init__(self):
        if not isinstance(self.orientation_refresh, str) or self.orientation_refresh not in (
            "none",
            "haar",
        ):
            raise ValueError("orientation_refresh must be none or haar")
        for name in ("chains", "warmup", "draws", "max_tree_depth"):
            _positive_integer(getattr(self, name), name, 4 if name == "draws" else 1)
        if not np.isfinite(self.target_accept) or not 0 < self.target_accept < 1:
            raise ValueError("target_accept must lie strictly between zero and one")
        if self.chain_method not in ("sequential", "parallel", "vectorized"):
            raise ValueError("unsupported chain_method")
        if self.mass_matrix not in ("dense", "diagonal"):
            raise ValueError("mass_matrix must be dense or diagonal")
        _positive_integer(self.max_dense_parameters, "max_dense_parameters")
        if (
            not np.isfinite([self.start_objective_window, self.start_jitter]).all()
            or min(self.start_objective_window, self.start_jitter) < 0
        ):
            raise ValueError("start window and jitter must be nonnegative finite")


def _feature_moments(problem):
    """Two-pass means/variances with one observation grouping, including masks."""
    n = len(problem.keys)
    lookup = {key: i for i, key in enumerate(problem.keys)}
    indices, observations = [], []
    for run, system in problem.systems.items():
        if hasattr(problem, "_packed"):
            ki = problem._packed[run][0]
        else:
            # Conditional calibration contains other participants' observations.
            ki = np.fromiter((lookup.get(k, -1) for k in system.keys), dtype=int)
        keep = ki >= 0
        indices.append(ki[keep])
        observations.append(system.values[keep])
    indices, values = np.concatenate(indices), np.concatenate(observations)
    counts = np.bincount(indices, minlength=n)
    means = np.divide(
        np.bincount(indices, weights=values, minlength=n),
        counts,
        out=np.full(n, np.nan),
        where=counts > 0,
    )
    variance = np.divide(
        np.bincount(indices, weights=(values - means[indices]) ** 2, minlength=n),
        counts,
        out=np.full(n, np.nan),
        where=counts > 0,
    )
    return means, variance


def initial_points(problem, count, seed):
    """Prior-quantile coordinates plus one data-based start.

    Preserve scrambled Sobol designs through SciPy's maximum dimension. Above
    that limit use seeded IID uniform prior quantiles, not a global Sobol design.
    """
    dimension = len(problem.names)
    if dimension <= qmc.Sobol.MAXDIM:
        u = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(
            int(np.ceil(np.log2(count)))
        )[:count]
    else:
        u = np.random.default_rng(seed).uniform(size=(count, dimension))
    prior_groups = _group_priors(problem.parameter_priors)
    points = np.empty_like(u)
    for indices, prior in prior_groups:
        points[:, indices] = prior.ppf(np.clip(u[:, indices], 1e-5, 1 - 1e-5))
    base = problem.initial.copy()
    directions = None
    if problem.features > 1:
        directions = np.random.default_rng(seed).normal(size=(len(problem.keys), problem.features))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    means, variances = _feature_moments(problem)
    for i, key in enumerate(problem.keys):
        mean, variance = means[i], variances[i]
        if problem.features == 1:
            base[i] = np.sqrt(max(0.01, variance))
        else:
            begin = i * problem.features
            base[begin : begin + problem.features] = np.sqrt(max(0.01, variance)) * directions[i]
        base[problem.indices[("offset", *key)]] = mean
    for indices, prior in prior_groups:
        base[indices] = np.clip(base[indices], prior.ppf(1e-4), prior.ppf(1 - 1e-4))
    points[0] = base
    return points


def _map_diagnostics(problem, x, config, result):
    value, gradient = problem.value_gradient(x)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite objective or gradient")
    projected = gradient.copy()
    boundary = []
    for j, (lo, hi) in enumerate(problem.bounds):
        at_lo = x[j] - lo <= 1e-7 * max(1.0, abs(x[j]))
        at_hi = hi - x[j] <= 1e-7 * max(1.0, abs(x[j]))
        if at_lo or at_hi:
            boundary.append(problem.names[j])
        if (at_lo and gradient[j] > 0) or (at_hi and gradient[j] < 0):
            projected[j] = 0.0
    norm = float(np.max(np.abs(projected)))
    return dict(
        parameters=x.tolist(),
        objective=value,
        optimizer_success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
        message=str(result.message),
        physical_projected_gradient=norm,
        boundary_parameters=boundary,
        meets_gradient_tolerance=norm <= config.physical_gradient_tolerance,
    )


class _CheckpointFailure(Exception):
    """Keep callback errors distinct from numerical optimizer failures."""

    def __init__(self, error):
        self.error = error
        super().__init__(str(error))


def _iteration_checkpoint(progress, physical, start, phase):
    """Snapshot accepted iterates without another objective/gradient evaluation."""
    if progress is None:
        return None
    iteration = 0
    started = time.perf_counter()

    def callback(intermediate_result):
        nonlocal iteration
        iteration += 1
        if iteration != 1 and iteration % 25:
            return
        event = dict(
            phase=phase,
            record=dict(
                start=start,
                iteration=iteration,
                objective=float(intermediate_result.fun),
                parameters=np.asarray(physical(intermediate_result.x)).tolist(),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        try:
            progress(event)
        except Exception as exc:
            raise _CheckpointFailure(exc) from exc

    return callback


def _refine(problem, record, config, evaluate, *, progress=None):
    """Optional stricter refinement of the selected physical-density MAP.

    Keep the original search record and every attempted refinement. Only a
    finite, nonworsening physical objective can replace the selected solution.
    The optional polisher can also verify a strictly negative exact variance
    difference when absolute objective totals lose that difference to roundoff;
    it records both changes and keeps the actual recomputed objective value.
    Reuse original transforms for open-support parameters. Multi-factor finite
    filter boxes use physical coordinates to avoid logistic saturation. No
    Jacobian enters either MAP objective.
    """
    started = time.perf_counter()
    record["pre_refinement"] = dict(record)
    ftol = min(config.ftol, 1e-15)
    gtol = min(config.gtol, 1e-9)
    refinement = dict(
        accepted=False,
        coordinates="unconstrained_physical_density",
        maxiter=config.refine_maxiter,
        ftol=ftol,
        gtol=gtol,
        iterations=None,
    )

    phase = "initialization"
    try:
        from .refinement_coordinates import filter_coordinates

        if config.conditioning == "diagonal":
            from .map_conditioning import minimize_conditioned

            phase = "optimization"
            result, report = minimize_conditioned(
                problem,
                record["parameters"],
                replace(config, ftol=ftol, gtol=gtol),
                maxiter=config.refine_maxiter,
                callback=_iteration_checkpoint(
                    progress, lambda x: x, record.get("start"), "refinement_iteration"
                ),
            )

            def physical(x):
                return x

            refinement.update(
                coordinates="affine_physical",
                conditioning=report,
                function_evaluations=int(result.nfev),
            )
        else:
            mixed = filter_coordinates(problem, record["parameters"])
            if mixed is None:
                initial = problem.to_unconstrained(record["parameters"])

                def physical(z):
                    return problem.from_unconstrained(z)[0]

                optimizer_bounds = None
            else:
                initial, physical, evaluate, optimizer_bounds, indices = mixed
                refinement.update(
                    coordinates="physical_filter_boxes_other_unconstrained",
                    bounded_filter_indices=indices,
                    bounded_filter_bounds=[problem.bounds[j] for j in indices],
                )
            phase = "optimization"
            result = minimize(
                evaluate,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=optimizer_bounds,
                callback=_iteration_checkpoint(
                    progress, physical, record.get("start"), "refinement_iteration"
                ),
                options=dict(
                    maxiter=config.refine_maxiter, ftol=ftol, gtol=gtol, maxcor=50, maxls=40
                ),
            )
        # Account for work even when evaluating its returned point fails.
        refinement["iterations"] = int(result.nit)
        x = np.asarray(physical(result.x))
        candidate = _map_diagnostics(problem, x, config, result)
        refinement.update(candidate)
        if candidate["objective"] <= record["objective"]:
            record.update(candidate)
            refinement["accepted"] = True
    except _CheckpointFailure as exc:
        raise exc.error from exc
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        refinement["failure"] = f"{type(exc).__name__}: {exc}"
        refinement["failure_phase"] = phase
        # Conversion cannot consume optimizer iterations. A rounded boundary
        # point can still have finite density and valid physical derivatives.
        # Optimizer exceptions retain unknown work and cannot reuse the budget.
        if phase == "initialization":
            refinement["iterations"] = 0
    # Retain the completed L-BFGS estimate before any optional polishing work.
    if progress is not None:
        progress(dict(phase="refinement_optimized", record=deepcopy(record)))
    # Relative objective changes can round to zero before physical stationarity.
    # A few local curvature steps use the remaining explicit iteration budget.
    used = refinement["iterations"]
    remaining = 0 if used is None else min(3, config.refine_maxiter - used)
    if not record["meets_gradient_tolerance"] and remaining > 0:
        from .polishing import polish

        refinement["before_polish"] = dict(refinement)
        polishing = polish(problem, record, config, remaining)
        refinement["polish"] = polishing
        if polishing["accepted_steps"]:
            refinement["accepted"] = True
            for key in (
                "parameters",
                "objective",
                "physical_projected_gradient",
                "meets_gradient_tolerance",
                "optimizer_success",
                "status",
                "message",
                "boundary_parameters",
            ):
                refinement[key] = record[key]
        refinement["iterations"] = used + polishing["steps"]
        record["iterations"] = refinement["iterations"]
    refinement["elapsed_seconds"] = time.perf_counter() - started
    record["refinement"] = refinement
    record["elapsed_seconds"] += refinement["elapsed_seconds"]


def search(problem, config, seed, *, progress=None):
    """Optimize physical posterior density in unconstrained coordinates.

    No transform Jacobian enters MAP. Physical projected gradients are checked
    separately to detect artificial stationarity near a saturated transform.
    An optional callback receives independent records before/after each restart
    and refinement, plus the first and every 25th accepted iterate. Callback
    failures propagate so failed checkpoints cannot silently masquerade as
    successfully retained evidence. Parallel restarts share immutable prepared
    observations and compiled functions. Progress always runs on the caller's
    thread; returned records retain start order, regardless of completion order.
    """
    if config.conditioning == "diagonal":
        from .map_conditioning import validate

        validate(problem)
    jax, _, _, _ = runtime()
    vg = jax.jit(jax.value_and_grad(lambda z: problem.objective(problem.from_unconstrained(z)[0])))

    def evaluate(z):
        value, gradient = vg(z)
        return float(value), np.asarray(gradient)

    def run_start(index, x0, emit, check_cancelled):
        def checkpoint(phase, record):
            if emit is not None:
                emit(dict(phase=phase, record=deepcopy(record)))

        def evaluate_start(z):
            check_cancelled()
            return evaluate(z)

        check_cancelled()
        started = time.perf_counter()
        record = dict(start=index, initial_parameters=x0.tolist())
        checkpoint("started", record)
        try:
            if config.conditioning == "diagonal":
                from .map_conditioning import minimize_conditioned

                result, report = minimize_conditioned(
                    problem,
                    x0,
                    config,
                    maxiter=config.maxiter,
                    callback=_iteration_checkpoint(emit, lambda x: x, index, "iteration"),
                    check_cancelled=check_cancelled,
                )
                record["conditioning"] = report
                x = result.x
            else:
                result = minimize(
                    evaluate_start,
                    problem.to_unconstrained(x0),
                    jac=True,
                    method="L-BFGS-B",
                    callback=_iteration_checkpoint(
                        emit, lambda z: problem.from_unconstrained(z)[0], index, "iteration"
                    ),
                    options=dict(
                        maxiter=config.maxiter,
                        ftol=config.ftol,
                        gtol=config.gtol,
                        maxcor=50,
                        maxls=40,
                    ),
                )
                x = np.asarray(problem.from_unconstrained(result.x)[0])
            check_cancelled()
            record.update(_map_diagnostics(problem, x, config, result))
            record["function_evaluations"] = int(result.nfev)
        except _CheckpointFailure as exc:
            raise exc.error from exc
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            record.update(
                objective=None,
                optimizer_success=False,
                failure=f"{type(exc).__name__}: {exc}",
            )
        record["elapsed_seconds"] = time.perf_counter() - started
        checkpoint("finished", record)
        return record

    points = initial_points(problem, config.starts, seed)
    workers = min(config.n_jobs, len(points))
    if workers > 1:
        # Trace/compile once before dispatch; no traced values cross threads.
        if config.conditioning == "diagonal":
            problem.value_gradient(points[0])
        else:
            vg.lower(problem.to_unconstrained(points[0])).compile()
    from .map_parallel import ordered_restarts

    records = ordered_restarts(run_start, points, workers, progress)
    finite = [r for r in records if r["objective"] is not None]
    if not finite:
        return None, records
    best = min(finite, key=lambda r: (r["objective"], r["start"]))
    if config.refine_maxiter and not best["meets_gradient_tolerance"]:
        if progress is not None:
            progress(dict(phase="refining", record=deepcopy(best)))
        _refine(problem, best, config, evaluate, progress=progress)
        if progress is not None:
            progress(dict(phase="refined", record=deepcopy(best)))
    return best, records


def log_likelihood_draws(problem, draws):
    """Evaluate at most 16 likelihoods at once, preserving chain/draw axes."""
    jax, jnp, _, _ = runtime()
    draws = np.asarray(draws)
    if draws.ndim != 3 or 0 in draws.shape or draws.shape[-1] != len(problem.names):
        raise ValueError("draws must have nonempty chain/draw/parameter axes")
    evaluate = jax.jit(jax.vmap(problem.nll))
    flat = draws.reshape(-1, draws.shape[-1])
    output = np.empty(len(flat))
    for first in range(0, len(flat), 16):
        # Materialize before dispatching another chunk so device buffers cannot
        # accumulate asynchronously across the entire posterior.
        output[first : first + 16] = -np.asarray(evaluate(jnp.asarray(flat[first : first + 16])))
    return output.reshape(draws.shape[:2])


def _metric_metadata(inverse_mass_matrix, adapted_position, active_dimension, requested):
    """Describe the adapted NumPyro metric from its actual array layout."""
    jax, _, _, _ = runtime()
    leaves = jax.tree_util.tree_leaves(inverse_mass_matrix)
    if len(leaves) != 1:
        raise RuntimeError("sampling produced an unsupported structured mass matrix")
    leaf = leaves[0]
    shape = tuple(leaf.shape)
    position_shape = tuple(adapted_position.shape)
    if shape == (*position_shape, active_dimension):
        actual = "dense"
        entries = active_dimension**2
    elif shape == position_shape:
        actual = "diagonal"
        entries = active_dimension
    else:
        raise RuntimeError("adapted inverse mass matrix shape does not match active parameters")
    if actual != requested:
        raise RuntimeError(
            f"requested {requested} mass matrix but NumPyro adapted {actual} geometry"
        )
    itemsize = np.dtype(leaf.dtype).itemsize
    return dict(
        requested=requested,
        actual=actual,
        active_dimension=active_dimension,
        estimated_bytes_per_chain=int(entries * itemsize),
        adapted_inverse_mass_matrix_shape=list(shape),
        adapted_inverse_mass_matrix_dtype=str(np.dtype(leaf.dtype)),
        adapted_values_stored=False,
    )


def _adaptation_diagnostics(adapted_state, adapted_position, active_dimension, requested):
    """Retain actual adapted values only for bounded, reviewable geometries."""
    jax, _, _, _ = runtime()
    metric = _metric_metadata(
        adapted_state.inverse_mass_matrix,
        adapted_position,
        active_dimension,
        requested,
    )
    inverse_mass_matrix = jax.tree_util.tree_leaves(adapted_state.inverse_mass_matrix)[0]
    step_size = adapted_state.step_size
    adaptation = dict(
        values_stored=active_dimension <= 64,
        step_size_shape=list(step_size.shape),
        step_size_dtype=str(step_size.dtype),
        inverse_mass_matrix_shape=list(inverse_mass_matrix.shape),
        inverse_mass_matrix_dtype=str(inverse_mass_matrix.dtype),
    )
    if adaptation["values_stored"]:
        adaptation.update(
            step_size=np.asarray(step_size).tolist(),
            inverse_mass_matrix=np.asarray(inverse_mass_matrix).tolist(),
        )
    else:
        adaptation["omission_reason"] = "active_dimension_exceeds_64"
    metric["adapted_values_stored"] = adaptation["values_stored"]
    return metric, adaptation


def _validate_initialization_seed(seed):
    if seed is None:
        return
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("initialization_seed must be an integer from 0 to 2**32 - 1")


def sample(
    problem,
    records,
    config,
    seed,
    *,
    blocks=None,
    progress=None,
    initialization_seed=None,
    factor_anchors=None,
    warmup_checkpoint=None,
):
    if hasattr(problem, "_posterior_update"):
        from .posterior_updates import validate_update_target

        validate_update_target(problem)
    timing = PhaseTimings(progress)
    with timing.phase("initialization"):
        _validate_initialization_seed(initialization_seed)
        execution_info(config)
        jax, jnp, _, _ = runtime()
        try:
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:
            raise ImportError("posterior diagnostics require multimodalsrm[bayesian]") from exc
        ordered = sorted(
            [r for r in records if r["objective"] is not None],
            key=lambda r: r["objective"],
        )
        if not ordered:
            raise ValueError("sampling requires a finite MAP reference")
        space = ParameterSubspace(problem, ordered[0]["parameters"], blocks=blocks)
        orientation_context = _orientation_context(space, factor_anchors)
        active_dimension = len(space.active_indices)
        if config.mass_matrix == "dense" and active_dimension > config.max_dense_parameters:
            raise ValueError(
                f"dense mass matrix active dimension {active_dimension} exceeds "
                f"max_dense_parameters={config.max_dense_parameters}"
            )
        conditional = []
        for record in ordered:
            x = np.array(record["parameters"], copy=True)
            x[list(space.fixed_indices)] = space.reference[list(space.fixed_indices)]
            conditional.append(
                dict(record, parameters=x.tolist(), objective=float(problem.objective(x)))
            )
        ordered = sorted(conditional, key=lambda r: r["objective"])
        eligible = [
            r
            for r in ordered
            if r["objective"] <= ordered[0]["objective"] + config.start_objective_window
        ]
        chosen = [eligible[i % len(eligible)] for i in range(config.chains)]
        physical_starts, adjustments = [], []
        for chain, record in enumerate(chosen):
            x = np.array(record["parameters"], copy=True)
            for j in space.active_indices:
                prior = problem.parameter_priors[j]
                before = x[j]
                if x[j] <= prior.lower:
                    x[j] = prior.ppf(1e-6)
                elif x[j] >= prior.upper:
                    x[j] = prior.ppf(1 - 1e-6)
                if x[j] != before:
                    adjustments.append(
                        dict(
                            chain=chain,
                            parameter=list(problem.names[j]),
                            map_value=float(before),
                            start_value=float(x[j]),
                        )
                    )
            physical_starts.append(x)
        starts = np.stack([space.to_unconstrained(x) for x in physical_starts])
        jitter_seed = seed if initialization_seed is None else initialization_seed
        starts += np.random.default_rng(jitter_seed).normal(0.0, config.start_jitter, starts.shape)
        kernel = NUTS(
            potential_fn=space.potential,
            dense_mass=config.mass_matrix == "dense",
            target_accept_prob=config.target_accept,
            max_tree_depth=config.max_tree_depth,
        )
        if config.orientation_refresh == "haar":
            from .orthogonal import orthogonal_kernel

            kernel = orthogonal_kernel(kernel, space)
        mc = MCMC(
            kernel,
            num_warmup=config.warmup,
            num_samples=config.draws,
            num_chains=config.chains,
            chain_method=config.chain_method,
            progress_bar=False,
        )
        execution = execution_info(config, effective_method=mc.chain_method)
    started = time.perf_counter()
    fields = ("energy", "potential_energy", "accept_prob", "num_steps")
    init = jnp.asarray(starts if config.chains > 1 else starts[0])
    if warmup_checkpoint is not None and warmup_checkpoint.resumed:
        with timing.phase("checkpoint_restore"):
            mc.post_warmup_state = warmup_checkpoint.restore(space, config)
        with timing.phase("sampling"):
            # A fresh NumPyro kernel initializes its compiled functions from z;
            # the restored HMC state and advanced keys supply the actual chain.
            mc.run(
                mc.post_warmup_state.rng_key,
                init_params=mc.post_warmup_state.z,
                extra_fields=fields,
            )
            jax.block_until_ready(mc.last_state)
    elif progress is None and warmup_checkpoint is None:
        with timing.phase("warmup_and_sampling"):
            mc.run(jax.random.PRNGKey(seed), init_params=init, extra_fields=fields)
            jax.block_until_ready(mc.last_state)
    else:
        with timing.phase("warmup"):
            mc.warmup(jax.random.PRNGKey(seed), init_params=init, extra_fields=fields)
            jax.block_until_ready(mc.post_warmup_state)
        if warmup_checkpoint is not None:
            with timing.phase("checkpoint_save"):
                warmup_checkpoint.save(mc.post_warmup_state, space, timing.seconds["warmup"])
        with timing.phase("sampling"):
            # Continue each chain's advanced key, not the original seed. This
            # preserves the trajectory of one uninterrupted MCMC.run call.
            mc.run(mc.post_warmup_state.rng_key, extra_fields=fields)
            jax.block_until_ready(mc.last_state)
    with timing.phase("physical_draws"):
        z = mc.get_samples(group_by_chain=True)
        xs = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda a: space.from_unconstrained(a)[0])))(z))
        extra = {k: np.asarray(v) for k, v in mc.get_extra_fields(group_by_chain=True).items()}
        metric, adaptation = _adaptation_diagnostics(
            mc.last_state.adapt_state,
            mc.last_state.z,
            active_dimension,
            config.mass_matrix,
        )
    with timing.phase("log_likelihood"):
        loglik = log_likelihood_draws(problem, xs)
    with timing.phase("diagnostics"):
        quantities = np.concatenate(
            [xs[..., list(space.active_indices)], loglik[..., None]], axis=-1
        )
        summary = diagnostic_summary(quantities)
        bfmi = energy_bfmi(extra["energy"])
    diagnostic = dict(
        execution=execution,
        metric=metric,
        adaptation=adaptation,
        phase_seconds=dict(timing.seconds),
        active_parameters=space.active_parameters,
        fixed_parameters=space.fixed_parameters,
        reference_parameters=space.reference.tolist(),
        chains=config.chains,
        draws=config.draws,
        divergences=int(extra["diverging"].sum()),
        max_rank_rhat=float(summary.r_hat.max()),
        min_bulk_ess=float(summary.ess_bulk.min()),
        min_tail_ess=float(summary.ess_tail.min()),
        min_bfmi=float(bfmi.min()),
        bfmi=bfmi.tolist(),
        tree_depth_saturations=int((extra["num_steps"] >= 2**config.max_tree_depth - 1).sum()),
        mean_accept=float(extra["accept_prob"].mean()),
        initialization_seed=int(jitter_seed),
        sampler_seed=int(seed),
        start_indices=[r["start"] for r in chosen],
        initialization_adjustments=adjustments,
        initial_unconstrained=starts.tolist(),
        parameters=[
            dict(
                name=list(name),
                **{k: row[k] for k in ("r_hat", "ess_bulk", "ess_tail")},
            )
            for name, row in zip([*space.active_parameters, ("log_likelihood",)], summary.rows())
        ],
        elapsed_seconds=time.perf_counter() - started,
        calibration_established=False,
        limits=dict(
            rank_rhat=1.01,
            bulk_ess=400,
            tail_ess=200,
            bfmi=0.2,
            divergences=0,
            depth_saturations=0,
        ),
    )
    if warmup_checkpoint is not None:
        diagnostic["warmup_checkpoint"] = warmup_checkpoint.metadata()
    diagnostic["passes"] = bool(
        config.chains >= 2
        and np.isfinite(summary.columns(("r_hat", "ess_bulk", "ess_tail"))).all()
        and np.isfinite(bfmi).all()
        and diagnostic["divergences"] == 0
        and diagnostic["max_rank_rhat"] <= 1.01
        and diagnostic["min_bulk_ess"] >= 400
        and diagnostic["min_tail_ess"] >= 200
        and diagnostic["min_bfmi"] >= 0.2
        and diagnostic["tree_depth_saturations"] == 0
    )
    diagnostic["raw_passes"] = diagnostic["passes"]
    diagnostic["orientation_refresh"] = config.orientation_refresh
    if orientation_context["applicable"]:
        from .orientation_diagnostics import orientation_diagnostics

        count = len(problem.keys) * problem.features
        orientation = orientation_diagnostics(
            xs[..., :count].reshape(*xs.shape[:2], len(problem.keys), problem.features),
            anchor_indices=orientation_context["anchor_indices"],
        )
        orientation.update(orientation_context)
        diagnostic["passes"] = bool(diagnostic["raw_passes"] and orientation["passes"])
    else:
        orientation = orientation_context
    if problem.features > 2:
        from .orientation_diagnostics import GENERAL_K_PROBE_SCHEMA

        diagnostic["orientation_probe_schema"] = GENERAL_K_PROBE_SCHEMA
    diagnostic["orientation"] = orientation
    return xs, extra, loglik, diagnostic


def _orientation_context(space, factor_anchors):
    """Assess only the full isotropic multifactor target used by the move."""
    from .orthogonal import validate_orthogonal_target

    try:
        validate_orthogonal_target(space)
    except ValueError as exc:
        return dict(applicable=False, status="not_assessed", passes=None, reason=str(exc))
    k = space.problem.features
    keys = list(space.problem.keys[:k] if factor_anchors is None else factor_anchors)
    keys = [tuple(k) for k in keys]
    if len(keys) != k or len(set(keys)) != k or any(k not in space.problem.keys for k in keys):
        raise ValueError(f"orientation factor_anchors must identify {k} distinct loading rows")
    return dict(
        applicable=True,
        anchor_indices=[space.problem.keys.index(k) for k in keys],
        anchor_keys=[list(k) for k in keys],
        anchor_source=(
            ("first_two_loading_rows" if k == 2 else "first_k_loading_rows")
            if factor_anchors is None
            else "supplied_factor_anchors"
        ),
    )
