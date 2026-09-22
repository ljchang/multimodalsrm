"""Exact response node updates with positive observation noise.

Group only identical (modality, native timestamp) observations. The likelihood
keeps every residual and normalization term. Information matrices may be rank
deficient: only positive predicted functional covariances and I + L.T D L
are factored, never the loading information D itself.
"""

import numpy as np

from ._backend import runtime


def eligible(problem):
    """Group fixed or learned responses when the noise prior excludes zero."""
    prior = problem.priors.noise
    return prior.family == "lognormal" or prior.lower > 0


def event_system(problem, x, run, times=None, query_modality=-1):
    """Realize responses and sort shifted nodes, optionally including queries.

    Node membership uses native modality/time pairs and never depends on x.
    Learned lags can reorder nodes or tie different modalities without merging
    them. Transition derivatives at zero elapsed time remain those of the
    existing scalar backend.
    """
    from .state_space_delays import event_system as delay_events
    from .state_space_parameters import parameter_event_system

    _, jnp, _, _ = runtime()
    model = problem.response_state_space
    nodes = problem.grouped_systems[run]
    if problem.parameterized_responses:
        state = model.realize(x, problem.indices)
        outputs, lags = state[2:]
    else:
        outputs = jnp.asarray(model.outputs)
        lags = jnp.concatenate((problem.arrays(x)[4], jnp.zeros(1)))
    shifted = jnp.asarray(nodes.times) - lags[nodes.modalities]
    if times is not None:
        shifted = jnp.concatenate((shifted, jnp.asarray(times) - lags[query_modality]))
    if problem.parameterized_responses:
        events = parameter_event_system(model, state, shifted, problem.features)
    else:
        events = delay_events(shifted, problem.delay_transition, problem.features)
    return events, outputs


def statistics(problem, x, run):
    """Recompute parameter-dependent sufficient statistics, excluding masked data."""
    from .multifactor import block_statistics

    _, jnp, _, _ = runtime()
    nodes = problem.grouped_systems[run]
    ki, gi, _ = problem._packed[run]
    weights, offsets, noise, _, _ = problem.arrays(x)
    weights = weights.reshape(len(problem.keys), problem.features)[ki]
    variance = noise[gi]
    residual = jnp.asarray(problem.systems[run].values) - offsets[ki]
    precision, information = block_statistics(
        weights, residual, variance, nodes.observation_nodes, len(nodes.times)
    )
    return precision, information.reshape(-1, problem.features), weights, residual, variance


def update(mean, covariance, A, Q, response, precision, information):
    """Assimilate one rank-K functional with an information-form Joseph update.

    Whiten the predicted functional covariance V = L L.T, so the small solve
    I + L.T D L is symmetric positive definite even for zero/rank-deficient
    loadings. The Joseph covariance avoids subtracting nearly equal matrices.
    Return the log determinant and prior displacement cost; observation
    residuals are scored separately without cancellation of large quadratics.
    """
    _, jnp, jsp, _ = runtime()
    factors = precision.shape[0]
    C = jnp.kron(jnp.eye(factors), response[None, :])
    pm = A @ mean
    pc = A @ covariance @ A.T + Q
    pc = (pc + pc.T) * 0.5
    cross = pc @ C.T
    V = C @ cross
    L = jnp.linalg.cholesky((V + V.T) * 0.5)
    B = jnp.eye(factors) + L.T @ precision @ L
    chol = jnp.linalg.cholesky((B + B.T) * 0.5)
    white_cross = jsp.linalg.solve_triangular(L, cross.T, lower=True).T
    alpha = jsp.linalg.cho_solve((chol, True), L.T @ (information - precision @ (C @ pm)))
    mean = pm + white_cross @ alpha
    gain = white_cross @ jsp.linalg.cho_solve((chol, True), L.T)
    residual_map = jnp.eye(len(mean)) - gain @ precision @ C
    covariance = residual_map @ pc @ residual_map.T + gain @ precision @ gain.T
    covariance = (covariance + covariance.T) * 0.5
    score = 2 * jnp.log(jnp.diag(chol)).sum() + alpha @ alpha
    return mean, covariance, score, C @ mean, pm, pc


def nll(problem, x, run):
    """Filter at nodes and evaluate every original observation's residual."""
    jax, jnp, _, _ = runtime()
    nodes = problem.grouped_systems[run]
    model = problem.response_state_space
    if problem.dynamic_state_space:
        system, outputs = event_system(problem, x, run)
    else:
        system = problem.state_space_systems[run]
        outputs = jnp.asarray(model.outputs)
    precision, information, weights, residual, variance = statistics(problem, x, run)
    order = system.order
    dimension = model.dimension * problem.features

    def step(carry, inputs):
        mean, covariance, score = carry
        mean, covariance, increment, projected, _, _ = update(mean, covariance, *inputs)
        return (mean, covariance, score + increment), projected

    final, means = jax.lax.scan(
        jax.checkpoint(step, prevent_cse=False),
        (jnp.zeros(dimension), jnp.eye(dimension), jnp.asarray(0.0)),
        (
            jnp.asarray(system.transition),
            jnp.asarray(system.process_covariance),
            outputs[nodes.modalities][order],
            precision[order],
            information[order],
        ),
    )
    # Each node's posterior mean conditions only on that node and earlier
    # observations. Together with its prior displacement cost this is exactly
    # the innovation quadratic, without subtracting large r.T R^-1 r terms.
    means = means[jnp.argsort(order)]
    error = residual - jnp.sum(weights * means[nodes.observation_nodes], axis=1)
    return 0.5 * (final[2] + jnp.sum(error**2 / variance + jnp.log(variance) + np.log(2 * np.pi)))


def smoother(problem, run, times, query_modality):
    """RTS smoothing over observation nodes plus unobserved query events."""
    from .state_space import StateSpaceSystem, _smooth_backward

    jax, jnp, _, _ = runtime()
    model = problem.response_state_space
    nodes = problem.grouped_systems[run]
    count = len(nodes.times)
    if not problem.dynamic_state_space:
        events = StateSpaceSystem.prepare(
            np.concatenate(
                (nodes.times - model.lags[nodes.modalities], times - model.lags[query_modality])
            ),
            model,
            problem.features,
        )
    dimension = model.dimension * problem.features

    def one(x):
        if problem.dynamic_state_space:
            merged, outputs = event_system(problem, x, run, times, query_modality)
        else:
            merged, outputs = events, jnp.asarray(model.outputs)
        order = merged.order
        query_indices = jnp.argsort(order)[count:]
        A, Q = jnp.asarray(merged.transition), jnp.asarray(merged.process_covariance)
        row = outputs[query_modality]
        rows = jnp.concatenate((outputs[nodes.modalities], jnp.tile(row, (len(times), 1))))[order]
        observed = jnp.asarray(order < count)
        precision, information, _, _, _ = statistics(problem, x, run)
        precision = jnp.concatenate(
            (precision, jnp.zeros((len(times), problem.features, problem.features)))
        )[order]
        information = jnp.concatenate((information, jnp.zeros((len(times), problem.features))))[
            order
        ]

        def forward(carry, inputs):
            A, Q, row, D, h, is_observed = inputs

            def assimilate(_):
                mean, covariance, _, _, pm, pc = update(*carry, A, Q, row, D, h)
                return mean, covariance, pm, pc

            def predict(_):
                mean, covariance = carry
                pm = A @ mean
                pc = A @ covariance @ A.T + Q
                pc = (pc + pc.T) * 0.5
                return pm, pc, pm, pc

            result = jax.lax.cond(is_observed, assimilate, predict, None)
            return result[:2], result

        last, history = jax.lax.scan(
            forward,
            (jnp.zeros(dimension), jnp.eye(dimension)),
            (A, Q, rows, precision, information, observed),
        )
        mean, covariance = _smooth_backward(
            last, history, A, semidefinite=False, dynamic=problem.dynamic_state_space
        )
        mean = mean[query_indices].reshape(len(times), problem.features, model.dimension) @ row
        covariance = covariance[query_indices].reshape(
            len(times), problem.features, model.dimension, problem.features, model.dimension
        )
        return mean, jnp.einsum("i,tkilj,j->tkl", row, covariance, row)

    return one
