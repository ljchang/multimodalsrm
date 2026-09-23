"""Research sequential filter using repeated per-factor transition blocks.

Full cross-factor covariances are retained. The optional compact Joseph
contraction is algebraically equivalent but needs independent low-noise
qualification. Neither route changes the response approximation or priors.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from multimodalsrm.bayesian.state_space_delays import event_system as delay_events
from multimodalsrm.bayesian.state_space_grouped import statistics
from multimodalsrm.bayesian.state_space_grouped import update as original_update
from multimodalsrm.bayesian.state_space_parameters import (
    parameter_event_system,
    parameter_transitions,
)


def sym(x):
    return (x + x.T) * 0.5


def predict(mean, covariance, A, Q, factors):
    """Propagate every factor pair with the same single-factor transition."""
    d = A.shape[0]
    pm = (mean.reshape(factors, d) @ A.T).reshape(-1)
    pc = jnp.einsum("ab,kblc,dc->kald", A, covariance.reshape(factors, d, factors, d), A)
    pc = pc.reshape(factors * d, factors * d) + jnp.kron(jnp.eye(factors), Q)
    return pm, sym(pc)


def update(mean, covariance, A, Q, response, precision, information, *, joseph="dense"):
    """Block prediction and unchanged small SPD information solve."""
    factors, d = precision.shape[0], len(response)
    if joseph == "original":
        return original_update(
            mean,
            covariance,
            jnp.kron(jnp.eye(factors), A),
            jnp.kron(jnp.eye(factors), Q),
            response,
            precision,
            information,
        )
    pm, pc = predict(mean, covariance, A, Q, factors)
    dimension = len(mean)
    cross = pc.reshape(dimension, factors, d) @ response
    V = jnp.einsum("a,kal->kl", response, cross.reshape(factors, d, factors))
    L = jnp.linalg.cholesky(sym(V))
    B = jnp.eye(factors) + L.T @ precision @ L
    chol = jnp.linalg.cholesky(sym(B))
    white_cross = jsp.linalg.solve_triangular(L, cross.T, lower=True).T
    alpha = jsp.linalg.cho_solve(
        (chol, True), L.T @ (information - precision @ (pm.reshape(factors, d) @ response))
    )
    mean = pm + white_cross @ alpha
    gain = white_cross @ jsp.linalg.cho_solve((chol, True), L.T)
    gp = gain @ precision
    if joseph == "dense":
        residual = jnp.eye(dimension) - (gp[:, :, None] * response).reshape(dimension, dimension)
        covariance = residual @ pc @ residual.T + gp @ gain.T
    elif joseph == "compact":
        # (I-GDC) P (I-GDC).T + GDG.T, associated in low-rank products.
        # This saves cubic work but can change cancellation in floating point.
        left = pc - gp @ cross.T
        covariance = left - (left.reshape(dimension, factors, d) @ response) @ gp.T + gp @ gain.T
    else:
        raise ValueError("joseph must be original, dense, or compact")
    score = 2 * jnp.log(jnp.diag(chol)).sum() + alpha @ alpha
    return mean, sym(covariance), score, mean.reshape(factors, d) @ response, pm, pc


def interval_groups(times, modalities, order, shifted, capacity):
    """Group equal elapsed values only when their lag derivatives also agree.

    The signature retains both endpoint modalities and the native time
    difference. Elapsed time is an additional key to preserve exact floating
    point values, even if equivalent time differences round differently.
    The stationary first event has its own constant-zero signature.
    """
    t, m = times[order], modalities[order]
    before = jnp.concatenate((jnp.array([-1]), m[:-1]))
    after = m.at[0].set(-1)
    native_delta = jnp.diff(t, prepend=t[0])
    delta = jnp.diff(shifted, prepend=shifted[0])
    keys = (before, after, native_delta, delta, jnp.arange(len(t)))
    a, b, c, d, indices = jax.lax.sort(keys, num_keys=4, is_stable=True)
    starts = jnp.concatenate(
        (
            jnp.array([True]),
            (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (c[1:] != c[:-1]) | (d[1:] != d[:-1]),
        )
    )
    labels = jnp.cumsum(starts) - 1
    inverse = jnp.zeros(len(t), dtype=labels.dtype).at[indices].set(labels)
    first = jnp.nonzero(starts, size=capacity, fill_value=0)[0]
    return indices[first], inverse, jnp.sum(starts)


def events(problem, x, run, *, transition_cache=0):
    """Store transitions once per time, rather than repeating K diagonal blocks."""
    model, nodes = problem.response_state_space, problem.grouped_systems[run]
    d = model.dimension
    if not problem.dynamic_state_space:
        full = problem.state_space_systems[run]
        return (
            full.order,
            jnp.asarray(full.transition[:, :d, :d]),
            jnp.asarray(full.process_covariance[:, :d, :d]),
            jnp.asarray(model.outputs),
        )
    if problem.parameterized_responses:
        state = model.realize(x, problem.indices)
        outputs, lags = state[2:]
        times = jnp.asarray(nodes.times) - lags[nodes.modalities]
        if transition_cache and len(nodes.times) > transition_cache:
            order = jnp.argsort(times, stable=True)
            shifted = times[order]
            delta = jnp.diff(shifted, prepend=shifted[0])
            first, inverse, count = interval_groups(
                jnp.asarray(nodes.times),
                jnp.asarray(nodes.modalities),
                order,
                shifted,
                transition_cache,
            )

            def reduced(_):
                a, q = parameter_transitions(model, state[0], state[1], delta[first], 1)
                return a[inverse], q[inverse]

            def full(_):
                return parameter_transitions(model, state[0], state[1], delta, 1)

            A, Q = jax.lax.cond(count <= transition_cache, reduced, full, None)
            return order, A, Q, outputs
        system = parameter_event_system(model, state, times, 1)
    else:
        outputs = jnp.asarray(model.outputs)
        lags = jnp.concatenate((problem.arrays(x)[4], jnp.zeros(1)))
        times = jnp.asarray(nodes.times) - lags[nodes.modalities]
        system = delay_events(times, problem.delay_transition, 1)
    return system.order, system.transition, system.process_covariance, outputs


def nll(problem, x, run, *, joseph="dense", transition_cache=0):
    nodes = problem.grouped_systems[run]
    order, A, Q, outputs = events(problem, x, run, transition_cache=transition_cache)
    precision, information, weights, residual, variance = statistics(problem, x, run)
    dimension = problem.features * problem.response_state_space.dimension

    def step(carry, inputs):
        m, p, score = carry
        m, p, increment, projected, _, _ = update(m, p, *inputs, joseph=joseph)
        return (m, p, score + increment), projected

    final, means = jax.lax.scan(
        jax.checkpoint(step, prevent_cse=False),
        (jnp.zeros(dimension), jnp.eye(dimension), jnp.asarray(0.0)),
        (A, Q, outputs[nodes.modalities][order], precision[order], information[order]),
    )
    means = means[jnp.argsort(order)]
    error = residual - jnp.sum(weights * means[nodes.observation_nodes], axis=1)
    return 0.5 * (final[2] + jnp.sum(error**2 / variance + jnp.log(variance) + np.log(2 * np.pi)))
