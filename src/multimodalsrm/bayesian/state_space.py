"""Matérn-3/2 inference with response states on native observation clocks.

The state of each factor is (value, derivative / rate), rate=sqrt(3)/length.
This normalization makes its stationary covariance the 2x2 identity. Runs
are independent; repeated measurements retain their individual noise. Responses
with strictly positive noise use exact modality/time node updates.
"""

import math
from dataclasses import dataclass

import numpy as np

from ._backend import runtime


def validate(problem):
    if problem.response_quadrature_order is not None:
        raise ValueError("state_space does not support response quadrature")
    if problem.run_baseline_sd:
        raise ValueError("state_space does not support run baselines")
    if problem.noise_timescales:
        raise ValueError("state_space does not support correlated noise")
    from .state_space_responses import ResponseStateSpace

    arguments = (
        problem.responses,
        problem.length_scale,
        problem.adapter.covariance_tolerance,
    )
    if problem.state_space_gaussian == "auto":
        return ResponseStateSpace.prepare(*arguments)
    return ResponseStateSpace.prepare(*arguments, gaussian_method=problem.state_space_gaussian)


def transitions(deltas, length_scale, features):
    """Exact stationary transitions, including zero and very small intervals."""
    # Above u=400 all transition terms are negligible in float64. Clipping
    # before polynomial evaluation also avoids inf * 0 for enormous gaps.
    with np.errstate(over="ignore"):
        u = np.minimum(np.asarray(deltas) * (np.sqrt(3.0) / length_scale), 400.0)
    e = np.exp(-u)
    A = np.empty((len(u), 2, 2))
    A[:, 0, 0], A[:, 0, 1] = e * (1 + u), e * u
    A[:, 1, 0], A[:, 1, 1] = -e * u, e * (1 - u)
    e2 = e**2
    q11 = -np.expm1(-2 * u) - 2 * u * (1 + u) * e2
    small = u < 1e-3
    v = u[small]
    # q11 = 4 integral_0^u s^2 exp(-2s) ds; the direct expression cancels
    # to third order. This convergent series preserves tiny process noise.
    q11[small] = sum(
        4 * (-2.0) ** j * v ** (j + 3) / (math.factorial(j) * (j + 3)) for j in range(9)
    )
    Q = np.empty_like(A)
    Q[:, 0, 0] = q11
    Q[:, 0, 1] = Q[:, 1, 0] = 2 * u**2 * e2
    Q[:, 1, 1] = -np.expm1(-2 * u) + 2 * u * (1 - u) * e2
    full_A = np.zeros((len(u), 2 * features, 2 * features))
    full_Q = np.zeros_like(full_A)
    for k in range(features):
        block = slice(2 * k, 2 * k + 2)
        full_A[:, block, block], full_Q[:, block, block] = A, Q
    return full_A, full_Q


@dataclass(frozen=True)
class StateSpaceSystem:
    order: np.ndarray
    transition: np.ndarray
    process_covariance: np.ndarray
    times: np.ndarray

    @classmethod
    def prepare(cls, times, model, features):
        order = np.argsort(times, kind="stable")
        sorted_times = np.asarray(times)[order]
        deltas = np.diff(sorted_times, prepend=sorted_times[0])
        A, Q = model.transitions(deltas, features)
        return cls(order, A, Q, sorted_times)


def operands(problem, x, run, system, outputs=None):
    _, jnp, _, _ = runtime()
    ki, gi, mi = problem._packed[run]
    weights, offsets, noise, _, _ = problem.arrays(x)
    weights = weights.reshape(len(problem.keys), problem.features)
    rows = jnp.asarray(problem.response_state_space.outputs if outputs is None else outputs)[mi]
    H = (weights[ki, :, None] * rows[:, None, :]).reshape(len(ki), -1)
    y = jnp.asarray(problem.systems[run].values) - offsets[ki]
    order = system.order
    return H[order], y[order], noise[gi][order]


def update(mean, covariance, A, Q, H, y, variance):
    """Predict and assimilate one scalar observation using a Joseph update."""
    _, jnp, _, _ = runtime()
    predicted_mean = A @ mean
    predicted_covariance = A @ covariance @ A.T + Q
    ph = predicted_covariance @ H
    innovation_variance = H @ ph + variance
    innovation = y - H @ predicted_mean
    gain = ph / innovation_variance
    mean = predicted_mean + gain * innovation
    residual_map = jnp.eye(len(mean)) - jnp.outer(gain, H)
    covariance = residual_map @ predicted_covariance @ residual_map.T + variance * jnp.outer(
        gain, gain
    )
    covariance = (covariance + covariance.T) * 0.5
    score = 0.5 * (jnp.log(2 * np.pi * innovation_variance) + innovation**2 / innovation_variance)
    return mean, covariance, score, predicted_mean, predicted_covariance


def nll(problem, x, run):
    """Marginal likelihood without a dense observation covariance or jitter."""
    if getattr(problem, "grouped_state_space", False):
        from .state_space_grouped import nll as grouped_nll

        return grouped_nll(problem, x, run)
    jax, jnp, _, _ = runtime()
    outputs = None
    if problem.parameterized_responses:
        from .state_space_parameters import parameter_event_system

        state = problem.response_state_space.realize(x, problem.indices)
        outputs = state[2]
        times = jnp.asarray(problem.systems[run].times) - state[3][problem._packed[run][2]]
        system = parameter_event_system(
            problem.response_state_space, state, times, problem.features
        )
    elif problem.learned_response_lags:
        from .state_space_delays import observation_system

        system = observation_system(problem, x, run)
    else:
        system = problem.state_space_systems[run]
    H, y, variance = operands(problem, x, run, system, outputs)

    def step(carry, obs):
        mean, covariance, score = carry
        mean, covariance, increment, _, _ = update(mean, covariance, *obs)
        return (mean, covariance, score + increment), None

    dimension = problem.response_state_space.dimension * problem.features
    initial = (jnp.zeros(dimension), jnp.eye(dimension), jnp.asarray(0.0))
    final, _ = jax.lax.scan(
        step,
        initial,
        (
            jnp.asarray(system.transition),
            jnp.asarray(system.process_covariance),
            H,
            y,
            variance,
        ),
    )
    return final[2]


def _smoother_gain(covariance, predicted, transition, semidefinite):
    """RTS gain on the representable covariance range for exact observations.

    A noiseless update removes a state direction. Over a nearly zero interval,
    process variance in that direction can be below floating-point resolution
    in a rotated state basis, so a positive-definite factorization is unsafe.
    The generalized conditional gain uses the Hermitian pseudoinverse there;
    its dtype-aware rank cutoff does not insert variance, jitter, or an
    eigenvalue floor. Strictly positive observation noise retains Cholesky,
    including resolved covariance scales below the pseudoinverse cutoff.
    """
    jax, jnp, jsp, _ = runtime()
    predicted = (predicted + predicted.T) * 0.5
    cross = transition @ covariance

    def positive_definite(_):
        chol = jnp.linalg.cholesky(predicted)
        return jsp.linalg.cho_solve((chol, True), cross).T

    return jax.lax.cond(
        semidefinite,
        lambda _: (jnp.linalg.pinv(predicted, hermitian=True) @ cross).T,
        positive_definite,
        None,
    )


def _smooth_backward(last, history, A, *, semidefinite, dynamic=False):
    """Common RTS recursion for scalar and grouped observation updates."""
    jax, jnp, _, _ = runtime()
    dimension = len(last[0])
    means, covariances, predicted_means, predicted_covariances = history
    if dynamic:
        from .state_space_delays import tied_smoother_update

        tied_update = tied_smoother_update()

    def backward(carry, operands):
        next_mean, next_covariance = carry
        mean, covariance, pm, pc, transition = operands

        def advance(_):
            gain = _smoother_gain(covariance, pc, transition, semidefinite)
            sm = mean + gain @ (next_mean - pm)
            sc = covariance + gain @ (next_covariance - pc) @ gain.T
            return sm, (sc + sc.T) * 0.5

        # At equal times both events share exactly the same state. Copying
        # its smoothed distribution also supports noiseless observations,
        # whose filtered covariance can be singular (so no solve is valid).
        result = jax.lax.cond(
            jnp.all(transition == jnp.eye(dimension)),
            lambda _: (
                tied_update(mean, covariance, pm, pc, transition, *carry) if dynamic else carry
            ),
            advance,
            None,
        )
        return result, result

    _, (sm, sc) = jax.lax.scan(
        backward,
        last,
        (
            means[:-1],
            covariances[:-1],
            predicted_means[1:],
            predicted_covariances[1:],
            A[1:],
        ),
        reverse=True,
    )
    return (
        jnp.concatenate((sm, last[0][None])),
        jnp.concatenate((sc, last[1][None])),
    )


def smoother(problem, run, times, query_modality=-1):
    """Prepare a differentiable fixed-query RTS smoother returning factor moments.

    Queries are zero-loading events: they introduce no observations or temporal
    discretization. Stationarity permits queries before the first observation.
    """
    if getattr(problem, "grouped_state_space", False):
        from .state_space_grouped import smoother as grouped_smoother

        return grouped_smoother(problem, run, times, query_modality)
    jax, jnp, _, _ = runtime()
    model = problem.response_state_space
    if not problem.dynamic_state_space:
        observations = problem.state_space_systems[run]
        events = StateSpaceSystem.prepare(
            np.concatenate((observations.times, times - model.lags[query_modality])),
            model,
            problem.features,
        )
    dimension = model.dimension * problem.features
    row = jnp.asarray(model.outputs[query_modality])

    def one(x):
        outputs = None
        if problem.parameterized_responses:
            from .state_space_parameters import parameter_event_system

            state = model.realize(x, problem.indices)
            outputs, lags = state[2:]
            shifted = jnp.asarray(problem.systems[run].times) - lags[problem._packed[run][2]]
            # Operands need only the observation order; transition integration
            # is performed once for the merged observation/query events.
            order = jnp.argsort(shifted, stable=True)
            obs = StateSpaceSystem(order, None, None, shifted[order])
            merged = parameter_event_system(
                model,
                state,
                jnp.concatenate((obs.times, jnp.asarray(times) - lags[query_modality])),
                problem.features,
            )
        elif problem.learned_response_lags:
            from .state_space_delays import event_system, observation_system

            obs = observation_system(problem, x, run)
            lags = jnp.concatenate((problem.arrays(x)[4], jnp.zeros(1)))
            merged = event_system(
                jnp.concatenate((obs.times, jnp.asarray(times) - lags[query_modality])),
                problem.delay_transition,
                problem.features,
            )
        else:
            obs, merged = observations, events
        query_indices = jnp.argsort(merged.order)[len(obs.times) :]
        A, Q = jnp.asarray(merged.transition), jnp.asarray(merged.process_covariance)
        H, y, variance = operands(problem, x, run, obs, outputs)
        semidefinite = jnp.any(variance == 0)
        H = jnp.concatenate((H, jnp.zeros((len(times), dimension))))[merged.order]
        y = jnp.concatenate((y, jnp.zeros(len(times))))[merged.order]
        variance = jnp.concatenate((variance, jnp.ones(len(times))))[merged.order]

        def forward(carry, obs):
            mean, covariance, _, predicted_mean, predicted_covariance = update(*carry, *obs)
            return (mean, covariance), (
                mean,
                covariance,
                predicted_mean,
                predicted_covariance,
            )

        last, (means, covariances, predicted_means, predicted_covariances) = jax.lax.scan(
            forward,
            (jnp.zeros(dimension), jnp.eye(dimension)),
            (A, Q, H, y, variance),
        )

        sm, sc = _smooth_backward(
            last,
            (means, covariances, predicted_means, predicted_covariances),
            A,
            semidefinite=semidefinite,
            dynamic=problem.dynamic_state_space,
        )
        sm, sc = sm[query_indices], sc[query_indices]
        projection = row if outputs is None else outputs[query_modality]
        sm = sm.reshape(len(times), problem.features, model.dimension) @ projection
        sc = sc.reshape(
            len(times),
            problem.features,
            model.dimension,
            problem.features,
            model.dimension,
        )
        return sm, jnp.einsum("i,tkilj,j->tkl", projection, sc, projection)

    return one


def project(problem, draws, run, times, *, key, include_noise, latent_loading):
    """Project smoothed factor moments without building a dense time covariance."""
    jax, jnp, _, _ = runtime()
    modality = -1 if key is None else problem.modalities.index(key[1])
    smooth = smoother(problem, run, np.asarray(times), modality)
    output = problem.response_state_space.outputs[modality]
    response_variance = float(output @ output)

    def one(x):
        mean, covariance = smooth(x)
        prior_variance = response_variance
        if problem.parameterized_responses:
            output = problem.response_state_space.realize(x, problem.indices)[2][modality]
            prior_variance = output @ output
        weights, offsets, noise, _, _ = problem.arrays(x)
        if key is None:
            loading = (
                jnp.eye(problem.features)[0]
                if latent_loading is None
                else jnp.asarray(latent_loading)
            )
            offset, measurement = 0.0, 0.0
        else:
            index = problem.keys.index(key)
            loading = weights.reshape(len(problem.keys), problem.features)[index]
            offset = offsets[index]
            measurement = noise[problem.groups.index(key[:2])] if include_noise else 0.0
        return (
            mean @ loading + offset,
            jnp.einsum("i,tij,j->t", loading, covariance, loading) + measurement,
            jnp.sum(loading**2) * prior_variance + measurement,
        )

    evaluate = jax.jit(jax.vmap(one))
    means = np.empty((len(draws), len(times)))
    variances = np.empty_like(means)
    for start in range(0, len(draws), 16):
        ds = slice(start, start + 16)
        mean, variance, prior = (np.asarray(v) for v in evaluate(jnp.asarray(draws[ds])))
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise FloatingPointError("nonfinite state-space posterior prediction")
        if np.any(variance < -1e-9 * np.maximum(1.0, np.abs(prior[:, None]))):
            raise FloatingPointError("materially negative conditional variance")
        means[ds], variances[ds] = mean, np.maximum(variance, 0.0)
    return means, variances
