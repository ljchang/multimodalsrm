"""Elapsed-time transitions for a fixed realization with learned response delays."""

import numpy as np

from ._backend import runtime


def transition_function(model):
    """Return a JAX transition with its exact elapsed-time derivative.

    The generator and response normalization are fixed. An analytic JVP avoids
    differentiating the discrete scaling choice used for stable process-noise
    integration, and preserves derivatives at exactly coincident events.
    """
    jax, jnp, jsp, _ = runtime()
    d = model.dimension
    F, b = jnp.asarray(model.generator), jnp.asarray(model.driving)
    identity = jnp.eye(d)
    block = jnp.block([[F, jnp.outer(b, b)], [jnp.zeros_like(F), -F.T]])
    size = float(np.linalg.norm(model.generator, ord=1))
    cutoff = (400 + 2 * d) / model.minimum_rate

    @jax.custom_jvp
    def transition(delta):
        interval = jnp.minimum(delta, cutoff)
        doublings = jnp.ceil(jnp.log2(jnp.maximum(size * interval / 0.5, 1.0))).astype(jnp.int32)
        step = interval / 2.0**doublings
        E = jsp.linalg.expm(block * step)
        A = E[:d, :d]
        Q = E[:d, d:] @ A.T

        def double(_, pair):
            A, Q = pair
            return A @ A, Q + A @ Q @ A.T

        A, Q = jax.lax.fori_loop(0, doublings, double, (A, Q))
        return (
            jnp.where(delta > cutoff, jnp.zeros_like(A), A),
            jnp.where(delta > cutoff, identity, (Q + Q.T) * 0.5),
        )

    @transition.defjvp
    def derivative(primals, tangents):
        (delta,), (tangent,) = primals, tangents
        A, Q = transition(delta)
        ab = A @ b
        return (A, Q), (tangent * (F @ A), tangent * jnp.outer(ab, ab))

    return transition


def event_system(times, transition, features):
    """Sort parameter-dependent event clocks and expand independent factors."""
    from .state_space import StateSpaceSystem

    jax, jnp, _, _ = runtime()
    order = jnp.argsort(times, stable=True)
    ordered = times[order]
    deltas = jnp.diff(ordered, prepend=ordered[0])
    A, Q = jax.vmap(transition)(deltas)
    return StateSpaceSystem(
        order, jnp.kron(jnp.eye(features), A), jnp.kron(jnp.eye(features), Q), ordered
    )


def observation_system(problem, x, run):
    _, jnp, _, _ = runtime()
    lags = problem.arrays(x)[4]
    times = jnp.asarray(problem.systems[run].times) - lags[problem._packed[run][2]]
    return event_system(times, problem.delay_transition, problem.features)


def tied_smoother_update():
    """Copy identical-time states while retaining elapsed-time sensitivities.

    At the tie P_pred=P and A=I. The gain can be written
    I+(P A.T-P_pred) pinv(P_pred): its value is I and its first derivative is
    valid on the range of P, including after noiseless observations. Smoothing
    corrections lie in this range. The pseudoinverse is needed only when
    differentiating; ordinary prediction keeps the exact, inexpensive copy.
    """
    jax, jnp, _, _ = runtime()

    @jax.custom_jvp
    def tied(mean, covariance, pm, pc, transition, next_mean, next_covariance):
        return next_mean, next_covariance

    @tied.defjvp
    def derivative(primals, tangents):
        inverse = jnp.linalg.pinv(primals[3], hermitian=True)

        def expression(mean, covariance, pm, pc, transition, next_mean, next_covariance):
            gain = jnp.eye(len(mean)) + (covariance @ transition.T - pc) @ inverse
            sm = mean + gain @ (next_mean - pm)
            sc = covariance + gain @ (next_covariance - pc) @ gain.T
            return sm, (sc + sc.T) * 0.5

        _, tangent = jax.jvp(expression, primals, tangents)
        return (primals[-2], primals[-1]), tangent

    return tied


def covariance(problem, x, times_a, modalities_a, times_b, modalities_b):
    """Differentiable dense diagnostic; inference never calls this function."""
    jax, jnp, _, _ = runtime()
    model = problem.response_state_space
    ma, mb = np.asarray(modalities_a), np.asarray(modalities_b)
    lags = jnp.concatenate((problem.arrays(x)[4], jnp.zeros(1)))
    ta, tb = jnp.asarray(times_a) - lags[ma], jnp.asarray(times_b) - lags[mb]
    delta = (ta[:, None] - tb[None, :]).ravel()
    ca = jnp.asarray(model.outputs)[np.repeat(ma, len(mb))]
    cb = jnp.asarray(model.outputs)[np.tile(mb, len(ma))]

    def entry(dt, a, b):
        A, _ = problem.delay_transition(jnp.abs(dt))
        return jnp.where(dt >= 0, a @ A @ b, b @ A @ a)

    # Bound temporary transition storage independently of the pair count.
    values = [
        jax.vmap(entry)(delta[i : i + 64], ca[i : i + 64], cb[i : i + 64])
        for i in range(0, len(delta), 64)
    ]
    return jnp.concatenate(values).reshape(len(ma), len(mb))
