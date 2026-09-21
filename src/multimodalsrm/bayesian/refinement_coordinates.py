"""Physical filter boxes for multi-factor MAP refinement, without a Jacobian."""

import numpy as np

from ._backend import runtime


def filter_coordinates(problem, x):
    """Return mixed coordinates/evaluator, or None for the original MAP path.

    Closed, finite filter boxes can attain their constrained optima without
    saturating a logistic transform. All other parameters retain their existing
    transforms and open-support behavior. This is never a sampling transform.
    """
    if getattr(problem, "features", 1) <= 1:
        return None
    indices = np.array(
        [
            j
            for j, name in enumerate(problem.names)
            if name[0] == "filter" and np.isfinite(problem.bounds[j]).all()
        ],
        dtype=int,
    )
    if not len(indices):
        return None
    x = np.asarray(x, float)
    bounds = np.asarray(problem.bounds)[indices]
    if (
        not np.isfinite(x[indices]).all()
        or np.any(x[indices] < bounds[:, 0])
        or np.any(x[indices] > bounds[:, 1])
    ):
        raise ValueError("filter parameters must lie within physical bounds")
    interior = x.copy()
    interior[indices] = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) / 2
    initial = problem.to_unconstrained(interior)
    initial[indices] = x[indices]
    jax, jnp, _, _ = runtime()

    def physical(z):
        z = jnp.asarray(z)
        safe = z.at[indices].set(0.0)
        return problem.from_unconstrained(safe)[0].at[indices].set(z[indices])

    # BayesianProblem's transforms act independently on each coordinate, so a
    # JVP along ones gives their diagonal derivatives. Reuse its compiled
    # physical density/gradient instead of compiling another large GP graph.
    transform = jax.jit(lambda z: jax.jvp(physical, (z,), (jnp.ones_like(z),)))

    def evaluate(z):
        point, derivative = transform(jnp.asarray(z))
        value, gradient = problem.value_gradient(np.asarray(point))
        return value, gradient * np.asarray(derivative)

    boxes = [(None, None)] * len(x)
    for j in indices:
        boxes[j] = problem.bounds[j]
    return initial, physical, evaluate, boxes, indices.tolist()
