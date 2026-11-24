"""Model-based (Laplace) standard errors for response lags at a MAP fit.

The negative log posterior with the latent GP marginalized is expanded to
second order at the MAP parameters, in physical coordinates. Its inverse
Hessian is the covariance of a Gaussian approximation to the posterior, and a
lag's standard error is the square root of the matching diagonal entry. These
errors assume the model is right; they say nothing about misspecification.
"""

import time
from itertools import combinations

import numpy as np

from ._backend import runtime

# A dense P x P Hessian takes P Hessian-vector products and an O(P^3)
# eigendecomposition. Above this many parameters that is not a quick report.
MAX_PARAMETERS = 2000
# Eigenvalues this small relative to the largest are treated as zero, exactly
# as numpy.linalg.pinv(..., rcond=1e-9, hermitian=True) does.
NULL_RCOND = 1e-9
# The same closeness to a bound that the MAP diagnostics report.
BOUND_TOLERANCE = 1e-7
LABEL = (
    "Model-based (Laplace) standard errors of response lags at the MAP. "
    "They assume the fitted model is correct and are too narrow when it is not. "
    "Response widths and other shape parameters are not covered."
)


def objective_hessian(problem, x, *, free=None, batch_size=32):
    """Symmetrized Hessian of ``problem.objective`` at ``x``.

    Built from Hessian-vector products (forward over reverse), ``batch_size``
    at a time, so peak memory is that many gradient evaluations rather than
    one per parameter. ``free`` is a Boolean mask selecting the rows and
    columns to compute; the default is all of them.
    """
    jax, jnp, _, _ = runtime()
    x = jnp.asarray(x)
    free = np.ones(len(x), dtype=bool) if free is None else np.asarray(free, dtype=bool)
    gradient = jax.grad(problem.objective)

    def product(direction):
        return jax.jvp(gradient, (x,), (direction,))[1]

    directions = jnp.eye(len(x))[free]
    hessian = np.asarray(jax.lax.map(product, directions, batch_size=batch_size))[:, free]
    return (hessian + hessian.T) / 2


def _check_supported(model, max_parameters):
    configuration = model.configuration_
    if configuration["inference"] != "map":
        raise ValueError(
            "uncertainty='laplace' requires a MAP fit; a posterior fit already has "
            "parameter draws, which timing_report() summarizes as mean and sd"
        )
    if configuration.get("parameter_source") == "original_training_MAP":
        raise ValueError(
            "uncertainty='laplace' requires the training fit: after frozen conditioning "
            "the parameters are not an optimum of the current observations"
        )
    if model.problem_.linear_algebra not in ("dense", "grouped", "state_space"):
        raise ValueError(
            "uncertainty='laplace' requires dense, grouped or state_space linear algebra"
        )
    count = len(model.problem_.names)
    if count > max_parameters:
        raise ValueError(
            f"uncertainty='laplace' builds a dense Hessian over all {count} parameters; "
            f"the limit is {max_parameters}"
        )


def lag_standard_errors(model, reference, *, max_parameters=MAX_PARAMETERS, batch_size=32):
    """Laplace standard errors of lags, relative lags and lag differences.

    ``reference`` is the modality shown at zero. A fixed lag has zero
    variance; a relative lag uses ``var_m + var_ref - 2 cov``. The Hessian is
    pseudo-inverted, because several factors leave exact zero directions
    (factor rotations) that lags do not depend on.

    A parameter resting on a bound is a constrained optimum: the objective
    need not curve upwards along it, so it is held fixed and the Hessian
    covers the remaining parameters (``parameters_at_bounds`` counts those
    held). A quantity is NaN when a lag it uses rests on a bound
    (``lags_at_bounds``) or when it does depend on a zero direction
    (``not_identified``). ``positive_definite`` is False when an eigenvalue
    outside the zero directions is negative: the fit is then not at a local
    optimum and the errors should not be used.
    """
    _check_supported(model, max_parameters)
    problem = model.problem_
    x = np.asarray(model.map_parameters_, dtype=float)
    lower, upper = np.asarray(problem.bounds, dtype=float).T
    slack = BOUND_TOLERANCE * np.maximum(1.0, np.abs(x))
    free = (x - lower > slack) & (upper - x > slack)
    began = time.perf_counter()
    hessian = objective_hessian(problem, x, free=free, batch_size=batch_size)
    seconds = time.perf_counter() - began
    values, vectors = np.linalg.eigh(hessian)
    tolerance = NULL_RCOND * np.max(np.abs(values))
    null = np.abs(values) <= tolerance
    positive_definite = bool(np.all(values[~null] > 0))
    covariance = np.zeros((len(x), len(x)))
    covariance[np.ix_(free, free)] = (vectors[:, ~null] / values[~null]) @ vectors[:, ~null].T
    null_vectors = np.zeros((len(x), int(null.sum())))
    null_vectors[free] = vectors[:, null]

    lags, contrasts, at_bounds = {}, {}, []
    for m, response in problem.responses.items():
        contrasts[m] = np.zeros(len(x))
        index = problem.indices.get(("filter", m, "lag"))
        if index is None:
            lags[m] = float(getattr(response.initial_kernel(), "lag", 0.0))
            continue
        lags[m] = float(x[index])
        contrasts[m][index] = 1.0
        if not free[index]:
            at_bounds.append(m)

    not_identified = []

    def entry(name, estimate, contrast, modalities):
        error = float("nan")
        if np.linalg.norm(null_vectors.T @ contrast) > 1e-6:
            not_identified.append(name)
        elif not set(modalities) & set(at_bounds):
            variance = float(contrast @ covariance @ contrast)
            error = float(np.sqrt(variance)) if variance >= 0 else float("nan")
        return dict(estimate=float(estimate), standard_error=error)

    features = problem.features
    return dict(
        method="laplace",
        label=LABEL,
        quantity="lag",
        units="timestamp units (normally seconds)",
        reference=reference,
        lag_model_clock={m: entry(f"lag {m}", lags[m], contrasts[m], [m]) for m in lags},
        relative_lag={
            m: entry(
                f"{m} - {reference}",
                lags[m] - lags[reference],
                contrasts[m] - contrasts[reference],
                [m, reference],
            )
            for m in lags
        },
        pairwise_lag_difference={
            f"{b} - {a}": entry(
                f"{b} - {a}", lags[b] - lags[a], contrasts[b] - contrasts[a], [a, b]
            )
            for a, b in combinations(lags, 2)
        },
        # False where free shape parameters move the peak away from the lag.
        applies_to_peak_delay={
            m: type(response.initial_kernel()).__name__ in ("Identity", "Gaussian")
            or not set(response.free_parameters) - {"lag"}
            for m, response in problem.responses.items()
        },
        lags_at_bounds=at_bounds,
        not_identified=not_identified,
        positive_definite=positive_definite,
        min_eigenvalue=float(values.min()),
        null_directions=int(null.sum()),
        expected_null_directions=features * (features - 1) // 2,
        parameters=len(x),
        parameters_at_bounds=int((~free).sum()),
        hessian_seconds=seconds,
    )
