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
# Relative to the largest eigenvalue: below this an eigenvalue outside the known
# invariant directions is not positive, and a common shift of every lag counts
# as an exact invariance.
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


def invariant_directions(problem, x):
    """Tangents of the exact continuous invariances of the objective at ``x``.

    Several factors share one latent covariance and one isotropic, unbounded
    Gaussian loading prior, so rotating the factors leaves the objective
    unchanged. For each factor pair ``a < b`` the tangent moves loadings only:
    ``dW[:, a] = W[:, b]`` and ``dW[:, b] = -W[:, a]``. One factor has no such
    direction: its only symmetry is the sign flip, which is discrete and which
    the nonnegative anchor loading removes. Returns one column per pair.
    """
    features, count = problem.features, len(problem.keys)
    pairs = list(combinations(range(features), 2))
    directions = np.zeros((len(x), len(pairs)))
    if pairs:
        # Loadings come first, one row of factors per observed feature.
        loadings = np.asarray(x[: count * features]).reshape(count, features)
        for column, (a, b) in enumerate(pairs):
            tangent = np.zeros_like(loadings)
            tangent[:, a], tangent[:, b] = loadings[:, b], -loadings[:, a]
            directions[: count * features, column] = tangent.ravel()
    return directions


def _covariance(model, max_parameters, batch_size):
    """Inverse Hessian at the MAP on the identified directions, with diagnostics.

    The Hessian is inverted on the orthogonal complement of the directions
    along which the objective is exactly invariant, which are known in closed
    form (``invariant_directions``) rather than found by an eigenvalue
    cut-off: ``null_directions`` is their number. Lags do not depend on
    factor rotations. The curvature along a rotation is zero only where the
    gradient vanishes, so ``null_direction_residual`` reports the largest
    ``|H q|`` over those directions relative to the largest eigenvalue; it
    shrinks as the fit is tightened. A common shift of every lag is a further
    invariance of the likelihood. Lag priors normally remove it, and when they
    are flat it is found numerically (``H q = 0`` holds for it at any point)
    and added.

    A parameter resting on a bound is a constrained optimum: the objective
    need not curve upwards along it, so it is held fixed and the Hessian
    covers the remaining parameters (``parameters_at_bounds`` counts those
    held). Loadings of several factors are unbounded, so rotations are never
    affected. ``positive_definite`` is False when the Hessian has
    a nonpositive eigenvalue on the complement: the fit is then not at a local
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
    scale = float(np.linalg.norm(hessian, 2))

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

    directions = invariant_directions(problem, x)[free]
    shift = sum(contrasts.values())[free]
    if (
        all(("filter", m, "lag") in problem.indices for m in lags)
        and not at_bounds
        and np.linalg.norm(hessian @ shift) <= NULL_RCOND * scale * np.linalg.norm(shift)
    ):
        directions = np.column_stack([directions, shift])
    nullity = directions.shape[1]
    basis = np.linalg.qr(directions, mode="complete")[0]
    null_basis, complement = basis[:, :nullity], basis[:, nullity:]
    residual = float(np.linalg.norm(hessian @ null_basis, 2)) / scale if nullity else 0.0
    values, vectors = np.linalg.eigh(complement.T @ hessian @ complement)
    positive_definite = bool(values.min() > NULL_RCOND * scale)
    keep = np.abs(values) > NULL_RCOND * scale
    vectors = complement @ vectors[:, keep]
    covariance = np.zeros((len(x), len(x)))
    covariance[np.ix_(free, free)] = (vectors / values[keep]) @ vectors.T
    null_vectors = np.zeros((len(x), nullity))
    null_vectors[free] = null_basis
    return dict(
        x=x,
        free=free,
        lags=lags,
        contrasts=contrasts,
        at_bounds=at_bounds,
        covariance=covariance,
        null_vectors=null_vectors,
        diagnostics=dict(
            positive_definite=positive_definite,
            min_eigenvalue=float(values.min()),
            null_directions=nullity,
            null_direction_residual=residual,
            parameters=len(x),
            parameters_at_bounds=int((~free).sum()),
            hessian_seconds=seconds,
        ),
    )


def lag_standard_errors(model, reference, *, max_parameters=MAX_PARAMETERS, batch_size=32):
    """Laplace standard errors of lags, relative lags and lag differences.

    ``reference`` is the modality shown at zero. A fixed lag has zero
    variance; a relative lag uses ``var_m + var_ref - 2 cov``. The covariance
    is the inverse Hessian that ``_covariance`` describes. A quantity is NaN
    when a lag it uses rests on a bound (``lags_at_bounds``) or when it
    depends on an invariant direction (``not_identified``).
    """
    state = _covariance(model, max_parameters, batch_size)
    problem = model.problem_
    lags, contrasts, at_bounds = state["lags"], state["contrasts"], state["at_bounds"]
    covariance, null_vectors = state["covariance"], state["null_vectors"]

    not_identified = []

    def entry(name, estimate, contrast, modalities):
        error = float("nan")
        if np.linalg.norm(null_vectors.T @ contrast) > 1e-6:
            not_identified.append(name)
        elif not set(modalities) & set(at_bounds):
            variance = float(contrast @ covariance @ contrast)
            error = float(np.sqrt(variance)) if variance >= 0 else float("nan")
        return dict(estimate=float(estimate), standard_error=error)

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
        **state["diagnostics"],
    )


def noise_timescale_standard_errors(model, *, max_parameters=MAX_PARAMETERS, batch_size=32):
    """Laplace standard errors of estimated noise timescales, in seconds.

    One entry per modality with the timescale, its standard error, and the
    lag-one autocorrelation at the median sampling interval with its
    delta-method standard error. A timescale resting on a bound is held fixed
    in the Hessian and gets NaN.
    """
    state = _covariance(model, max_parameters, batch_size)
    problem = model.problem_
    x, free, covariance = state["x"], state["free"], state["covariance"]
    modalities, at_bounds = {}, []
    for index in problem._noise_timescale_indices:
        modality = problem.names[index][1]
        interval = problem.noise_sampling[modality][0]
        timescale = float(x[index])
        correlation = float(np.exp(-interval / timescale))
        error = float("nan")
        if free[index]:
            error = float(np.sqrt(covariance[index, index]))
        else:
            at_bounds.append(modality)
        modalities[modality] = dict(
            timescale=timescale,
            standard_error=error,
            lag_one_autocorrelation=correlation,
            autocorrelation_standard_error=correlation * interval / timescale**2 * error,
        )
    return dict(
        method="laplace",
        label=(
            "Model-based (Laplace) standard errors of noise timescales at the MAP. "
            "They assume the fitted model is correct and are too narrow when it is not."
        ),
        quantity="noise_timescale",
        units="seconds",
        modalities=modalities,
        timescales_at_bounds=at_bounds,
        **state["diagnostics"],
    )
