"""Standard errors for response lags at a MAP fit: Laplace and cluster-robust.

The negative log posterior with the latent GP marginalized is expanded to
second order at the MAP parameters, in physical coordinates. Its inverse
Hessian is the covariance of a Gaussian approximation to the posterior, and a
lag's standard error is the square root of the matching diagonal entry. These
errors assume the model is right; they say nothing about misspecification.

The cluster-robust (sandwich) errors drop that assumption for anything that
is independent between runs. The objective is ``sum_run nll_run(x) -
log_prior(x)``, so the MAP is a penalized estimator and the run scores
``g_b = grad nll_b`` sum to the gradient of the log prior, not to zero. The
prior is treated as a fixed penalty: it stays in the Hessian ``H``, and the
score covariance uses scores centred across runs, ``s_b = g_b - mean(g)``,
which shares the prior gradient equally among runs. With
``J = sum_b s_b s_b'`` the covariance is ``H^+ J H^+``, so a contrast ``a``
has variance ``sum_b (a' H^+ s_b)^2``: a sum of B squares with B - 1 degrees
of freedom, scaled by ``B / (B - 1)``.
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
# Two clusters give a variance from a single difference; below FEW_CLUSTERS the
# sandwich is flagged as noisy.
MIN_CLUSTERS = 3
FEW_CLUSTERS = 10
ROBUST_LABEL = (
    "Cluster-robust (sandwich) standard errors of response lags at the MAP, with "
    "the fitted runs as clusters and the small-sample factor B / (B - 1). They "
    "allow for misspecification that is independent between runs, and are noisy "
    "when runs are few: use t quantiles with clusters - 1 degrees of freedom."
)
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


def run_scores(problem, x):
    """Gradient of each run's negative log likelihood at ``x``.

    Shape ``(runs, parameters)``, rows in the order of ``problem.systems``.
    The prior is not included, so at an interior MAP the rows sum to the
    gradient of ``problem.log_prior``.
    """
    jax, jnp, _, _ = runtime()

    def likelihoods(v):
        return jnp.stack([problem.run_nll(v, run) for run in problem.systems])

    return np.asarray(jax.jit(jax.jacrev(likelihoods))(jnp.asarray(x)))


def _check_supported(model, max_parameters, robust=False):
    option = "uncertainty='robust'" if robust else "uncertainty='laplace'"
    configuration = model.configuration_
    if configuration["inference"] != "map":
        raise ValueError(
            f"{option} requires a MAP fit; a posterior fit already has "
            "parameter draws, which timing_report() summarizes as mean and sd"
        )
    if configuration.get("parameter_source") == "original_training_MAP":
        raise ValueError(
            f"{option} requires the training fit: after frozen conditioning "
            "the parameters are not an optimum of the current observations"
        )
    if model.problem_.linear_algebra not in ("dense", "grouped", "state_space"):
        raise ValueError(f"{option} requires dense, grouped or state_space linear algebra")
    count = len(model.problem_.names)
    if count > max_parameters:
        raise ValueError(
            f"{option} builds a dense Hessian over all {count} parameters; "
            f"the limit is {max_parameters}. Pass timing_report(..., max_parameters=N) "
            "to raise it. The cost is one Hessian-vector product per parameter plus a "
            "dense eigendecomposition, so time grows at least with the square of the "
            "parameter count: one 3,532-parameter grouped fit took about 10 minutes"
        )
    clusters = len(model.problem_.systems)
    if robust and clusters < MIN_CLUSTERS:
        raise ValueError(
            f"{option} uses the fitted runs as clusters and needs at least "
            f"{MIN_CLUSTERS}; this fit has {clusters}. Cut each recording into "
            "contiguous time blocks with multimodalsrm.split_runs(data, blocks) and refit"
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


def lag_standard_errors(
    model, reference, *, max_parameters=MAX_PARAMETERS, batch_size=32, robust=False
):
    """Laplace standard errors of lags, relative lags and lag differences.

    ``reference`` is the modality shown at zero. A fixed lag has zero
    variance; a relative lag uses ``var_m + var_ref - 2 cov``.

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
    affected. A quantity is NaN when a lag it uses rests on a bound
    (``lags_at_bounds``) or when it depends on an invariant direction
    (``not_identified``). ``positive_definite`` is False when the Hessian has
    a nonpositive eigenvalue on the complement: the fit is then not at a local
    optimum and the errors should not be used.

    ``robust=True`` adds ``robust_standard_error`` to every quantity: the
    cluster-robust sandwich with the fitted runs as clusters (see the module
    docstring), computed with the same pseudo-inverse over the same free
    parameters, and NaN wherever the Laplace error is. It includes the factor
    ``B / (B - 1)`` for ``clusters = B`` runs; ``few_clusters`` is True below
    ten. Fewer than three runs raise an error.
    """
    _check_supported(model, max_parameters, robust)
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
    if robust:
        began = time.perf_counter()
        scores = run_scores(problem, x)
        score_seconds = time.perf_counter() - began
        clusters = len(scores)
        # Row b is H^+ s_b; covariance is zero outside the free parameters.
        influence = (scores - scores.mean(axis=0)) @ covariance

    not_identified = []

    def entry(name, estimate, contrast, modalities):
        error = sandwich = float("nan")
        if np.linalg.norm(null_vectors.T @ contrast) > 1e-6:
            not_identified.append(name)
        elif not set(modalities) & set(at_bounds):
            variance = float(contrast @ covariance @ contrast)
            error = float(np.sqrt(variance)) if variance >= 0 else float("nan")
            if robust:
                squares = float(((influence @ contrast) ** 2).sum())
                sandwich = float(np.sqrt(clusters / (clusters - 1) * squares))
        result = dict(estimate=float(estimate), standard_error=error)
        if robust:
            result["robust_standard_error"] = sandwich
        return result

    extra = {}
    if robust:
        extra = dict(
            robust_label=ROBUST_LABEL,
            clusters=clusters,
            degrees_of_freedom=clusters - 1,
            few_clusters=clusters < FEW_CLUSTERS,
            score_seconds=score_seconds,
        )
    return dict(
        method="robust" if robust else "laplace",
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
        null_directions=nullity,
        null_direction_residual=residual,
        parameters=len(x),
        parameters_at_bounds=int((~free).sum()),
        hessian_seconds=seconds,
        **extra,
    )
