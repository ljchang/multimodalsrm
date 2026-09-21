"""Optional staged diagonal conditioning of the unchanged physical MAP density."""

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def validate(problem):
    """Limit this experimental solver to the validated full multifactor model."""
    if getattr(problem, "features", 1) <= 1:
        raise ValueError("diagonal conditioning requires multiple factors")
    if (
        getattr(problem, "linear_algebra", None) not in ("dense", "grouped")
        or getattr(problem, "noise_timescales", None)
        or getattr(problem, "run_baseline_sd", None)
        or hasattr(problem, "joint")
    ):
        raise ValueError(
            "diagonal conditioning requires a full dense/grouped model with "
            "independent observation noise; conditional calibration is unsupported"
        )


class InformationScaling:
    """Positive conditional-information approximation, not a posterior Hessian.

    Counts use retained native observations. Loading scales use the current
    response's marginal prior variance; offset/noise scales use their Gaussian
    observation information. The few filter coordinates use local differences
    of the physical gradient. These approximations only change solver units.
    """

    def __init__(self, problem):
        validate(problem)
        self.problem = problem
        self.feature_counts = np.zeros(len(problem.keys))
        self.group_counts = np.zeros(len(problem.groups))
        for run in problem.systems:
            ki, gi, _ = problem._packed[run]
            self.feature_counts += np.bincount(ki, minlength=len(problem.keys))
            self.group_counts += np.bincount(gi, minlength=len(problem.groups))
        self.keys = {key: i for i, key in enumerate(problem.keys)}
        self.groups = {group: i for i, group in enumerate(problem.groups)}
        self.modalities = {modality: i for i, modality in enumerate(problem.modalities)}
        self.filters = [i for i, name in enumerate(problem.names) if name[0] == "filter"]

    def at(self, x, evaluate=None):
        p = self.problem
        if evaluate is None:
            evaluate = p.value_gradient
        noise = np.asarray(p.arrays(x)[2])
        modalities = np.arange(len(p.modalities))
        zeros = np.zeros(len(modalities))
        variance = np.diag(
            np.asarray(p.temporal_covariance(x, zeros, modalities, zeros, modalities))
        )
        precision = np.ones(len(x))
        for j, name in enumerate(p.names):
            prior = p.parameter_priors[j]
            prior_precision = 1 / prior.scale**2 if prior.family == "normal" else 0
            if name[0] in ("loading", "offset"):
                count = self.feature_counts[self.keys[tuple(name[1:4])]]
                gi = self.groups[tuple(name[1:3])]
                latent_variance = variance[self.modalities[name[2]]] if name[0] == "loading" else 1
                precision[j] = count * latent_variance / noise[gi] + prior_precision
            elif name[0] == "noise":
                gi = self.groups[tuple(name[1:])]
                log_precision = 1 / prior.scale**2 if prior.family == "lognormal" else 0
                precision[j] = (self.group_counts[gi] / 2 + log_precision) / noise[
                    gi
                ] ** 2 + prior_precision
        for j in self.filters:
            h = np.cbrt(np.finfo(float).eps) * max(1, abs(x[j]))
            low, high = p.bounds[j]
            plus, minus = x.copy(), x.copy()
            plus[j], minus[j] = min(x[j] + h, high), max(x[j] - h, low)
            if p.parameter_priors[j].family == "lognormal" and minus[j] <= 0:
                minus[j] = x[j] / 2
            if not plus[j] > minus[j]:
                raise FloatingPointError("no representable conditioning probe")
            curvature = (evaluate(plus)[1][j] - evaluate(minus)[1][j]) / (plus[j] - minus[j])
            precision[j] = max(abs(curvature), 1 / p.parameter_priors[j].scale ** 2)
        if not np.isfinite(precision).all() or np.any(precision <= 0):
            raise FloatingPointError("nonpositive or nonfinite conditioning precision")
        scales = np.clip(1 / np.sqrt(precision), 1e-8, 1e4)
        return scales, dict(
            gradient_evaluations=2 * len(self.filters),
            minimum_scale=float(scales.min()),
            maximum_scale=float(scales.max()),
        )


def _projected_gradient(x, gradient, bounds):
    eps = 1e-7 * np.maximum(1, np.abs(x))
    active = ((x - bounds[:, 0] <= eps) & (gradient > 0)) | (
        (bounds[:, 1] - x <= eps) & (gradient < 0)
    )
    return float(np.max(np.abs(np.where(active, 0, gradient))))


def minimize_conditioned(
    problem, x, config, *, maxiter, callback=None, check_cancelled=lambda: None
):
    """Restart bounded L-BFGS after <=200 iterations with updated solver units.

    There are at most 30 stages, sharing one iteration budget. Retain only finite
    nonworsening physical points, reject open-support boundary trials, and check
    physical qualification separately from the scaled optimizer's termination.
    No Jacobian, loss weighting, regularization, or covariance change is made.
    """
    scaling = InformationScaling(problem)
    bounds = np.asarray(problem.bounds)
    positive = np.array(
        [
            p.family == "lognormal" or n[0] == "noise"
            for n, p in zip(problem.names, problem.parameter_priors)
        ]
    )
    x = np.array(x, dtype=float, copy=True)
    calls = 0
    iterations = 0
    evaluations = 0
    stages = []

    def physical_vg(point):
        nonlocal calls
        check_cancelled()
        calls += 1
        return problem.value_gradient(point)

    value, gradient = physical_vg(x)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("conditioning requires a finite initial point")
    best = [value, x.copy(), gradient.copy()]
    qualified = _projected_gradient(x, gradient, bounds) <= config.physical_gradient_tolerance
    result = OptimizeResult(success=qualified, status=0, message="Initial point checked")
    for stage in range(30):
        if qualified or iterations >= maxiter:
            break
        origin = best[1].copy()
        before = best[0]
        try:
            scales, scale_report = scaling.at(origin, physical_vg)
        except (ArithmeticError, np.linalg.LinAlgError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            stages.append(
                dict(
                    stage=stage,
                    iterations=0,
                    failure=failure,
                    objective=best[0],
                    optimizer_success=False,
                )
            )
            result = OptimizeResult(
                success=False,
                status=2,
                message=f"Conditioning refresh failed: {failure}",
            )
            break
        boxes = (bounds - origin[:, None]) / scales[:, None]

        def physical(z):
            # Remove only roundoff when reconstructing an attained box bound.
            return np.clip(origin + scales * z, bounds[:, 0], bounds[:, 1])

        def evaluate(z):
            point = physical(z)
            check_cancelled()
            if not np.isfinite(point).all() or np.any(point[positive] <= 0):
                return np.inf, np.zeros_like(z)
            v, g = physical_vg(point)
            if not np.isfinite(v) or not np.isfinite(g).all():
                return np.inf, np.zeros_like(z)
            if v < best[0] or (
                v == best[0]
                and _projected_gradient(point, g, bounds)
                < _projected_gradient(best[1], best[2], bounds)
            ):
                best[:] = [v, point.copy(), g.copy()]
            return v, g * scales

        def accepted(intermediate_result):
            nonlocal iterations
            iterations += 1
            if callback is not None:
                callback(
                    OptimizeResult(x=physical(intermediate_result.x), fun=intermediate_result.fun)
                )

        result = minimize(
            evaluate,
            np.zeros_like(origin),
            jac=True,
            method="L-BFGS-B",
            bounds=boxes,
            callback=accepted,
            options=dict(
                maxiter=min(200, maxiter - iterations),
                maxcor=50,
                maxls=40,
                ftol=min(config.ftol, 1e-15),
                gtol=min(config.gtol, 1e-8),
            ),
        )
        evaluations += int(result.nfev)
        projected = _projected_gradient(best[1], best[2], bounds)
        qualified = projected <= config.physical_gradient_tolerance
        stages.append(
            dict(
                stage=stage,
                iterations=int(result.nit),
                objective=best[0],
                physical_projected_gradient=projected,
                message=str(result.message),
                optimizer_success=bool(result.success),
                **scale_report,
            )
        )
        if result.nit == 0 or (best[0] >= before and np.array_equal(best[1], origin)):
            break
    if not qualified and (iterations >= maxiter or len(stages) == 30):
        result = OptimizeResult(
            success=False, status=1, message="Conditioned search budget exhausted"
        )
    result.x, result.fun, result.jac = best[1], best[0], best[2]
    result.nit, result.nfev = iterations, evaluations
    report = dict(
        method="diagonal",
        coordinates="affine_physical",
        stages=stages,
        gradient_evaluations=calls,
        max_stages=30,
        stage_maxiter=200,
        ftol=min(config.ftol, 1e-15),
        gtol=min(config.gtol, 1e-8),
    )
    return result, report
