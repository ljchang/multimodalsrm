"""Bounded local Newton polishing using sequential physical-gradient probes."""

import time

import numpy as np
from scipy.optimize import OptimizeResult


def _variance_repair(
    problem, record, config, x, gradient, free, report, attempt, source_value=None
):
    """One bounded exact-profile proposal inside the current outer step."""
    from .fitting import _map_diagnostics
    from .noise_profile import variance_profile

    if not len(free):
        return False
    coordinate = free[np.argmax(np.abs(gradient[free]))]
    if problem.names[coordinate][0] != "noise":
        return False
    detail = dict(
        coordinate=list(problem.names[coordinate]),
        coordinate_index=int(coordinate),
        accepted=False,
        gradient_evaluations=0,
        change_evaluations=0,
    )
    attempt["variance_profile"] = detail
    if source_value is None:
        source_value = record["objective"]
    detail["source_objective"] = source_value
    try:
        profile = variance_profile(problem, x, coordinate)
        detail.update(profile.diagnostics)
        if not profile.available:
            detail["failure"] = profile.reason
            return False
        detail.update(profile_gradient=profile.gradient, profile_curvature=profile.curvature)
        if profile.curvature <= 0:
            raise FloatingPointError("nonpositive variance profile curvature")
        lo, hi = problem.bounds[coordinate]
        step = np.clip(
            -profile.gradient / profile.curvature,
            max(-0.25 * x[coordinate], 0.99 * (lo - x[coordinate])),
            min(0.25 * x[coordinate], 0.99 * (hi - x[coordinate])),
        )
        candidate_x = x.copy()
        candidate_x[coordinate] += step
        step = float(candidate_x[coordinate] - x[coordinate])
        if step == 0:
            raise FloatingPointError("no representable variance step")
        detail.update(
            step=step,
            stable_change=profile.change(step),
            cancellation_bound=profile.cancellation_bound(step),
            change_evaluations=1,
        )
        if not detail["stable_change"] < -detail["cancellation_bound"]:
            raise FloatingPointError("variance change is not negative beyond cancellation")
        report["gradient_evaluations"] += 1
        detail["gradient_evaluations"] += 1
        candidate = _map_diagnostics(
            problem,
            candidate_x,
            config,
            OptimizeResult(
                success=True,
                status=0,
                nit=1,
                message="Exact variance-profile numerical polishing",
            ),
        )
        detail.update(
            rounded_absolute_change=candidate["objective"] - source_value,
            objective=candidate["objective"],
            physical_projected_gradient=candidate["physical_projected_gradient"],
        )
        if candidate["physical_projected_gradient"] >= np.max(np.abs(gradient[free])):
            raise FloatingPointError("variance step did not lower fresh projected gradient")
        candidate["optimizer_success"] = candidate["meets_gradient_tolerance"]
        candidate["status"] = 0 if candidate["meets_gradient_tolerance"] else 1
        record.update(candidate)
        detail["accepted"] = attempt["accepted"] = True
        report["accepted_steps"] += 1
        return True
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        detail["failure"] = f"{type(exc).__name__}: {exc}"
        return False


def polish(problem, record, config, steps):
    """Improve stationarity without changing the density or gradient threshold.

    Only free coordinates enter the Hessian, capped by polish_max_parameters.
    Larger problems retain the current estimate and report the skipped dense
    step; the physical-gradient qualification threshold is unchanged.
    Central/one-sided gradient probes
    remain inside prior support; no dense autodiff Hessian is materialized.
    Positive noise variances use relative probe sizes: a unit-scaled probe
    can span a large fraction of a small variance and misestimate curvature.
    The symmetric Hessian is diagonally equilibrated before its eigenvalue
    safeguard, so stiff variance coordinates do not flatten valid curvature
    in other units. Steps and convergence checks remain in physical units.
    General steps lower the projected gradient without raising the computed
    objective. When only one free coordinate violates the gradient tolerance,
    a noise variance gets an exact-profile proposal before the general sweep.
    Multi-factor fits also try this when all violations are noise variances,
    rebuilding a single-coordinate profile within each existing outer step.
    If general polishing fails, a bounded single-variance profile
    can verify a density decrease below the resolution of the absolute total.
    That fallback retains the actual recomputed total, which may round upward,
    and must lower the freshly evaluated physical projected gradient. It uses
    the same outer step budget. Failures retain the previous parameters.
    """
    from .fitting import _map_diagnostics

    started = time.perf_counter()
    report = dict(steps=0, accepted_steps=0, gradient_evaluations=0, backtracks=0, attempts=[])
    bounds = np.asarray(problem.bounds)

    def evaluate(x):
        report["gradient_evaluations"] += 1
        value, gradient = problem.value_gradient(x)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite polishing probe")
        return value, gradient

    for _ in range(steps):
        if record["meets_gradient_tolerance"]:
            break
        attempt = dict(accepted=False)
        report["attempts"].append(attempt)
        report["steps"] += 1
        free = []
        try:
            x = np.asarray(record["parameters"])
            source_value, gradient = evaluate(x)
            tolerance = 1e-7 * np.maximum(1.0, np.abs(x))
            low = x - bounds[:, 0] <= tolerance
            high = bounds[:, 1] - x <= tolerance
            active = (low & (gradient > 0)) | (high & (gradient < 0))
            free = np.flatnonzero(~active)
            above = free[np.abs(gradient[free]) > config.physical_gradient_tolerance]
            noise_only = len(above) and all(problem.names[j][0] == "noise" for j in above)
            if noise_only and (len(above) == 1 or getattr(problem, "features", 1) > 1):
                attempt["variance_profile_priority"] = (
                    "only_coordinate_above_tolerance"
                    if len(above) == 1
                    else "only_noise_coordinates_above_tolerance"
                )
                if _variance_repair(
                    problem,
                    record,
                    config,
                    x,
                    gradient,
                    free,
                    report,
                    attempt,
                    source_value,
                ):
                    continue
            if len(free) > config.polish_max_parameters:
                skipped = dict(
                    skipped="dense_parameter_limit",
                    free_parameters=len(free),
                    parameter_limit=config.polish_max_parameters,
                    dense_hessian_bytes=len(free) ** 2 * np.dtype(float).itemsize,
                )
                attempt.update(skipped)
                report.update(skipped)
                break
            H = np.empty((len(free), len(free)))
            increments = []
            for column, j in enumerate(free):
                scale = max(1.0, abs(x[j]))
                if problem.names[j][0] == "noise" and x[j] > 0:
                    scale = x[j]
                h = np.cbrt(np.finfo(float).eps) * scale
                left, right = x[j] - bounds[j, 0], bounds[j, 1] - x[j]
                if min(left, right) > h:
                    plus, minus = x.copy(), x.copy()
                    plus[j] += h
                    minus[j] -= h
                    difference = (evaluate(plus)[1] - evaluate(minus)[1]) / (2 * h)
                else:
                    direction = 1.0 if right >= left else -1.0
                    h = min(h, max(left, right) * 0.25)
                    probe = x.copy()
                    probe[j] += direction * h
                    if probe[j] == x[j]:
                        raise FloatingPointError("no representable interior polishing probe")
                    difference = (evaluate(probe)[1] - gradient) / (direction * h)
                H[:, column] = difference[free]
                increments.append(float(h))
            H = (H + H.T) * 0.5
            scales = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-8))
            values, vectors = np.linalg.eigh(scales[:, None] * H * scales[None, :])
            floor = max(1e-8, np.max(np.abs(values)) * 1e-10)
            direction = np.zeros_like(x)
            direction[free] = -scales * (
                vectors @ ((vectors.T @ (scales * gradient[free])) / np.maximum(values, floor))
            )
            direction[(low & (direction < 0)) | (high & (direction > 0))] = 0.0
            direction /= max(1.0, np.max(np.abs(direction) / (0.25 * np.maximum(1.0, np.abs(x)))))
            slope = float(gradient @ direction)
            if not np.isfinite(slope) or slope >= 0:
                raise FloatingPointError("no descent polishing direction")
            alpha = 1.0
            for j, d in enumerate(direction):
                if d > 0:
                    alpha = min(alpha, 0.99 * (bounds[j, 1] - x[j]) / d)
                elif d < 0:
                    alpha = min(alpha, 0.99 * (bounds[j, 0] - x[j]) / d)
            attempt.update(
                free_coordinates=free.tolist(),
                finite_difference_steps=increments,
                curvature_coordinates="diagonally_equilibrated",
                hessian_coordinate_scales=scales.tolist(),
                minimum_hessian_eigenvalue=float(values[0]),
                eigenvalue_floor=float(floor),
                directional_derivative=slope,
            )
            for backtrack in range(12):
                candidate_x = x + alpha * direction
                report["gradient_evaluations"] += 1
                candidate = _map_diagnostics(
                    problem,
                    candidate_x,
                    config,
                    OptimizeResult(
                        success=True,
                        status=0,
                        nit=1,
                        message="Safeguarded Newton polishing",
                    ),
                )
                if (
                    candidate["objective"] <= record["objective"]
                    and candidate["physical_projected_gradient"]
                    < record["physical_projected_gradient"]
                ):
                    candidate["optimizer_success"] = candidate["meets_gradient_tolerance"]
                    candidate["status"] = 0 if candidate["meets_gradient_tolerance"] else 1
                    record.update(candidate)
                    attempt.update(
                        accepted=True,
                        step_scale=float(alpha),
                        objective=candidate["objective"],
                        physical_projected_gradient=candidate["physical_projected_gradient"],
                    )
                    report["accepted_steps"] += 1
                    break
                report["backtracks"] += 1
                alpha *= 0.5
            if not attempt["accepted"]:
                attempt["general_failure"] = (
                    "no nonworsening step with a smaller projected gradient"
                )
                if "variance_profile" not in attempt and _variance_repair(
                    problem,
                    record,
                    config,
                    x,
                    gradient,
                    free,
                    report,
                    attempt,
                    source_value,
                ):
                    continue
                attempt["failure"] = attempt["general_failure"]
                break
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            attempt["failure"] = f"{type(exc).__name__}: {exc}"
            if (
                len(free)
                and "variance_profile" not in attempt
                and _variance_repair(
                    problem,
                    record,
                    config,
                    x,
                    gradient,
                    free,
                    report,
                    attempt,
                    source_value,
                )
            ):
                attempt["general_failure"] = attempt.pop("failure")
                continue
            break
    report["elapsed_seconds"] = time.perf_counter() - started
    return report
