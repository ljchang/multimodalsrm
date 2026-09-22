"""Bounded stationary response estimation in physical/log coordinates."""

import copy

import numpy as np
from scipy.optimize import minimize

from .kernels import BatemanSCR, Gaussian


def group_kernels(kernels, responses):
    groups = {}
    for m, response in responses.items():
        local = [mods[m] for mods in kernels.values() if m in mods]
        if not local or response.pooling == "none":
            continue
        coords = {p: np.mean([k.to_coordinates()[p] for k in local]) for p in local[0].parameters}
        groups[m] = local[0].from_coordinates(coords)
    return groups


def kernel_penalty(kernels, responses):
    groups = group_kernels(kernels, responses)
    total = 0.0
    for m, response in responses.items():
        local = [mods[m] for mods in kernels.values() if m in mods]
        if not local:
            continue
        if response.pooling == "partial":
            center = groups[m].to_coordinates()
            total += (
                0.5
                * response.pooling_strength
                * sum(sum((k.to_coordinates()[p] - center[p]) ** 2 for p in center) for k in local)
                / len(local)
            )
        targets = local if response.pooling == "none" else [groups[m]]
        for k in targets:
            factor = 1 / len(targets)
            if response.lag_prior is not None:
                prior = response.lag_prior
                total += factor * 0.5 * ((k.parameters["lag"] - prior.mean) / prior.sd) ** 2
            if response.prior is not None:
                prior = response.prior
                ref = (
                    prior.reference.parameters
                    if hasattr(prior.reference, "parameters")
                    else prior.reference
                )
                for p, v in ref.items():
                    c = np.log(v) if p in k.positive_parameters else v
                    total += factor * 0.5 * prior.strength * (k.to_coordinates()[p] - c) ** 2
    return float(total)


def _kernel_penalty_gradient(kernels, responses):
    """Prior/pooling derivatives in the same log/physical coordinates as fitting."""
    gradients = {
        (s, m, p): 0.0 for s, mods in kernels.items() for m, k in mods.items() for p in k.parameters
    }
    for m, response in responses.items():
        subjects = [s for s in kernels if m in kernels[s]]
        if not subjects:
            continue
        coordinates = {s: kernels[s][m].to_coordinates() for s in subjects}
        center = {
            p: np.mean([c[p] for c in coordinates.values()]) for p in coordinates[subjects[0]]
        }
        n = len(subjects)
        if response.pooling == "partial":
            for s in subjects:
                for p in center:
                    gradients[s, m, p] += (
                        response.pooling_strength * (coordinates[s][p] - center[p]) / n
                    )
        for s in subjects:
            target = coordinates[s] if response.pooling == "none" else center
            if response.lag_prior is not None:
                prior = response.lag_prior
                gradients[s, m, "lag"] += (target["lag"] - prior.mean) / (n * prior.sd**2)
            if response.prior is not None:
                prior = response.prior
                ref = (
                    prior.reference.parameters
                    if hasattr(prior.reference, "parameters")
                    else prior.reference
                )
                for p, value in ref.items():
                    c = np.log(value) if p in kernels[s][m].positive_parameters else value
                    gradients[s, m, p] += prior.strength * (target[p] - c) / n
    return gradients


class KernelObjective:
    """Fixed-map/latent loss in component space, centered on the current fit.

    Each row retains W.T diag(c) W and W.T diag(c) residual. Candidate loss
    differences are quadratic in delta(H Z), avoiding repeated voxel predictions
    and cancellation from subtracting large uncentered sufficient statistics.
    Full reconstruction supplies the baseline (including any custom penalties)
    and remains the independent acceptance check. Other response families keep
    finite-difference optimization of this same accelerated loss.
    """

    def __init__(self, kernels, responses, prepared, latents, loadings, full_objective):
        self.responses, self.prepared = responses, prepared
        self.full_objective = full_objective
        self.baseline = float(full_objective(kernels)) - kernel_penalty(kernels, responses)
        self.gradient_method = (
            "complex_step"
            if any(
                responses[m].free_parameters and type(k) is BatemanSCR
                for mods in kernels.values()
                for m, k in mods.items()
            )
            else "analytic"
        )
        self.analytic_gradient = all(
            not responses[m].free_parameters or type(k) in (Gaussian, BatemanSCR)
            for mods in kernels.values()
            for m, k in mods.items()
        )
        self.statistics = []
        for b in prepared(kernels):
            w, z = loadings[b.subject][b.modality], latents[b.subject][b.run]
            y = b.H @ z
            linear = (b.coefficients * (b.values - y @ w.T)) @ w
            if np.all(b.coefficients == b.coefficients[:, :1]):
                gram = b.coefficients[:, :1, None] * (w.T @ w)[None, :, :]
            else:
                gram = np.einsum("tf,fi,fj->tij", b.coefficients, w, w, optimize=True)
            self.statistics.append((b, z, y, linear, gram))

    def _evaluate(self, kernels, gradient):
        value = self.baseline + kernel_penalty(kernels, self.responses)
        gradients = _kernel_penalty_gradient(kernels, self.responses) if gradient else None
        for b, z, y, linear, gram in self.statistics:
            kernel = kernels[b.subject][b.modality]
            free = self.responses[b.modality].free_parameters
            h, dh = self.prepared.operator(b, kernel, derivatives=bool(gradient and free))
            delta = h @ z - y
            gdelta = np.einsum("tij,tj->ti", gram, delta)
            value += np.sum(delta * (gdelta - 2 * linear))
            if gradient:
                for p in free:
                    derivative = 2 * np.sum((gdelta - linear) * (dh[p] @ z))
                    if p in kernel.positive_parameters:
                        derivative *= kernel.parameters[p]
                    gradients[b.subject, b.modality, p] += derivative
        return float(value), gradients

    def __call__(self, kernels):
        return self._evaluate(kernels, False)[0]

    def value_gradient(self, kernels):
        return self._evaluate(kernels, True)


def optimize_kernels(kernels, responses, objective, max_iter):
    """objective(candidate_kernels) includes all reconstruction and penalties."""
    keys = []
    initial = []
    bounds = []
    for m, response in responses.items():
        subjects = [s for s in kernels if m in kernels[s]]
        if not subjects:
            continue
        groups = [subjects] if response.pooling == "shared" else [[s] for s in subjects]
        for group in groups:
            for p in response.free_parameters:
                kernel = kernels[group[0]][m]
                lo, hi = response.parameter_bounds()[p]
                if p in kernel.positive_parameters:
                    lo, hi = np.log([lo, hi])
                keys.append((group, m, p))
                initial.append(kernel.to_coordinates()[p])
                bounds.append((lo, hi))
    if not keys:
        return kernels, dict(success=True, accepted=False, message="all kernels fixed")

    def unpack(vector):
        result = copy.deepcopy(kernels)
        coordinates = {
            (s, m): k.to_coordinates() for s, mods in kernels.items() for m, k in mods.items()
        }
        for value, (subjects, m, p) in zip(vector, keys):
            for s in subjects:
                coordinates[s, m][p] = value
        for (s, m), coords in coordinates.items():
            result[s][m] = kernels[s][m].from_coordinates(coords)
        # The group is the mean transformed coordinate, hence centered offsets.
        group_kernels(result, responses)  # validate coupled group shape constraints
        return result

    def fun(vector):
        try:
            value = objective(unpack(vector))
            return value if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError):
            return 1e100

    analytic = bool(getattr(objective, "analytic_gradient", False))

    def value_gradient(vector):
        try:
            value, gradients = objective.value_gradient(unpack(vector))
            gradient = np.array(
                [sum(gradients[s, m, p] for s in subjects) for subjects, m, p in keys]
            )
            if np.isfinite(value) and np.isfinite(gradient).all():
                return value, gradient
        except (ValueError, FloatingPointError):
            pass
        return 1e100, np.zeros(len(keys))

    before = fun(initial)
    result = minimize(
        value_gradient if analytic else fun,
        initial,
        jac=True if analytic else None,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )
    # Never accept an update based only on the contracted candidate loss.
    verified_after = result.fun
    if hasattr(objective, "full_objective"):
        before = float(objective.full_objective(kernels))
        verified_after = float(objective.full_objective(unpack(result.x)))
    accepted = bool(np.isfinite(verified_after) and verified_after <= before)
    diagnostic = dict(
        success=bool(result.success),
        accepted=accepted,
        message=str(result.message),
        objective_before=float(before),
        objective_after=float(verified_after),
        gradient_method=getattr(objective, "gradient_method", "analytic")
        if analytic
        else "finite_difference",
        function_evaluations=int(result.nfev),
        boundary_parameters=[
            str(keys[i])
            for i, (x, (lo, hi)) in enumerate(zip(result.x, bounds))
            if min(x - lo, hi - x) < 1e-7
        ],
    )
    return unpack(result.x) if accepted else kernels, diagnostic


def kernel_stationarity(kernels, responses, objective):
    """Bound-aware finite differences at the final simultaneous training state.

    Shared parameters move together; positive shapes use log coordinates.
    Gradients are objective per named coordinate, not unit-free comparisons
    across parameters. This diagnostic does not change the fitted kernels or
    certify a global optimum. Invalid perturbations are reported as unavailable.
    """
    rows = []
    before = float(objective(kernels))
    for modality, response in responses.items():
        subjects = [s for s in kernels if modality in kernels[s]]
        if not subjects:
            continue
        groups = [subjects] if response.pooling == "shared" else [[s] for s in subjects]
        for group in groups:
            kernel = kernels[group[0]][modality]
            for parameter in response.free_parameters:
                lo, hi = response.parameter_bounds()[parameter]
                positive = parameter in kernel.positive_parameters
                if positive:
                    lo, hi = np.log([lo, hi])
                x = kernel.to_coordinates()[parameter]
                step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x))
                left, right = max(lo, x - step), min(hi, x + step)

                def evaluate(value):
                    if value == x:
                        return before
                    candidate = copy.deepcopy(kernels)
                    try:
                        for subject in group:
                            original = kernels[subject][modality]
                            coordinates = original.to_coordinates()
                            coordinates[parameter] = value
                            candidate[subject][modality] = original.from_coordinates(coordinates)
                        group_kernels(candidate, responses)
                        return float(objective(candidate))
                    except (ValueError, FloatingPointError):
                        return np.nan

                gradient = None
                if np.isfinite(before) and right > left:
                    lower, upper = evaluate(left), evaluate(right)
                    if np.isfinite([lower, upper]).all():
                        value = (upper - lower) / (right - left)
                        if np.isfinite(value):
                            gradient = float(value)
                tolerance = 1e-8 * max(1.0, abs(x))
                at_lower = bool(x - lo <= tolerance)
                at_upper = bool(hi - x <= tolerance)
                projected = gradient
                if gradient is not None and (
                    (at_lower and gradient > 0) or (at_upper and gradient < 0)
                ):
                    projected = 0.0
                rows.append(
                    {
                        "subjects": list(group),
                        "modality": modality,
                        "parameter": parameter,
                        "coordinate": "log" if positive else "physical",
                        "gradient": gradient,
                        "projected_gradient": projected,
                        "at_lower_bound": at_lower,
                        "at_upper_bound": at_upper,
                    }
                )
    success = bool(np.isfinite(before) and all(r["gradient"] is not None for r in rows))
    return {
        "scope": "final_fixed_mapping_latent_kernel_coordinates",
        "method": "bounded_finite_difference",
        "success": success,
        "parameters": rows,
        "max_abs_projected_gradient": max((abs(r["projected_gradient"]) for r in rows), default=0.0)
        if success
        else None,
    }
