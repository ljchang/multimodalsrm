"""Add mappings in frozen fitted coordinates, with fixed donor trajectories."""

import copy
from dataclasses import replace

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .data import TimeSeries, normalize_data
from .grouping import complete_affinity, latent_groups
from .inference import component_support, condition, filter_data, require_support
from .kernel_optimization import kernel_penalty, optimize_kernels
from .objective import (
    _validate_loading_penalty_scaling,
    complete_objective,
    fit_preprocessing,
    make_blocks,
    solve_latents,
    solve_loadings,
)


def _extend_graph(model, new_subjects, affinity):
    graph = copy.deepcopy(model.affinity_)
    if model.latent_pooling not in ("neighborhood", "components"):
        if affinity is not None:
            raise ValueError("affinity is only valid for graph pooling calibration")
        if model.latent_pooling == "shared":
            return complete_affinity(model.subjects_ + new_subjects)
        return None
    for s in new_subjects:
        graph[s] = {}
    if affinity is not None:
        if not isinstance(affinity, dict):
            raise ValueError("calibration affinity must name new-to-existing edges")
        for s, row in affinity.items():
            if s not in graph or not isinstance(row, dict):
                raise ValueError("affinity contains unknown subject names")
            for t, value in row.items():
                if t not in graph or not np.isfinite(value) or value < 0 or s == t:
                    raise ValueError(
                        "affinity edges must name distinct subjects and be nonnegative"
                    )
                if s in model.subjects_ and t in model.subjects_:
                    if value != model.affinity_[s].get(t, 0):
                        raise ValueError("existing graph edges are frozen")
                elif s in new_subjects and t in new_subjects:
                    raise ValueError("calibration requires new-to-existing edges")
                else:
                    if t in affinity and s in affinity[t] and affinity[t][s] != value:
                        raise ValueError(
                            "affinity must be symmetric when both directions are supplied"
                        )
                    graph[s][t] = graph[t][s] = float(value)
    if model.latent_pooling == "components":
        old_groups = latent_groups(model.subjects_, "components", model.affinity_)
        for group in latent_groups(model.subjects_ + new_subjects, "components", graph):
            anchors = [g for g in old_groups if set(g).intersection(group)]
            if len(anchors) > 1:
                raise ValueError("calibration graph cannot merge existing components")
            if set(group).intersection(new_subjects) and len(anchors) != 1:
                raise ValueError("each new subject must connect to exactly one existing component")
    return graph


def calibrate(model, data, *, reference=None, affinity=None, modality_weights=None):
    """Return an independent model with new maps; existing maps act only as donors.

    ``modality_weights`` may replace the complete normalized source-weight policy
    on the returned model, allowing positive-weight calibration of an omitted map.
    Frozen fitted arrays and training results remain unchanged.
    """
    check_is_fitted(model, "loadings_")
    if reference not in (None, "training"):
        raise ValueError("reference must be None (explicit aligned donors) or 'training'")
    data = normalize_data(data)
    unknown = {m for runs in data.values() for mods in runs.values() for m in mods} - set(
        model.responses_
    )
    if unknown:
        raise ValueError(f"new modalities require configured responses: {sorted(unknown)}")
    result = copy.deepcopy(model)
    result.loading_penalty_scaling = _validate_loading_penalty_scaling(
        getattr(model, "loading_penalty_scaling", "pair")
    )
    result.__dict__.pop("_legacy_loading_penalty_scaling_default", None)
    result.configuration_["loading_penalty_scaling"] = result.loading_penalty_scaling
    if modality_weights is not None:
        weights = dict(modality_weights)
        if (
            set(weights) != set(model.responses_)
            or not all(np.isfinite(v) and v >= 0 for v in weights.values())
            or not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                "modality_weights must cover responses, be nonnegative, and sum to one"
            )
        result.modality_weights_ = weights
        result.modality_weights = copy.deepcopy(weights)
    new = filter_data(
        data,
        lambda s, r, m: (
            m not in model.loadings_.get(s, {}) and result.modality_weights_.get(m, 0) > 0
        ),
    )
    if not new:
        raise ValueError(
            "calibration requires a new positive-weight mapping; existing mappings cannot be replaced"
        )
    unknown = {m for runs in new.values() for mods in runs.values() for m in mods} - set(
        model.responses_
    )
    if unknown:
        raise ValueError("new modalities require configured responses")
    new_subjects = [s for s in new if s not in model.subjects_]
    result.subjects_ = model.subjects_ + new_subjects
    result.affinity_ = _extend_graph(model, new_subjects, affinity)
    runs = list(dict.fromkeys(r for sruns in new.values() for r in sruns))
    if reference == "training":
        if not set(runs) <= set(model.run_grids_):
            raise ValueError("training reference requires matching stimulus run names")
        grids = {r: model.run_grids_[r].copy() for r in runs}
        domains = {r: model.run_domains_[r] for r in runs}
        fixed = {
            s: {r: values[r].copy() for r in runs if r in values}
            for s, values in model._training_latent_arrays_.items()
        }
        # Previously calibrated participants need not have stored training runs.
        # Recover their conditional reference from actual stored anchors once.
        population_reference = (
            {
                r: np.mean([values[r] for values in fixed.values() if r in values], axis=0)
                for r in runs
            }
            if model.latent_pooling == "population"
            else None
        )
        fixed, _ = solve_latents(
            [],
            grids,
            model.subjects_,
            {},
            model.features,
            model.latent_pooling,
            model.latent_strength,
            model.latent_ridge,
            model.temporal_strength,
            model.affinity_,
            fixed_latents=fixed,
            population_reference=population_reference,
        )
        anchor_blocks = [b for b in model.observation_blocks_ if b.run in runs]
    else:
        donors = filter_data(
            data,
            lambda s, r, m: (
                m in model.loadings_.get(s, {}) and model.modality_weights_.get(m, 0) > 0
            ),
        )
        if not donors:
            raise ValueError(
                "calibration needs aligned fitted donor observations or a training reference anchor"
            )
        fixed, grids, domains, anchor_blocks, _ = condition(model, donors)
        if not set(runs) <= set(grids):
            raise ValueError("calibration runs must align with usable donor run names")
        grids = {r: grids[r] for r in runs}
        domains = {r: domains[r] for r in runs}
        fixed = {s: {r: z[r].copy() for r in runs} for s, z in fixed.items()}
    if reference != "training":
        population_reference = (
            {r: np.mean([fixed[s][r] for s in model.subjects_], axis=0) for r in runs}
            if model.latent_pooling == "population"
            else None
        )
    for s, sruns in new.items():
        for r in sruns:
            require_support(result, s, r, anchor_blocks)
    if result.latent_pooling == "components":
        # Keep numerical grids and frozen trajectories global, but learn each
        # new map only from its component's raw reference time bounds. Mask
        # before preprocessing so excluded values cannot alter means/scales.
        supported = {}
        groups = latent_groups(result.subjects_, "components", result.affinity_)
        for s, sruns in new.items():
            group = next(g for g in groups if s in g)
            for r, mods in sruns.items():
                if reference == "training":
                    # Raw training observations are not retained. Stored output
                    # domains precede response trimming, unlike block.valid.
                    bounds = next(
                        model.training_latents_[member][r].metadata["domain"]
                        for member in group
                        if r in model.training_latents_.get(member, {})
                    )
                else:
                    # Use the donor source policy, even when calibration changes
                    # the returned model's weights to enable a missing mapping.
                    support_model = copy.copy(result)
                    support_model.modality_weights_ = model.modality_weights_
                    _, bounds = component_support(
                        support_model, donors, anchor_blocks, s, r, domains[r]
                    )
                local = {}
                for m, ts in mods.items():
                    lo, hi = result.responses_[m].support_envelope()
                    valid = (ts.times - hi >= bounds[0] - 1e-10) & (
                        ts.times - lo <= bounds[1] + 1e-10
                    )
                    mask = ts.mask & valid[:, None]
                    if mask.any():
                        local[m] = TimeSeries(ts.values, ts.times, mask)
                if not local:
                    raise ValueError(
                        f"no usable observations overlapping reference support in {s}/{r}"
                    )
                supported.setdefault(s, {})[r] = local
        requested = {(s, m) for s, sruns in new.items() for mods in sruns.values() for m in mods}
        retained = {
            (s, m) for s, sruns in supported.items() for mods in sruns.values() for m in mods
        }
        if requested != retained:
            raise ValueError(
                "each new mapping needs usable observations overlapping the reference support"
            )
        new = supported
    stats = fit_preprocessing(new, result.modality_weights_)
    kernels = {}
    for s, mods in stats.items():
        result.preprocessing_.setdefault(s, {}).update(mods)
        kernels[s] = {}
        for m in mods:
            response = model.responses_[m]
            if response.pooling in ("shared", "partial") and m in model.group_kernels_:
                kernels[s][m] = copy.deepcopy(model.group_kernels_[m])
            else:
                kernels[s][m] = response.initial_kernel()

    def blocks_for(k):
        return make_blocks(
            new,
            grids,
            domains,
            result.preprocessing_,
            k,
            result.responses_,
            result.modality_weights_,
            result.gap_threshold,
        )

    blocks = blocks_for(kernels)
    used = {(b.subject, b.modality) for b in blocks}
    if used != {(s, m) for s, mods in kernels.items() for m in mods}:
        raise ValueError(
            "each new mapping needs usable observations overlapping the reference support"
        )
    z = copy.deepcopy(fixed)
    for s in new_subjects:
        if result.latent_pooling == "components":
            group = next(
                g for g in latent_groups(result.subjects_, "components", result.affinity_) if s in g
            )
            donor = next(d for d in group if d in fixed)
            z[s] = {r: fixed[donor][r].copy() for r in runs}
            fixed[s] = copy.deepcopy(z[s])
        else:
            z[s] = {r: np.mean([fixed[d][r] for d in model.subjects_], axis=0) for r in runs}
    # Optimize only new effective kernels. Shared kernels and all group priors
    # remain fixed; partial penalties center on the established group coordinates.
    optimization_responses = {
        m: replace(
            response,
            pooling="none",
            estimate=response.estimate and response.pooling != "shared",
        )
        for m, response in result.responses_.items()
    }

    def penalty(k):
        independent = {m: r for m, r in result.responses_.items() if r.pooling == "none"}
        total = kernel_penalty(k, independent)
        for m, response in result.responses_.items():
            if response.pooling == "partial":
                center = model.group_kernels_.get(m, response.initial_kernel()).to_coordinates()
                local = [mods[m] for mods in k.values() if m in mods]
                if local:
                    total += (
                        0.5
                        * response.pooling_strength
                        * sum(
                            sum((kernel.to_coordinates()[p] - v) ** 2 for p, v in center.items())
                            for kernel in local
                        )
                        / len(local)
                    )
        return total

    def objective(z, w, k, b):
        return complete_objective(
            b,
            grids,
            z,
            w,
            result.latent_pooling,
            result.latent_strength,
            result.latent_ridge,
            result.temporal_strength,
            result.loading_ridge,
            result.affinity_,
            penalty(k),
            population_reference=population_reference,
            loading_penalty_scaling=result.loading_penalty_scaling,
        )

    W = solve_loadings(blocks, z, result.loading_ridge, result.loading_penalty_scaling)
    history = [objective(z, W, kernels, blocks)]
    diagnostics = []
    converged = False
    for iteration in range(result.max_iter):
        previous = history[-1]
        W = solve_loadings(blocks, z, result.loading_ridge, result.loading_penalty_scaling)
        kernels, kdiag = optimize_kernels(
            kernels,
            optimization_responses,
            lambda k: objective(z, W, k, blocks_for(k)),
            result.kernel_max_iter,
        )
        blocks = blocks_for(kernels)
        z, solver = solve_latents(
            blocks,
            grids,
            result.subjects_,
            W,
            result.features,
            result.latent_pooling,
            result.latent_strength,
            result.latent_ridge,
            result.temporal_strength,
            result.affinity_,
            fixed_latents=fixed,
            population_reference=population_reference,
        )
        current = objective(z, W, kernels, blocks)
        if not np.isfinite(current) or current > previous + 1e-9:
            raise RuntimeError("calibration objective increased or became nonfinite")
        history.append(current)
        diagnostics.append({"latent_solver": solver, "kernel_optimizer": kdiag})
        if abs(previous - current) <= result.tol * max(1.0, abs(previous)) and kdiag["success"]:
            converged = True
            break
    for s, mods in W.items():
        result.loadings_.setdefault(s, {}).update(mods)
        result.subject_kernels_.setdefault(s, {}).update(kernels[s])
    for s, mods in kernels.items():
        for m in mods:
            if m not in result.group_kernels_ and result.responses_[m].pooling != "none":
                result.group_kernels_[m] = result.responses_[m].initial_kernel()
    result._refresh_latent_groups()
    result.calibration_population_reference_ = copy.deepcopy(population_reference)
    result.calibration_reference_latents_ = copy.deepcopy(fixed)
    result.calibration_diagnostics_ = {
        "converged": converged,
        "objective_history": history,
        "iterations": diagnostics,
        "reference": reference or "donors",
        "run_domains": domains,
        "new_mappings": sorted(used),
    }
    result.configuration_["calibration_source_weights"] = copy.deepcopy(result.modality_weights_)
    if not converged:
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        warnings.warn(
            "calibration did not converge; returning finite conditional estimate",
            ConvergenceWarning,
            stacklevel=2,
        )
    return result
