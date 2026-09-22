"""Translate a bounded regularized fit into a GP MAP starting point only."""

import copy
import time
import warnings
from dataclasses import replace

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..estimator import MultimodalSRM
from ..kernels import Gaussian


def _latent_dt(problem):
    """Resolve the initializer grid from native clocks, with a 10,000-cell cap.

    This grid belongs only to R-MSRM; the GP keeps its native observations.
    """
    intervals = [problem.length_scale]
    for runs in problem.adapter._training_data.values():
        for mods in runs.values():
            for series in mods.values():
                times = series.times[series.mask.any(axis=1)]
                if len(times) > 1:
                    intervals.append(float(np.median(np.diff(times))))
    duration = sum(b - a for a, b in problem.adapter.domains_.values())
    return max(min(intervals) / 2, duration / 10_000)


def _translate(problem, fitted):
    # Shared paths occur once per participant; count each run once when
    # mapping their scale to the GP's unit stationary latent variance.
    paths = {}
    for runs in fitted._training_latent_arrays_.values():
        for run, values in runs.items():
            grid = fitted.run_grids_[run]
            lo, hi = fitted.run_domains_[run]
            paths.setdefault(run, values[(grid >= lo) & (grid <= hi)])
    scale = np.std(np.concatenate(list(paths.values())), axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    sign = 1.0
    if problem.features == 1:
        subject, modality, feature = problem.anchor
        if fitted.loadings_[subject][modality][feature, 0] < 0:
            sign = -1.0

    point = problem.initial.copy()
    for subject, modality in problem.groups:
        stats = fitted.preprocessing_[subject][modality]
        supplied = problem.adapter.preprocessing_[subject][modality]
        units = stats["scale"] / supplied["scale"]
        loading = fitted.loadings_[subject][modality] * units[:, None] * scale * sign
        offset = (stats["mean"] - supplied["mean"]) / supplied["scale"]
        for feature in range(len(offset)):
            key = subject, modality, feature
            point[problem.indices[("offset", *key)]] = offset[feature]
            if problem.features == 1:
                point[problem.indices[("loading", *key)]] = loading[feature, 0]
            else:
                for factor in range(problem.features):
                    point[problem.indices[("loading", *key, factor)]] = loading[feature, factor]

    # Use every run and only scored, observed feature entries. Multiplying
    # before masking preserves feature-specific units and avoids NaN leakage.
    sums = {group: [0.0, 0] for group in problem.groups}
    for block in fitted.observation_blocks_:
        group = block.subject, block.modality
        # The internal grid includes the last interpolation cell; public
        # SeriesResult masks that padded endpoint with NaN.
        latent = fitted._training_latent_arrays_[block.subject][block.run]
        prediction = (block.H @ latent) @ fitted.loadings_[block.subject][block.modality].T
        units = (
            fitted.preprocessing_[block.subject][block.modality]["scale"]
            / problem.adapter.preprocessing_[block.subject][block.modality]["scale"]
        )
        residual = ((block.values - prediction) * units)[block.mask & block.valid[:, None]]
        sums[group][0] += float(residual @ residual)
        sums[group][1] += residual.size
    for group, (squared, count) in sums.items():
        if not count:
            raise ValueError(f"R initialization has no observed residuals for {group}")
        point[problem.indices[("noise", *group)]] = squared / count
    for modality, kernel in fitted.group_kernels_.items():
        for name, value in kernel.parameters.items():
            index = problem.indices.get(("filter", modality, name))
            if index is not None:
                point[index] = value

    # Initializers need interior coordinates even when the R optimum is on a
    # response bound or its residual variance vanishes. GP priors are unchanged.
    for indices, prior in problem._prior_groups:
        point[indices] = np.clip(point[indices], prior.ppf(1e-4), prior.ppf(1 - 1e-4))
    if not np.isfinite(point).all():
        raise ValueError("R initialization produced nonfinite parameters")
    return point


def r_initial_point(problem, seed):
    """Return a full-training start and diagnostics, or a documented fallback.

    Conditional calibration and posterior updates retain their own target and
    initialization contracts. No fitted R estimator is retained on the GP.
    """
    if not getattr(problem, "_full_training_target", False) or hasattr(
        problem, "_posterior_update"
    ):
        return None, dict(method="data", r_init="skipped", reason="not a full training target")
    started = time.perf_counter()
    report = dict(method="r_msrm", r_init="used", latent_dt=_latent_dt(problem))
    try:
        responses = copy.deepcopy(problem.responses)
        fixed = []
        for modality, response in responses.items():
            if response.free_parameters and type(response.initial_kernel()) is not Gaussian:
                # R has analytic response gradients only for Gaussian kernels.
                # Avoid an expensive finite-difference fit inside a cheap seed;
                # the original GP still learns every requested parameter.
                responses[modality] = replace(response, estimate=False)
                fixed.append(modality)
        report["fixed_response_modalities"] = fixed
        fitted = MultimodalSRM(
            features=problem.features,
            latent_dt=report["latent_dt"],
            responses=responses,
            max_iter=40,
            kernel_max_iter=20,
            n_init=1,
            n_jobs=1,
            init="hybrid",
            random_state=seed,
        )
        # The bounded preliminary fit need not converge; its status is recorded
        # separately from the GP's unchanged convergence diagnostics.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="multimodalsrm")
            fitted.fit(problem.adapter._training_data)
        point = _translate(problem, fitted)
        report.update(iterations=int(fitted.n_iter_), converged=bool(fitted.converged_))
        # Reuse the compiled physical score needed by final MAP diagnostics.
        # Eager objective evaluation would separately compile many primitives.
        value, gradient = problem.value_gradient(point)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("R initialization has nonfinite GP objective or gradient")
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError) as exc:
        point = None
        report.update(method="data", r_init="fallback", reason=f"{type(exc).__name__}: {exc}")
        warnings.warn(
            f"R-MSRM initialization failed; using the data-based GP start ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    return point, report
