"""Descriptive fit diagnostics from saved records; no inference or gate changes."""

import math
from copy import deepcopy
from itertools import combinations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..kernels import BachSCR, DoubleGamma, Gamma, Gaussian, Identity

FAMILIES = {k.__name__: k for k in (Identity, Gaussian, Gamma, DoubleGamma, BachSCR)}


def structured_shape(kernel, reference_lag):
    """Positive-lobe peak/FWHM; coordinates relative to fixed reference lag."""
    grid = np.linspace(*kernel.support, 4097)
    values = kernel.evaluate(grid)
    i = int(np.argmax(values))
    peak = float(grid[i])
    if 0 < i < len(grid) - 1:
        peak = float(
            minimize_scalar(
                lambda t: -float(kernel.evaluate(t)),
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
            ).x
        )
    half = float(kernel.evaluate(peak)) / 2
    roots = []
    for a, b, va, vb in zip(grid[:-1], grid[1:], values[:-1] - half, values[1:] - half):
        if va * vb < 0:
            roots.append(brentq(lambda t: float(kernel.evaluate(t)) - half, a, b))
    left, right = [r for r in roots if r < peak], [r for r in roots if r > peak]
    return dict(
        shape_type="curve",
        peak_delay_seconds=peak - reference_lag,
        fwhm_seconds=min(right) - max(left) if left and right else None,
        support_relative_seconds=[float(v - reference_lag) for v in kernel.support],
        has_negative_lobe=bool(np.any(values < 0)),
        shape_metric_method="positive_lobe_numerical",
        reference_lag_seconds=reference_lag,
    )


def _finite_start(start):
    value, parameters = start.get("objective"), start.get("parameters")
    return (
        value is not None
        and np.isfinite(value)
        and parameters is not None
        and np.isfinite(parameters).all()
    )


def _parameters(names, values):
    """Reject ambiguous name/vector pairings before reporting any fitted shapes."""
    names = [tuple(n) for n in names]
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (len(names),)
        or not np.isfinite(values).all()
        or len(set(names)) != len(names)
        or any(not n or not isinstance(n[0], str) for n in names)
    ):
        raise ValueError("invalid or duplicate parameter names / parameter vector")
    return dict(zip(names, values.tolist()))


def summarize_fit(record):
    """Summarize one saved MAP/posterior fit without mutating its record.

    ``record`` contains ``configuration``, ``parameter_names``, ``map``,
    ``restarts`` and optional ``sampling`` diagnostics. Curves and metrics use
    the selected physical-density MAP, even for a posterior fit. Restart ranges
    include all finite starts, including inferior/nonstationary fits, and are
    descriptive optimization summaries, never uncertainty intervals.

    Gaussian peak delay is fitted lag minus fixed reference lag, or minus the
    across-modality mean lag when no fixed reference is available. FWHM is
    2*sqrt(2*log(2))*width. Identity is an impulse with undefined FWHM. Neither
    these summaries nor saved convergence gates establish timing recovery or
    calibration. Missing metadata yields explicitly incomplete shape summaries.
    """
    config = record.get("configuration") or {}
    best = record.get("map") or {}
    sampling = record.get("sampling") or {}
    starts = record.get("restarts") or []
    finite = [s for s in starts if _finite_start(s)]
    reference = config.get("reference_convention") or {}
    result = dict(
        availability="incomplete",
        map_passed=None
        if config.get("conditioning_mode") == "frozen"
        else best.get("meets_gradient_tolerance"),
        training_map_passed=best.get("meets_gradient_tolerance")
        if config.get("conditioning_mode") == "frozen"
        else None,
        conditioning_mode=config.get("conditioning_mode", "training"),
        diagnostic_scope=config.get("diagnostic_scope", "fitted_problem"),
        sampling_passed=sampling.get("passes"),
        inference=config.get("inference"),
        uncertainty=config.get("uncertainty"),
        calibration_established=False,
        absolute_physiological_timing_established=False,
        shape_estimate="selected_MAP",
        reference=deepcopy(reference),
        run_baseline=deepcopy(config.get("run_baseline")),
        objective=best.get("objective"),
        physical_projected_gradient=best.get("physical_projected_gradient"),
        gradient_tolerance=config.get("search", {}).get("physical_gradient_tolerance"),
        boundary_parameters=deepcopy(best.get("boundary_parameters", [])),
        linear_system_sizes=deepcopy(config.get("linear_system_sizes", {})),
        restarts=dict(
            total=len(starts),
            finite=len(finite),
            stationary=sum(bool(s.get("meets_gradient_tolerance")) for s in finite),
        ),
        filters=[],
        relative_delays=[],
        unavailable_reason=None,
    )
    if config.get("response_quadrature") is not None:
        result["response_quadrature"] = deepcopy(config["response_quadrature"])
    names = record.get("parameter_names")
    metadata = config.get("response_metadata")
    reference_lag = reference.get("reference_lag_seconds")
    centered = (
        reference.get("relative_lag_definition") == "modality_lag_minus_mean_lag"
        and reference.get("reference_modality") is None
    )
    if (
        names is None
        or best.get("parameters") is None
        or not metadata
        or (
            not centered
            and (reference_lag is None or reference.get("reference_modality") not in metadata)
        )
    ):
        result["unavailable_reason"] = (
            "missing MAP, parameter names, response metadata or reference"
        )
        return result
    if not centered and not math.isfinite(reference_lag):
        raise ValueError("reference lag must be finite")
    for response in metadata.values():
        family = response.get("family")
        if family not in FAMILIES:
            continue
        schema = set(FAMILIES[family]().parameters)
        free = response.get("free_parameters", [])
        if (
            set(response.get("parameters", {})) != schema
            or not set(free) <= schema
            or len(set(free)) != len(free)
        ):
            raise ValueError(f"{family} parameter schema disagrees with response metadata")
    parameters = _parameters(names, best["parameters"])
    restart_parameters = [_parameters(names, s["parameters"]) for s in finite]
    expected = {("filter", m, p) for m, r in metadata.items() for p in r.get("free_parameters", [])}
    found = {n for n in parameters if n[0] == "filter"}
    if found != expected:
        raise ValueError("filter parameter names disagree with response metadata")
    if centered:
        if reference_lag is not None:
            raise ValueError("centered lag convention cannot specify a fixed reference lag")
        reference_lag = float(
            np.mean(
                [
                    parameters.get(("filter", m, "lag"), r.get("parameters", {}).get("lag", 0.0))
                    for m, r in metadata.items()
                ]
            )
        )
        result["relative_lag_origin_seconds"] = reference_lag
    else:
        ref_response = metadata[reference["reference_modality"]]
        ref_family = ref_response.get("family")
        if ref_family not in FAMILIES:
            result["unavailable_reason"] = "unsupported reference family; timing unavailable"
            return result
        if (
            "lag" in ref_response.get("free_parameters", [])
            or ref_response["parameters"].get("lag", 0.0) != reference_lag
            or reference.get("reference_family", ref_family) != ref_family
        ):
            raise ValueError("reference must match the named response's fixed lag and family")
    boundaries = {tuple(n) for n in best.get("boundary_parameters", [])}
    for modality, response in metadata.items():
        initial = deepcopy(response.get("parameters", {}))
        fitted = dict(initial)
        estimated = list(response.get("free_parameters", []))
        ranges = {}
        for parameter in estimated:
            key = ("filter", modality, parameter)
            fitted[parameter] = parameters[key]
            values = [p[key] for p in restart_parameters]
            ranges[parameter] = (
                dict(min=min(values), max=max(values), n=len(values))
                if values
                else dict(min=None, max=None, n=0)
            )
        row = dict(
            modality=modality,
            family=response.get("family"),
            parameters=fitted,
            initial_parameters=initial,
            estimated_parameters=estimated,
            boundary_parameters=[p for p in estimated if ("filter", modality, p) in boundaries],
            bounds=deepcopy(response.get("bounds", {})),
            restart_ranges=ranges,
            shape_type="unsupported",
            peak_delay_seconds=None,
            fwhm_seconds=None,
            support_relative_seconds=None,
        )
        if row["family"] == "Gaussian":
            width, lag = fitted.get("width"), fitted.get("lag")
            if width is None or lag is None or not np.isfinite([width, lag]).all() or width <= 0:
                raise ValueError("Gaussian parameters require finite lag and positive finite width")
            peak = lag - reference_lag
            row.update(
                shape_type="curve",
                peak_delay_seconds=peak,
                fwhm_seconds=2 * math.sqrt(2 * math.log(2)) * width,
                support_relative_seconds=[peak - 6 * width, peak + 6 * width],
            )
        elif row["family"] == "Identity":
            row.update(
                shape_type="impulse",
                peak_delay_seconds=-reference_lag,
                support_relative_seconds=[-reference_lag, -reference_lag],
            )
        elif row["family"] in FAMILIES:
            row.update(structured_shape(FAMILIES[row["family"]](**fitted), reference_lag))
        result["filters"].append(row)
    for a, b in combinations(result["filters"], 2):
        if a["peak_delay_seconds"] is not None and b["peak_delay_seconds"] is not None:
            result["relative_delays"].append(
                dict(
                    from_modality=a["modality"],
                    to_modality=b["modality"],
                    delay_seconds=b["peak_delay_seconds"] - a["peak_delay_seconds"],
                )
            )
    if all(r["shape_type"] != "unsupported" for r in result["filters"]):
        result["availability"] = "complete"
    else:
        result["unavailable_reason"] = "unsupported filter family; no shape metrics inferred"
    return result
