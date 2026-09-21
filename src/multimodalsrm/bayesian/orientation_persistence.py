"""Validate recorded orientation evidence without rerunning MCMC or ArviZ."""

import numpy as np

from .blocks import ParameterSubspace
from .fitting import _orientation_context
from .orientation_diagnostics import (
    GENERAL_K_PROBE_SCHEMA,
    ORIENTATION_LIMITS,
    orientation_quantity_names,
    orientation_quantity_values,
)


def validate_orientation_record(model):
    d = model.sampling_diagnostics_
    mode = model.sampler_config_.orientation_refresh
    fields = {"raw_passes", "orientation_refresh", "orientation"}
    present = fields.intersection(d)
    legacy = "orientation_refresh" not in model.configuration_.get("sampler", {})
    k = model.problem_.features
    marker = d.get("orientation_probe_schema")
    if not present and mode == "none" and legacy and "orientation_probe_schema" not in d:
        return  # Legacy archives retain their original diagnostic semantics.
    if present != fields or d["orientation_refresh"] != mode or type(d["raw_passes"]) is not bool:
        raise ValueError("posterior orientation sampler identity is inconsistent")
    rows = d["parameters"]
    expected_raw = bool(
        d["chains"] >= 2
        and all(np.isfinite([r[k] for k in ("r_hat", "ess_bulk", "ess_tail")]).all() for r in rows)
        and np.isfinite(d["bfmi"]).all()
        and d["divergences"] == 0
        and d["max_rank_rhat"] <= 1.01
        and d["min_bulk_ess"] >= 400
        and d["min_tail_ess"] >= 200
        and d["min_bfmi"] >= 0.2
        and d["tree_depth_saturations"] == 0
    )
    if d["raw_passes"] != expected_raw:
        raise ValueError("posterior orientation raw gate contradicts diagnostics")
    o = d["orientation"]
    # Historical K>2 archives explicitly declined assessment before general-K
    # support. Preserve that exact record and its old gate; never qualify it.
    old_record = dict(
        applicable=False,
        status="not_assessed",
        passes=None,
        reason="Haar refresh requires exactly two factors",
    )
    if (
        k > 2
        and "orientation_probe_schema" not in d
        and mode == "none"
        and o == old_record
        and d["passes"] == d["raw_passes"]
    ):
        return
    if k > 2 and marker != GENERAL_K_PROBE_SCHEMA:
        raise ValueError("posterior orientation probe schema is missing or invalid")
    if k == 2 and "orientation_probe_schema" in d:
        raise ValueError("posterior orientation probe schema differs from factor count")
    space = ParameterSubspace(model.problem_, model.map_parameters_, blocks=model.sample_blocks)
    context = _orientation_context(space, getattr(model, "_factor_anchor_keys_", None))
    if not isinstance(o, dict) or o.get("applicable") is not context["applicable"]:
        raise ValueError("posterior orientation applicability differs from target")
    if not context["applicable"]:
        if (
            mode != "none"
            or o.get("status") != "not_assessed"
            or o.get("passes") is not None
            or d["passes"] != d["raw_passes"]
        ):
            raise ValueError("posterior orientation gate is invalid for this target")
        return
    if k > 2 and o.get("probe_schema") != GENERAL_K_PROBE_SCHEMA:
        raise ValueError("posterior orientation coordinate schema is missing or invalid")
    for name in ("anchor_indices", "anchor_keys", "anchor_source"):
        if o.get(name) != context[name]:
            raise ValueError("posterior orientation anchors differ from fitted convention")
    if o.get("limits") != ORIENTATION_LIMITS:
        raise ValueError("posterior orientation diagnostic limits differ")
    x = model.parameter_draws_
    w = x[..., : k * len(model.problem_.keys)].reshape(*x.shape[:2], -1, k)
    anchor = w[..., context["anchor_indices"], :]
    sv = np.linalg.svd(anchor, compute_uv=False)
    valid = sv[..., -1] > ORIENTATION_LIMITS["rank_tolerance"] * sv[..., 0]
    chains, draws = x.shape[:2]
    expected_counts = dict(
        chains=chains,
        draws_per_chain=draws,
        total_draws=chains * draws,
        retained_draws=chains * draws,
        valid_anchor_draws=int(valid.sum()),
        singular_anchor_draws=int((~valid).sum()),
    )
    if o.get("counts") != expected_counts:
        raise ValueError("posterior orientation counts differ from retained draws")
    status = (
        "singular_anchor"
        if not valid.all()
        else "insufficient_chains_or_draws"
        if chains < 2 or draws < 4
        else "computed"
    )
    if o.get("status") != status:
        raise ValueError("posterior orientation status differs from retained draws")
    names = orientation_quantity_names(k)
    records = o.get("per_quantity")
    if not isinstance(records, list) or len(records) != len(names):
        raise ValueError("posterior orientation coordinate records are invalid")
    quantities = None
    if status == "computed":
        quantities = orientation_quantity_values(anchor)
    for index, (record, name) in enumerate(zip(records, names)):
        if not isinstance(record, dict) or record.get("name") != name:
            raise ValueError("posterior orientation coordinate order differs")
        if quantities is None:
            allowed = {status}
        elif not np.any(np.ptp(quantities[..., index], axis=1)):
            expected_status = (
                "all_draws_constant"
                if np.ptp(quantities[..., index]) == 0
                else "within_chain_constants_disagree"
            )
            if record.get("status") != expected_status:
                raise ValueError(
                    "posterior orientation constant traces require undefined diagnostics"
                )
            allowed = {expected_status}
        else:
            allowed = {"computed", "undefined_diagnostic"}
        if record.get("status") not in allowed:
            raise ValueError("posterior orientation coordinate status differs from retained draws")
        values = [record.get(k) for k in ("r_hat", "ess_bulk", "ess_tail")]
        numeric = all(type(v) in (float, int) and np.isfinite(v) and v >= 0 for v in values)
        if record.get("status") == "computed":
            if not numeric:
                raise ValueError("posterior orientation computed diagnostics must be finite")
            passed = values[0] <= 1.01 and values[1] >= 400 and values[2] >= 200
        elif record.get("status") == "undefined_diagnostic":
            if not any(v is None for v in values) or not all(
                v is None or (type(v) in (float, int) and np.isfinite(v) and v >= 0) for v in values
            ):
                raise ValueError(
                    "posterior orientation undefined diagnostics require "
                    "finite values or null, including at least one null"
                )
            passed = False
        else:
            if any(v is not None for v in values):
                raise ValueError("posterior orientation undefined diagnostics must be null")
            passed = False
        if type(record.get("passes")) is not bool or record["passes"] != passed:
            raise ValueError("posterior orientation coordinate pass flag is inconsistent")
    expected_pass = status == "computed" and all(r["passes"] for r in records)
    if type(o.get("passes")) is not bool or o["passes"] != expected_pass:
        raise ValueError("posterior orientation aggregate pass flag is inconsistent")
    if d["passes"] != (d["raw_passes"] and expected_pass):
        raise ValueError("posterior orientation combined pass flag is inconsistent")
    for output, source, reduction in [
        ("max_rank_rhat", "r_hat", max),
        ("min_bulk_ess", "ess_bulk", min),
        ("min_tail_ess", "ess_tail", min),
    ]:
        numbers = [r[source] for r in records if r[source] is not None]
        expected = reduction(numbers) if numbers else None
        if o.get(output) != expected:
            raise ValueError("posterior orientation diagnostic aggregate differs")
    trace = o.get("chains")
    if not isinstance(trace, list) or len(trace) != chains:
        raise ValueError("posterior orientation chain records are invalid")
    determinant = quantities[..., 0] if quantities is not None else None
    for i, row in enumerate(trace):
        expected = dict(
            chain=i,
            retained_draws=draws,
            valid_anchor_draws=int(valid[i].sum()),
            positive_determinant_fraction=float(np.mean(determinant[i] > 0))
            if determinant is not None
            else None,
            reflection_transitions=int(np.sum(determinant[i, 1:] != determinant[i, :-1]))
            if determinant is not None
            else None,
        )
        if row != expected:
            raise ValueError("posterior orientation occupancy differs from retained draws")
