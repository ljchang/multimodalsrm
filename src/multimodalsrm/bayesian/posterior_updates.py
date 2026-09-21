"""Explicit joint posterior targets with frozen reference observations.

Every target is a concrete BayesianProblem over a disjoint observation union.
The ledger is reconstructible without fitting/preprocessing estimation. Its
content identity and a reconstruction check protect the sampler/archive
boundaries from accidentally fitting a different target.
"""

import copy
from collections.abc import Mapping
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..data import normalize_data
from . import _archive
from .observation_adapter import BayesianObservationAdapter
from .problem import BayesianProblem
from .response_scope import (
    quadrature_state,
    validate_posterior_response_target,
    validate_posterior_responses,
)

ADAPTER_FIELDS = (
    "subjects_",
    "modalities_",
    "responses_",
    "grids_",
    "domains_",
    "preprocessing_",
    "configuration_",
    "_training_data",
)
KINDS = ("donor", "calibration", "participant")
SCOPES = {
    "donor": "reference_plus_current_donors",
    "calibration": "joint_group_and_new_participant",
    "participant": "reference_plus_one_participant_new_runs",
}


def content_id(value):
    from .posterior_persistence import _training_digest

    return _training_digest(value)


def adapter_state(adapter):
    return copy.deepcopy(
        dict(
            constructor=adapter.get_params(deep=False),
            fitted={name: getattr(adapter, name) for name in ADAPTER_FIELDS},
        )
    )


def restore_adapter(state):
    """Restore supplied units and frozen preprocessing, without estimating it."""
    if set(state) != {"constructor", "fitted"} or set(state["fitted"]) != set(ADAPTER_FIELDS):
        raise ValueError("invalid posterior update adapter state")
    adapter = BayesianObservationAdapter(**copy.deepcopy(state["constructor"]))
    for name, value in state["fitted"].items():
        setattr(adapter, name, copy.deepcopy(value))
    return adapter


def validate_group(model):
    check_is_fitted(model, "parameter_draws_")
    if model.inference != "posterior" or model.configuration_.get("inference") != "posterior":
        raise ValueError("posterior updates require a fitted posterior group")
    if not _archive.same(model._specification(), model.specification_):
        raise ValueError("model specification changed after fitting")
    p = model.problem_
    if (
        type(p) is not BayesianProblem
        or model.linear_algebra not in ("dense", "grouped")
        or model.run_baseline_sd
        or model.noise_timescales
        or model.sample_blocks is not None
    ):
        raise ValueError(
            "posterior updates require a full dense/grouped target with independent noise and no run baselines"
        )
    validate_posterior_response_target(p)
    if hasattr(model, "posterior_update_"):
        validate_update_target(p)
        if not _archive.same(model.posterior_update_, p._posterior_update):
            raise ValueError("posterior update evidence ledger differs from target")


def reference_state(model):
    """Donor batches replace; a participant calibration establishes a reference."""
    validate_group(model)
    if getattr(model, "posterior_update_", {}).get("kind") in ("donor", "participant"):
        return copy.deepcopy(model.posterior_update_["reference"])
    p = model.problem_
    reference = copy.deepcopy(
        dict(
            adapter=adapter_state(model.adapter_),
            anchor=p.anchor,
            reference_convention=p.reference_convention,
            factor_anchor_keys=getattr(model, "_factor_anchor_keys_", None),
            parameter_names=p.names,
            parameter_priors=p.parameter_priors,
            specification=model.specification_,
            source_draws_digest=content_id(model.parameter_draws_),
            training_fit=model.training_fit_,
        )
    )
    reference["source_fit_id"] = content_id(reference)
    return reference


def validate_reference(reference):
    """Bind the complete frozen snapshot and source draw digest to its identity."""
    fields = {
        "adapter",
        "anchor",
        "reference_convention",
        "factor_anchor_keys",
        "parameter_names",
        "parameter_priors",
        "specification",
        "source_draws_digest",
        "training_fit",
        "source_fit_id",
    }
    if not isinstance(reference, dict) or set(reference) != fields:
        raise ValueError("invalid posterior update reference snapshot")
    digest = reference["source_draws_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid posterior update reference draw identity")
    expected = content_id({k: v for k, v in reference.items() if k != "source_fit_id"})
    if expected != reference["source_fit_id"]:
        raise ValueError("posterior update reference identity differs from snapshot")


def same_pytree(actual, expected):
    """Compare effective NumPyro distributions/transforms, including all leaves."""
    from ._backend import runtime

    jax, _, _, _ = runtime()
    left, left_tree = jax.tree_util.tree_flatten(actual)
    right, right_tree = jax.tree_util.tree_flatten(expected)
    if left_tree != right_tree:
        return False
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        for a, b in zip(left, right)
    )


def validate_preprocessing(data, preprocessing):
    """Check supplied-unit mappings without estimating statistics on load."""
    if not isinstance(preprocessing, dict) or set(preprocessing) != set(data):
        raise ValueError("posterior update preprocessing participant mapping differs")
    for subject, runs in data.items():
        dimensions = {
            modality: ts.values.shape[1]
            for modalities in runs.values()
            for modality, ts in modalities.items()
        }
        supplied = preprocessing[subject]
        if not isinstance(supplied, dict) or set(supplied) != set(dimensions):
            raise ValueError("posterior update preprocessing modality mapping differs")
        for modality, dimension in dimensions.items():
            stats = supplied[modality]
            if not isinstance(stats, dict) or set(stats) != {
                "mean",
                "scale",
                "constant_features",
            }:
                raise ValueError("invalid posterior update preprocessing fields")
            for field, dtype in (
                ("mean", "f"),
                ("scale", "f"),
                ("constant_features", "b"),
            ):
                array = stats[field]
                if (
                    not isinstance(array, np.ndarray)
                    or array.shape != (dimension,)
                    or array.dtype.kind != dtype
                ):
                    raise ValueError("invalid posterior update preprocessing shape or dtype")
            if not np.all(stats["mean"] == 0) or not np.all(stats["scale"] == 1):
                raise ValueError("posterior update preprocessing must preserve supplied units")


def _union(reference, evidence):
    result = copy.deepcopy(reference)
    for subject, runs in evidence.items():
        for run, modalities in runs.items():
            current = result.setdefault(subject, {}).setdefault(run, {})
            if set(current) & set(modalities):
                raise ValueError("posterior update contains duplicate evidence")
            current.update(copy.deepcopy(modalities))
    return result


def rebuild_target(state):
    """Rebuild and validate the exact evidence union; never optimize/sample."""
    required = {
        "version",
        "kind",
        "reference",
        "source_fit_id",
        "evidence",
        "added_preprocessing",
        "targets",
        "participant",
        "priors",
        "linear_algebra",
        "specification",
    }
    if (
        not isinstance(state, dict)
        or set(state) != required
        or state["version"] != 1
        or state["kind"] not in KINDS
    ):
        raise ValueError("invalid posterior update evidence ledger")
    ref = state["reference"]
    validate_reference(ref)
    if state["source_fit_id"] != ref["source_fit_id"]:
        raise ValueError("posterior update source fit identity differs")
    specification = state["specification"]
    if (
        not _archive.same(specification, ref["specification"])
        or not _archive.same(specification["priors"], asdict(state["priors"]))
        or specification["linear_algebra"] != state["linear_algebra"]
    ):
        raise ValueError("posterior update specification differs from reference")
    adapter = restore_adapter(ref["adapter"])
    from .gp_hyperparameters import length_scale_settings

    length_scale, length_scale_prior = length_scale_settings(specification["length_scale"])
    for name in (
        "features",
        "responses",
        "length_scale",
        "max_observations",
        "covariance_tolerance",
    ):
        expected = specification.get(name, 1 if name == "features" else None)
        if name == "length_scale":
            expected = length_scale
        if not _archive.same(getattr(adapter, name), expected):
            raise ValueError("posterior update adapter differs from reference specification")
    if specification.get("run_baseline_sd") or specification.get("noise_timescales"):
        raise ValueError("posterior update reference specification has unsupported options")
    # Constructor validation resolves default responses and observation identities;
    # it does not estimate preprocessing or select a new time/support domain.
    expected_adapter = clone(adapter)
    expected_adapter._validate(adapter._training_data)
    validate_posterior_responses(adapter.responses_, specification.get("response_quadrature_order"))
    if any(
        not _archive.same(getattr(adapter, name), getattr(expected_adapter, name))
        for name in ("subjects_", "modalities_", "responses_")
    ):
        raise ValueError("posterior update reference response or participant mapping differs")
    validate_preprocessing(adapter._training_data, adapter.preprocessing_)
    if adapter.standardize or state["linear_algebra"] not in ("dense", "grouped"):
        raise ValueError("posterior update requires supplied units and dense/grouped algebra")
    evidence = normalize_data(state["evidence"])
    original = adapter._training_data
    original_domains = dict(adapter.domains_)
    if state["kind"] == "calibration":
        if len(evidence) != 1 or set(evidence) & set(original):
            raise ValueError("calibration requires exactly one new participant ID")
        subject = next(iter(evidence))
        if state["participant"] != subject or set(state["added_preprocessing"]) != {subject}:
            raise ValueError("calibration participant/preprocessing differs from evidence")
        validate_preprocessing(evidence, state["added_preprocessing"])
        for run, modalities in evidence[subject].items():
            if run not in original_domains:
                raise ValueError("calibration run IDs must name the same training stimulus")
            lo, hi = original_domains[run]
            for modality, ts in modalities.items():
                if modality not in adapter.responses_:
                    raise ValueError("calibration modality has no trained group filter")
                observed = ts.times[ts.mask.any(axis=1)]
                if observed.min() < lo or observed.max() > hi:
                    raise ValueError("calibration observations extend outside the training run")
        adapter.preprocessing_.update(copy.deepcopy(state["added_preprocessing"]))
        adapter.subjects_.append(subject)
        prediction_runs = original_domains
    else:
        if state["added_preprocessing"]:
            raise ValueError("donor updates cannot alter preprocessing")
        grids, prediction_runs = adapter._grids(evidence)
        if set(prediction_runs) & set(original_domains):
            raise ValueError(
                "donor run IDs overlap training; use distinct IDs for independent runs"
            )
        if state["kind"] == "participant" and set(evidence) != {state["participant"]}:
            raise ValueError("participant update must contain only that participant's new evidence")
        if state["kind"] == "donor":
            targets = state["targets"]
            if not isinstance(targets, dict) or not targets:
                raise ValueError("donor update requires prediction targets")
            for s, modalities in targets.items():
                if not modalities or any(
                    m not in adapter.preprocessing_.get(s, {}) for m in modalities
                ):
                    raise ValueError("target mapping is unavailable")
                if any(set(mods) & set(modalities) for mods in evidence.get(s, {}).values()):
                    raise ValueError("excluded target payload occurs in update evidence")
        adapter.grids_.update(grids)
        adapter.domains_.update(prediction_runs)
    adapter._training_data = _union(original, evidence)
    p = BayesianProblem(
        adapter,
        state["priors"],
        linear_algebra=state["linear_algebra"],
        length_scale_prior=length_scale_prior,
        response_quadrature_order=specification.get("response_quadrature_order"),
        conventions_from=SimpleNamespace(
            anchor=ref["anchor"], reference_convention=ref["reference_convention"]
        ),
    )
    old_priors = dict(zip(ref["parameter_names"], ref["parameter_priors"]))
    current_priors = dict(zip(p.names, p.parameter_priors))
    if any(current_priors.get(name) != prior for name, prior in old_priors.items()):
        raise ValueError("posterior update changed a reference parameter prior")
    if state["kind"] != "calibration" and p.names != ref["parameter_names"]:
        raise ValueError("posterior update changed the reference parameter layout")
    if state["kind"] == "calibration":
        for key in p.keys:
            if (
                key[0] == state["participant"]
                and sum(s.keys.count(key) for s in p.systems.values()) < p.features + 1
            ):
                raise ValueError(
                    "each calibration feature needs at least features + 1 eligible observations"
                )
    from .multifactor import validate_anchors

    if p.features > 1:
        validate_anchors(p, ref["factor_anchor_keys"])
    p._posterior_update = copy.deepcopy(state)
    p._posterior_update_id = content_id(state)
    return adapter, p, prediction_runs


def validate_update_target(problem):
    """Authenticate actual observations, priors and cached numerical layout."""
    state = getattr(problem, "_posterior_update", None)
    if type(problem) is not BayesianProblem or state is None:
        raise ValueError(
            "Haar refresh requires a full training constructor target or validated update"
        )
    if content_id(state) != getattr(problem, "_posterior_update_id", None):
        raise ValueError("posterior update evidence identity changed")
    _, rebuilt, _ = rebuild_target(state)
    fields = (
        "systems",
        "keys",
        "groups",
        "names",
        "indices",
        "parameter_priors",
        "priors",
        "bounds",
        "_prior_groups",
        "_packed",
        "grouped_systems",
        "responses",
        "reference_convention",
        "anchor",
        "length_scale",
        "length_scale_prior",
        "features",
        "linear_algebra",
        "run_baseline_sd",
        "noise_timescales",
        "response_quadrature_order",
        "spectral",
        "state_space_gaussian",
    )
    if any(not _archive.same(getattr(problem, name), getattr(rebuilt, name)) for name in fields):
        raise ValueError("posterior update target differs from reconstructed evidence")
    if not _archive.same(quadrature_state(problem), quadrature_state(rebuilt)):
        raise ValueError("posterior update response quadrature differs from reconstructed evidence")
    if not same_pytree(problem.distributions, rebuilt.distributions) or not same_pytree(
        problem.transforms, rebuilt.transforms
    ):
        raise ValueError("posterior update effective density or transforms differ from target")
    if not _archive.same(adapter_state(problem.adapter), adapter_state(rebuilt.adapter)):
        raise ValueError("posterior update adapter differs from evidence")
    return True


def prepare_update(
    model,
    evidence,
    *,
    kind,
    targets=None,
    participant=None,
    added_preprocessing=None,
    search=None,
    sampler=None,
    random_state=None,
):
    """Shared constructor for independent-run and calibration target builders."""
    reference = reference_state(model)
    result = clone(model)
    if search is not None:
        result.search = search
    if sampler is not None:
        result.sampler = sampler
    if random_state is not None:
        result.random_state = random_state
    result._config()
    state = copy.deepcopy(
        dict(
            version=1,
            kind=kind,
            reference=reference,
            source_fit_id=reference["source_fit_id"],
            evidence=evidence,
            added_preprocessing=added_preprocessing or {},
            targets=targets,
            participant=participant,
            priors=model.priors,
            linear_algebra=model.linear_algebra,
            specification=model.specification_,
        )
    )
    result.adapter_, result.problem_, result.prediction_runs_ = rebuild_target(state)
    result.posterior_update_ = state
    result.training_data_ = result.adapter_._training_data
    result.training_fit_ = copy.deepcopy(reference["training_fit"])
    result.targets_ = copy.deepcopy(targets)
    result._factor_anchor_keys_ = reference["factor_anchor_keys"]
    result.specification_ = result._specification()
    return result


def finish_update(result, *, progress=None):
    validate_update_target(result.problem_)
    result._fit_problem(progress=progress)
    result.configuration_.update(update_configuration(result.posterior_update_))
    if result.posterior_update_["kind"] == "calibration":
        from .posterior_persistence import _fit

        result.training_fit_ = copy.deepcopy(_fit(result))
    return result


def update_configuration(state):
    return dict(
        conditioning_mode="joint",
        posterior_target_kind=state["kind"],
        parameter_source="joint_reference_and_evidence",
        parameter_refits=1,
        diagnostic_scope=SCOPES[state["kind"]],
        source_fit_id=state["source_fit_id"],
        evidence_id=content_id(state),
    )


def prepare_condition(model, donors, *, targets, donor_layout="runs"):
    from .prediction import donor_data

    validate_group(model)
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must map subjects to nonempty modality lists")
    excluded = set()
    selected_targets = {}
    for subject, modalities in targets.items():
        if isinstance(modalities, str) or not modalities:
            raise ValueError("targets must contain nonempty modality lists")
        selected_targets[subject] = tuple(modalities)
        for modality in selected_targets[subject]:
            if (subject, modality) not in model.problem_.groups:
                raise ValueError("target mapping is unavailable")
            excluded.add((subject, modality))
    selected = donor_data(donors, excluded, donor_layout)
    if not selected:
        raise ValueError("no usable donor observations remain after target exclusion")
    return prepare_update(model, selected, kind="donor", targets=selected_targets)


def condition_posterior(model, donors, *, targets, donor_layout="runs", progress=None):
    return finish_update(
        prepare_condition(model, donors, targets=targets, donor_layout=donor_layout),
        progress=progress,
    )
