"""Participant-wise inference in fixed, common training coordinates."""

import copy
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..data import normalize_data
from ..kernels import Identity
from .prediction import queries
from .problem import BayesianProblem
from .results import GaussianMixtureSeries


def _select(data, modalities, layout, available):
    if layout not in ("runs", "shorthand"):
        raise ValueError("data_layout must be runs or shorthand")
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty subject mapping")
    if modalities is not None:
        if (
            not isinstance(modalities, (list, tuple))
            or not modalities
            or any(not isinstance(m, str) for m in modalities)
            or len(set(modalities)) != len(modalities)
            or not set(modalities) <= available
        ):
            raise ValueError("modalities must be a nonempty unique list of trained modalities")
    output = {}
    for subject, entries in data.items():
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError("subject entries must be nonempty mappings")
        runs = {"run-01": entries} if layout == "shorthand" else entries
        output[subject] = {}
        for run, mods in runs.items():
            if not isinstance(mods, Mapping):
                raise ValueError("runs must contain modality mappings")
            selected = {m: mods[m] for m in mods if modalities is None or m in modalities}
            if not selected:
                raise ValueError(f"no selected observations for {subject}/{run}")
            output[subject][run] = selected
    return normalize_data(output)


def _instantaneous(model, system, times):
    """Independent unit-prior Gaussian solve at each exactly observed time."""
    training = model.training_fit_
    config = training["configuration"]
    parameters = dict(zip(model.problem_.names, training["map_parameters"]))
    k = model.features
    rotation = np.asarray(config["factor_orientation"]["rotation"]) if k > 1 else np.eye(1)
    weights = (
        np.array(
            [
                [parameters[("loading", *key, f)] for f in range(k)]
                if k > 1
                else [parameters[("loading", *key)]]
                for key in system.keys
            ]
        )
        @ rotation
    )
    offsets = np.array([parameters[("offset", *key)] for key in system.keys])
    noise = np.array([parameters[("noise", *key[:2])] for key in system.keys])
    means, variances = (
        np.full((len(times), k), np.nan),
        np.full((len(times), k), np.nan),
    )
    counts = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        rows = system.times == t
        counts[i] = rows.sum()
        if not counts[i]:
            continue
        w, variance = weights[rows], noise[rows]
        precision = np.eye(k) + w.T @ (w / variance[:, None])
        covariance = np.linalg.solve(precision, np.eye(k))
        means[i] = np.linalg.solve(
            precision, w.T @ ((system.values[rows] - offsets[rows]) / variance)
        )
        variances[i] = np.diag(covariance)
    metadata = dict(
        quantity="unfiltered_shared_latent",
        uncertainty="conditional_on_training_MAP",
        posterior_scope="instantaneous_marginal_model",
        alignment_readout="instantaneous",
        temporal_conditioning="same_timestamp_only",
        latent_marginal_variance=1.0,
        reference_convention=copy.deepcopy(config["reference_convention"]),
        conditioning_mode="independent",
        parameter_source="original_training_MAP",
        parameter_conditioning="original_training_MAP",
        parameter_refits=0,
        calibration_established=False,
        include_noise=False,
        n_parameter_draws=1,
        n_conditioning_observations=int(counts.sum()),
        n_conditioning_observations_per_query=counts.tolist(),
        support_rule="exact_observed_timestamp_with_at_least_one_feature",
    )
    if k > 1:
        metadata["factor_orientation"] = copy.deepcopy(config["factor_orientation"])
    return GaussianMixtureSeries(means[None], variances[None], times, counts > 0, metadata)


def independent_transform(model, data, *, times, modalities, data_layout, readout="gp"):
    check_is_fitted(model, "parameter_draws_")
    if readout not in ("gp", "instantaneous"):
        raise ValueError("readout must be gp or instantaneous")
    if (
        model.inference != "map"
        or model.training_fit_["inference"] != "map"
        or model.configuration_["inference"] != "map"
    ):
        raise ValueError("independent transform requires fitted MAP inference")
    if model._specification() != model.specification_:
        raise ValueError("model specification changed after fitting; call fit again")
    selected = _select(data, modalities, data_layout, {m for _, m in model.problem_.groups})
    if readout == "instantaneous":
        chosen = {m for runs in selected.values() for mods in runs.values() for m in mods}
        if (
            model.linear_algebra == "spectral"
            or any(
                type(model.adapter_.responses_[m].initial_kernel()) is not Identity
                for m in chosen
                if m in model.adapter_.responses_
            )
            or chosen & set(model.noise_timescales or {})
            or chosen & set(model.run_baseline_sd or {})
        ):
            raise ValueError(
                "instantaneous readout requires Identity responses, white noise, no run baselines and no spectral approximation"
            )
    from .identity import training_reference_id

    reference_id = training_reference_id(model)
    output = {}
    for subject, runs in selected.items():
        # Domain selection and likelihood construction must BOTH be subject-only.
        adapter = copy.deepcopy(model.adapter_)
        if readout == "instantaneous":
            domains = {}
            for run, mods in runs.items():
                observed = np.concatenate([ts.times[ts.mask.any(axis=1)] for ts in mods.values()])
                if not len(observed):
                    raise ValueError("each instantaneous run requires an observed timestamp")
                domains[run] = (observed.min(), observed.max())
        else:
            _, domains = adapter._grids({subject: runs})
        if set(domains) & set(adapter.domains_):
            raise ValueError("transform run IDs overlap training; use distinct IDs")
        systems, _ = adapter._systems({subject: runs}, domains)
        current = clone(model)
        current._config()
        current.adapter_ = adapter
        current.training_data_ = adapter._training_data
        current.training_fit_ = copy.deepcopy(model.training_fit_)
        current.prediction_runs_ = domains
        current.targets_ = None
        query = queries(current, times)
        if readout == "instantaneous":
            results = {run: _instantaneous(model, systems[run], q) for run, q in query.items()}
        else:
            current.problem_ = BayesianProblem(
                adapter,
                model.priors,
                anchor=model._anchor_choice(),
                reference_modality=model.reference_modality,
                systems=systems,
                linear_algebra=model.linear_algebra,
                spectral=model.spectral,
                run_baseline_sd=model.run_baseline_sd,
                noise_timescales=model.noise_timescales,
                response_quadrature_order=model.response_quadrature_order,
                state_space_gaussian=model.state_space_gaussian,
                conventions_from=model.problem_,
                length_scale_prior=model.problem_.length_scale_prior,
            )
            current._condition_frozen()
            current.configuration_["conditioning_mode"] = "independent"
            results = current.infer_latent(times=query)
        output[subject] = {
            run: replace(
                result,
                metadata={
                    **result.metadata,
                    "independent_subject_inference": True,
                    "conditioning_subject": subject,
                    "conditioning_run": run,
                    "conditioning_modalities": sorted(runs[run]),
                    "training_reference_id": reference_id,
                },
            )
            for run, result in results.items()
        }
    return output
