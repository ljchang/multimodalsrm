"""Independent participant calibration against an unchanged training group."""

import copy
import time
import warnings
from dataclasses import asdict, replace

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from ..data import normalize_data, readonly_array
from . import _archive
from .calibration_problem import CalibrationProblem
from .fitting import SearchConfig, search
from .model import MAPFitError


def prepare(group, data):
    from .persistence import _prepare, _validate_training_model

    _validate_training_model(group, None)
    if group.linear_algebra not in ("dense", "grouped") or group.run_baseline_sd:
        raise ValueError("calibration supports dense/grouped MAP without run baselines")
    data = normalize_data(data)
    if len(data) != 1:
        raise ValueError("calibrate exactly one new participant at a time")
    subject = next(iter(data))
    if subject in group.training_data_:
        raise ValueError("calibration requires a new participant ID")
    for run, mods in data[subject].items():
        if run not in group.adapter_.domains_:
            raise ValueError("calibration run IDs must name the same training stimulus")
        lo, hi = group.adapter_.domains_[run]
        for m, ts in mods.items():
            if m not in group.problem_.responses:
                raise ValueError("calibration modality has no trained group filter")
            observed = ts.times[ts.mask.any(axis=1)]
            if observed.min() < lo or observed.max() > hi:
                raise ValueError("calibration observations extend outside the training run")
    adapter, joint = _prepare(
        group, {**group.training_data_, **data}, conventions_from=group.problem_
    )
    if adapter.domains_ != group.adapter_.domains_:
        raise ValueError("calibration must preserve original training domains")
    for key in joint.keys:
        if key[0] != subject:
            continue
        count = sum(s.keys.count(key) for s in joint.systems.values())
        if count < group.features + 1:
            raise ValueError(
                "each calibration feature needs at least features + 1 eligible observations"
            )
    return data, adapter, CalibrationProblem(joint, group, subject)


class ParticipantCalibration:
    """One new participant's MAP mappings in a frozen group's coordinates.

    Created by ``group.calibrate`` or ``load_model``. Preprocessing is explicit;
    transform consumes the units used during calibration. No group refit or
    parameter uncertainty calibration is implied by this object.
    """

    def _configuration(self):
        from .identity import training_reference_id

        return dict(
            kind="participant_calibration_MAP",
            participant=self.subject_,
            features=self._group.features,
            reference_id=training_reference_id(self._group),
            reference_configuration=copy.deepcopy(self._group.configuration_),
            original_group_meets_gradient_tolerance=self._group.map_diagnostics_[
                "meets_gradient_tolerance"
            ],
            calibrated_parameters=list(self.problem_.names),
            calibration_runs=list(self.calibration_data_[self.subject_]),
            search=asdict(self.search_config_),
            seed=self.seed_,
            objective="negative_conditional_log_likelihood_minus_new_parameter_log_prior",
            latent_marginalization="analytic_conditional_on_group_observations",
            diagnostic_scope="new_participant_parameters_only",
            uncertainty="conditional_on_group_MAP_and_participant_calibration_MAP",
            calibration_established=False,
        )

    def _configuration_matches(self, saved):
        """Compare additive search defaults without changing saved provenance."""
        from .persistence import _search_for_validation

        if not isinstance(saved, dict):
            return False
        normalized = {**saved, "search": _search_for_validation(saved.get("search"))}
        return _archive.same(normalized, self._configuration())

    def _install(self, fit, timing):
        from .persistence import _fit_metadata, _parameters

        _fit_metadata(fit, len(self.problem_.names), timing)
        _parameters(self, list(self.problem_.names), fit["map_parameters"], fit)
        if not self._configuration_matches(fit["configuration"]):
            raise ValueError("calibration configuration differs from frozen reference")
        self.map_parameters_ = readonly_array(fit["map_parameters"])
        self.parameter_names_ = list(self.problem_.names)
        self.map_diagnostics_ = copy.deepcopy(fit["map_diagnostics"])
        self.restart_diagnostics_ = copy.deepcopy(fit["restart_diagnostics"])
        self.objective_ = fit["objective"]
        self.configuration_ = copy.deepcopy(fit["configuration"])
        self.phase_seconds_ = copy.deepcopy(timing)
        self._fit = copy.deepcopy(fit)
        self._timing = copy.deepcopy(timing)
        self._frozen_contract = copy.deepcopy(self._contract())

    def _contract(self, *, adapter=None, problem=None):
        """Only prepared numerical state used by the frozen inference path."""
        adapter = self._adapter if adapter is None else adapter
        p = self.problem_ if problem is None else problem
        joint = p.joint
        return dict(
            data=self.calibration_data_,
            subject=self.subject_,
            modalities=self.calibration_modalities_,
            adapter={
                k: getattr(adapter, k)
                for k in ("preprocessing_", "domains_", "responses_", "length_scale")
            },
            free={
                k: getattr(p, k)
                for k in (
                    "reference",
                    "active",
                    "names",
                    "indices",
                    "keys",
                    "groups",
                    "bounds",
                    "parameter_priors",
                    "group_nll",
                    "features",
                )
            },
            joint={
                **(
                    {"length_scale_prior": joint.length_scale_prior}
                    if joint.length_scale_prior is not None
                    else {}
                ),
                **{
                    k: getattr(joint, k)
                    for k in (
                        "systems",
                        "groups",
                        "keys",
                        "names",
                        "indices",
                        "bounds",
                        "parameter_priors",
                        "responses",
                        "reference_convention",
                        "run_baseline_sd",
                        "noise_timescales",
                        "noise_systems",
                        "length_scale",
                        "linear_algebra",
                        "spectral",
                        "covariance_error_bound",
                        "modalities",
                        "_packed",
                        "grouped_systems",
                    )
                },
            },
        )

    def _validate(self):
        from .persistence import _validate_training_model

        _validate_training_model(self._group, None, rebuild=False)
        if not _archive.same(self._contract(), self._frozen_contract):
            raise ValueError("calibration observations or inference contract changed")
        if not self._configuration_matches(self.configuration_):
            raise ValueError("calibration reference or settings changed")
        for key, value in (
            ("map_parameters", self.map_parameters_),
            ("map_diagnostics", self.map_diagnostics_),
            ("restart_diagnostics", self.restart_diagnostics_),
            ("objective", self.objective_),
            ("configuration", self.configuration_),
        ):
            if not _archive.same(value, self._fit[key]):
                raise ValueError("calibration state changed after fitting")
        if self.parameter_names_ != self.problem_.names or not _archive.same(
            self.phase_seconds_, self._timing
        ):
            raise ValueError("calibration parameter order or timings changed")

    def transform(self, data, *, times, modalities=None, data_layout="runs", readout="gp"):
        """Independently transform this participant on new stimulus runs."""
        from .transform import _select

        self._validate()
        selected = _select(data, modalities, data_layout, set(self.calibration_modalities_))
        if set(selected) != {self.subject_}:
            raise ValueError("transform requires the calibrated participant only")
        # The private inference carrier reuses the existing native GP query path.
        # It is never exported as a new group fit or optimized.
        carrier = clone(self._group)
        carrier._config()
        carrier.adapter_ = self._adapter
        carrier.problem_ = self.problem_.joint
        carrier.training_data_ = self._adapter._training_data
        carrier.specification_ = carrier._specification()
        carrier.training_fit_ = copy.deepcopy(self._group.training_fit_)
        carrier.training_fit_["map_parameters"] = np.asarray(
            self.problem_.expand(self.map_parameters_)
        )
        carrier.parameter_draws_ = readonly_array(
            carrier.training_fit_["map_parameters"][None, None, :]
        )
        carrier.configuration_ = copy.deepcopy(self._group.configuration_)
        output = carrier.transform(selected, times=times, readout=readout)
        for runs in output.values():
            for run, result in runs.items():
                runs[run] = replace(
                    result,
                    metadata={
                        **result.metadata,
                        "parameter_source": "group_MAP_plus_participant_calibration_MAP",
                        "parameter_conditioning": "group_MAP_plus_participant_calibration_MAP",
                        "calibration_runs": list(self.configuration_["calibration_runs"]),
                        "uncertainty": self.configuration_["uncertainty"],
                        "training_reference_id": self.configuration_["reference_id"],
                        "calibration_diagnostic_scope": "new_participant_parameters_only",
                        "calibration_meets_gradient_tolerance": self.map_diagnostics_[
                            "meets_gradient_tolerance"
                        ],
                    },
                )
        return output


def calibrate(group, data, *, search_config=None, random_state=None, progress=None):
    config = SearchConfig(starts=4, maxiter=400) if search_config is None else search_config
    if not isinstance(config, SearchConfig):
        raise ValueError("search must be SearchConfig")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or not 0 <= random_state < 2**32
    ):
        raise ValueError("random_state must be an integer in [0, 2**32) or None")
    data, adapter, problem = prepare(group, data)
    result = ParticipantCalibration()
    result._group = copy.deepcopy(group)
    result.calibration_data_ = copy.deepcopy(data)
    result._adapter, result.problem_ = adapter, problem
    result.subject_ = next(iter(data))
    result.calibration_modalities_ = [m for s, m in problem.joint.groups if s == result.subject_]
    result.search_config_ = config
    result.seed_ = int(
        random_state if random_state is not None else np.random.SeedSequence().generate_state(1)[0]
    )
    start = time.perf_counter()
    best, records = search(problem, config, result.seed_, progress=progress)
    timing = {"calibration_map": time.perf_counter() - start}
    if best is None:
        raise MAPFitError(records)
    fit = dict(
        inference="map",
        map_parameters=np.asarray(best["parameters"]),
        map_diagnostics=best,
        restart_diagnostics=records,
        objective=best["objective"],
        configuration=result._configuration(),
    )
    result._install(fit, timing)
    if not best["meets_gradient_tolerance"]:
        warnings.warn(
            "Participant calibration physical gradient threshold was not met; inspect map_diagnostics_",
            ConvergenceWarning,
            stacklevel=2,
        )
    return result
