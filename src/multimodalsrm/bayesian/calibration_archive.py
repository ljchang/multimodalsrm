"""Version 2 state: original group and participant fit have separate provenance."""

import copy

from . import _archive
from .calibration import ParticipantCalibration, prepare
from .fitting import SearchConfig


def state(model, standardizer):
    from .persistence import _model_state, _scaler

    model._validate()
    _, adapter, problem = prepare(model._group, model.calibration_data_)
    if not _archive.same(model._contract(adapter=adapter, problem=problem), model._contract()):
        raise ValueError("calibration observations or prepared reference changed after fitting")
    if not model._configuration_matches(model.configuration_):
        raise ValueError("calibration reference or settings changed")
    _scaler(standardizer, model.problem_)
    return dict(
        kind="participant_calibration",
        reference=_model_state(model._group),
        calibration_data=model.calibration_data_,
        parameter_names=model.parameter_names_,
        fit=model._fit,
        phase_seconds=model.phase_seconds_,
        search=model.search_config_,
        seed=model.seed_,
        standardizer=standardizer,
    )


def restore(state):
    from .persistence import _restore_model, _scaler

    try:
        if set(state) != {
            "kind",
            "reference",
            "calibration_data",
            "parameter_names",
            "fit",
            "phase_seconds",
            "search",
            "seed",
            "standardizer",
        }:
            raise ValueError("invalid participant calibration archive fields")
        group, scaler = _restore_model(state["reference"])
        if scaler is not None:
            raise ValueError("group scaler must be separate from participant calibration")
        config, seed = state["search"], state["seed"]
        if not isinstance(config, SearchConfig) or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid calibration search or seed")
        data, adapter, problem = prepare(group, state["calibration_data"])
        if state["parameter_names"] != problem.names:
            raise ValueError("calibration parameter order differs from reconstructed model")
        result = ParticipantCalibration()
        result._group = group
        result.calibration_data_ = data
        result._adapter, result.problem_ = adapter, problem
        result.subject_ = next(iter(data))
        result.calibration_modalities_ = [m for _, m in problem.groups]
        result.search_config_, result.seed_ = config, seed
        result._install(state["fit"], state["phase_seconds"])
        scaler = copy.deepcopy(state["standardizer"])
        _scaler(scaler, problem)
        return result, scaler
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid participant calibration archive: {exc}") from exc
