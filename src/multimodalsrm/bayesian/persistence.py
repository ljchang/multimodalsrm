"""Rebuild a training MAP estimator from validated, non-executable state."""

import copy
from dataclasses import asdict

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..data import readonly_array
from . import _archive
from .fitting import SamplerConfig, SearchConfig
from .model import BayesianMultimodalSRM
from .multifactor import orientation
from .observation_adapter import BayesianObservationAdapter
from .problem import BayesianProblem
from .workflow import TrainingStandardizer

STATE_KEYS = {
    "constructor",
    "training",
    "fit",
    "parameter_names",
    "phase_seconds",
    "log_likelihood_draws",
    "standardizer",
}


def _search_for_validation(config):
    """Compare legacy search provenance without rewriting the saved record."""
    expected = asdict(SearchConfig())
    expected["r_init"] = False  # Historical fits never ran the R initializer.
    optional = {"polish_max_parameters", "n_jobs", "conditioning", "r_init"}
    if isinstance(config, dict) and set(expected) - optional <= set(config) <= set(expected):
        return {**expected, **config}
    return config


def _restore_search_default(constructor, fit):
    """Make an implicit pre-R-init search explicit without rewriting fit evidence."""
    search = fit.get("configuration", {}).get("search", {})
    if constructor.get("search") is None and isinstance(search, dict) and "r_init" not in search:
        constructor["search"] = SearchConfig(r_init=False)
    return constructor


def _sampler_for_validation(config):
    """Compare legacy sampler provenance without rewriting the saved record."""
    expected = asdict(SamplerConfig())
    optional = {"mass_matrix", "max_dense_parameters", "orientation_refresh"}
    if isinstance(config, dict) and set(expected) - optional <= set(config) <= set(expected):
        return {**expected, **config}
    return config


def _finite(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _fit_metadata(fit, count, timing, *, inference="map"):
    expected = {
        "inference",
        "map_parameters",
        "map_diagnostics",
        "restart_diagnostics",
        "objective",
        "configuration",
    }
    if not isinstance(fit, dict) or set(fit) != expected or fit["inference"] != inference:
        raise ValueError(f"invalid training {inference} fit record")
    if not isinstance(fit["configuration"], dict) or not _finite(fit["objective"]):
        raise ValueError("invalid MAP configuration or objective")
    restarts = fit["restart_diagnostics"]
    if not isinstance(restarts, list) or not restarts:
        raise ValueError("restart diagnostics must be a nonempty list")
    for record in [fit["map_diagnostics"], *restarts]:
        selected = record is fit["map_diagnostics"]
        if not isinstance(record, dict) or "objective" not in record:
            raise ValueError("invalid MAP diagnostics")
        value = record["objective"]
        if value is None and not selected:
            if "failure" in record and not isinstance(record["failure"], str):
                raise ValueError("invalid restart failure diagnostic")
            continue
        required = {
            "parameters",
            "optimizer_success",
            "physical_projected_gradient",
            "meets_gradient_tolerance",
        }
        if not _finite(value) or not required <= record.keys():
            raise ValueError("invalid finite MAP diagnostics")
        point = np.asarray(record["parameters"])
        if point.shape != (count,) or not np.isfinite(point).all():
            raise ValueError("invalid restart parameters")
        if any(
            type(record[k]) not in (bool, np.bool_)
            for k in ("optimizer_success", "meets_gradient_tolerance")
        ):
            raise ValueError("MAP diagnostic flags must be Boolean")
        gradient = record["physical_projected_gradient"]
        if not _finite(gradient) or gradient < 0:
            raise ValueError("invalid MAP gradient diagnostic")
    if not isinstance(timing, dict) or any(
        not isinstance(k, str) or not _finite(v) or v < 0 for k, v in timing.items()
    ):
        raise ValueError("invalid fit timing diagnostics")


def _prepare(model, training, *, conventions_from=None):
    from .gp_hyperparameters import length_scale_settings

    length_scale, length_scale_prior = length_scale_settings(model.length_scale)
    adapter = BayesianObservationAdapter(
        features=model.features,
        latent_dt=1.0,
        responses=model.responses,
        length_scale=length_scale,
        standardize=False,
        max_observations=model.max_observations,
        covariance_tolerance=model.covariance_tolerance,
    )
    adapter._prepare(training)
    problem = BayesianProblem(
        adapter,
        model.priors,
        anchor=model._anchor_choice(),
        reference_modality=model.reference_modality,
        linear_algebra=model.linear_algebra,
        spectral=model.spectral,
        run_baseline_sd=model.run_baseline_sd,
        noise_timescales=model.noise_timescales,
        response_quadrature_order=model.response_quadrature_order,
        state_space_gaussian=model.state_space_gaussian,
        conventions_from=conventions_from,
        length_scale_prior=length_scale_prior,
    )
    return adapter, problem


def _training_problem(model):
    """Verify saved observations describe the already fitted native problem."""
    adapter, problem = _prepare(model, model.training_data_)
    from .response_scope import quadrature_state

    if not _archive.same(quadrature_state(problem), quadrature_state(model.problem_)):
        raise ValueError("training response quadrature changed after fitting")
    for field in ("preprocessing_", "domains_", "responses_"):
        if not _archive.same(getattr(adapter, field), getattr(model.adapter_, field)):
            raise ValueError(f"training adapter {field} changed after fitting")
    for field in (
        "systems",
        "groups",
        "keys",
        "names",
        "bounds",
        "parameter_priors",
        "responses",
        "reference_convention",
        "run_baseline_sd",
        "noise_timescales",
        "noise_systems",
        "length_scale",
        "length_scale_prior",
        "linear_algebra",
        "spectral",
        "state_space_gaussian",
        "covariance_error_bound",
        "response_quadrature_order",
    ):
        if not _archive.same(getattr(problem, field), getattr(model.problem_, field)):
            raise ValueError(f"training problem {field} changed after fitting")


def _scaler(standardizer, problem):
    if standardizer is None:
        return
    if type(standardizer) is not TrainingStandardizer or not isinstance(
        standardizer.statistics, dict
    ):
        raise ValueError("standardizer must be TrainingStandardizer or None")
    if set(standardizer.statistics) != set(problem.groups):
        raise ValueError("standardizer mappings differ from fitted model")
    for key, stats in standardizer.statistics.items():
        count = sum(k[:2] == key for k in problem.keys)
        if not isinstance(stats, dict) or set(stats) != {"mean", "scale"}:
            raise ValueError("invalid standardizer statistics")
        mean, scale = np.asarray(stats["mean"]), np.asarray(stats["scale"])
        if (
            mean.shape != (count,)
            or scale.shape != (count,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 1e-12)
        ):
            raise ValueError("standardizer requires finite means and positive compatible scales")


def _parameters(model, names, point, fit):
    if names != list(model.problem_.names):
        raise ValueError("saved parameter order differs from reconstructed model")
    point = np.asarray(point)
    bounds = np.asarray(model.problem_.bounds)
    if (
        point.shape != (len(names),)
        or not np.isfinite(point).all()
        or np.any(point < bounds[:, 0])
        or np.any(point > bounds[:, 1])
    ):
        raise ValueError("invalid MAP shape, values or parameter support")
    for name, x in zip(names, point):
        if name[0] == "noise" and x <= 0:
            raise ValueError("MAP noise variance must be positive")
    if (
        not np.isfinite(fit["objective"])
        or fit["objective"] != fit["map_diagnostics"]["objective"]
        or not np.array_equal(point, fit["map_diagnostics"]["parameters"])
    ):
        raise ValueError("MAP parameters or objective differ from fit diagnostics")


def _orientation(expected, saved):
    """Allow float64 QR/SVD roundoff, but retain the original coordinates."""
    numeric = {"rotation", "anchor_singular_values"}
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("invalid training factor orientation fields")
    for key in expected:
        if key not in numeric:
            if not _archive.same(expected[key], saved[key]):
                raise ValueError("training factor orientation provenance differs")
            continue
        a, b = np.asarray(expected[key]), np.asarray(saved[key])
        if (
            b.dtype.kind not in "fiu"
            or a.shape != b.shape
            or not np.isfinite(b).all()
            or not np.allclose(a, b, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("training factor orientation differs from MAP anchors")
        if key == "rotation" and not np.allclose(b.T @ b, np.eye(len(b)), rtol=0, atol=1e-10):
            raise ValueError("training factor orientation is not orthogonal")


def _validate_training_model(model, standardizer, *, rebuild=True):
    if type(model) is not BayesianMultimodalSRM:
        raise ValueError("archive requires a built-in BayesianMultimodalSRM")
    check_is_fitted(model, ["training_fit_", "parameter_draws_", "configuration_"])
    if model.configuration_.get("inference") == "posterior":
        raise ValueError(
            "model persistence and calibration require fitted MAP; posterior is unsupported"
        )
    fit = model.training_fit_
    _fit_metadata(fit, len(model.parameter_names_), model.phase_seconds_)
    if any(
        x != "map" for x in (model.inference, fit["inference"], model.configuration_["inference"])
    ):
        raise ValueError("model archive requires fitted MAP inference")
    if (
        model.targets_ is not None
        or "conditioning_mode" in model.configuration_
        or model.prediction_runs_ != model.adapter_.domains_
        or set(model.problem_.systems) != set(model.adapter_.domains_)
    ):
        raise ValueError("save the original training model, not a conditioned copy")
    if not _archive.same(model._specification(), model.specification_):
        raise ValueError("model specification changed after fitting")
    if (
        not _archive.same(model.search or SearchConfig(), model.search_config_)
        or not _archive.same(model.sampler or SamplerConfig(), model.sampler_config_)
        or (model.random_state is not None and model.random_state != model.seed_)
        or asdict(model.search_config_) != _search_for_validation(model.configuration_["search"])
        or asdict(model.sampler_config_) != _sampler_for_validation(model.configuration_["sampler"])
        or model.seed_ != model.configuration_["seed"]
    ):
        raise ValueError("fitted search, sampler or seed configuration changed")
    for key, value in (
        ("map_parameters", model.map_parameters_),
        ("map_diagnostics", model.map_diagnostics_),
        ("restart_diagnostics", model.restart_diagnostics_),
        ("objective", model.objective_),
        ("configuration", model.configuration_),
    ):
        if not _archive.same(value, fit[key]):
            raise ValueError(f"current {key} differs from original training fit")
    if (
        model.parameter_draws_.shape != (1, 1, len(model.map_parameters_))
        or not np.array_equal(model.parameter_draws_[0, 0], model.map_parameters_)
        or model.sampling_diagnostics_ is not None
        or model.sample_stats_
    ):
        raise ValueError("inconsistent MAP draws or posterior state")
    _parameters(model, model.parameter_names_, model.map_parameters_, fit)
    if model.features > 1:
        expected = orientation(model.problem_, model.map_parameters_, model.factor_anchors)
        _orientation(expected, model.factor_orientation_)
        if not _archive.same(model.factor_orientation_, model.configuration_["factor_orientation"]):
            raise ValueError("training factor orientation differs from configuration")
    _scaler(standardizer, model.problem_)
    if rebuild:
        _training_problem(model)


def save_model(path, model, *, standardizer=None):
    from .calibration import ParticipantCalibration
    from .calibration_archive import state as calibration_state

    if type(model) is ParticipantCalibration:
        return _archive.write(path, calibration_state(model, standardizer), version=2)
    if (
        type(model) is BayesianMultimodalSRM
        and getattr(model, "configuration_", {}).get("inference") == "posterior"
    ):
        if hasattr(model, "posterior_update_"):
            from .updated_posterior_persistence import state as updated_state

            return _archive.write(path, updated_state(model, standardizer), version=4)
        from .posterior_persistence import state as posterior_state

        return _archive.write(path, posterior_state(model, standardizer), version=3)
    _archive.write(path, _model_state(model, standardizer))


def _model_state(model, standardizer=None):
    _validate_training_model(model, standardizer)
    return dict(
        constructor=model.get_params(deep=False),
        training=model.training_data_,
        fit=model.training_fit_,
        parameter_names=model.parameter_names_,
        phase_seconds=model.phase_seconds_,
        log_likelihood_draws=getattr(model, "log_likelihood_draws_", None),
        standardizer=standardizer,
    )


def load_model(path):
    state = _archive.read(path)
    from .calibration_archive import restore

    if isinstance(state, dict) and state.get("kind") == "posterior_updated":
        from .updated_posterior_persistence import restore as restore_updated

        return restore_updated(state)
    if isinstance(state, dict) and state.get("kind") == "posterior_training":
        from .posterior_persistence import restore as restore_posterior

        return restore_posterior(state)
    if isinstance(state, dict) and state.get("kind") == "participant_calibration":
        return restore(state)
    return _restore_model(state)


def _restore_model(state):
    try:
        if not isinstance(state, dict) or set(state) != STATE_KEYS:
            raise ValueError("invalid fitted-model state fields")
        constructor = _restore_search_default(copy.deepcopy(state["constructor"]), state["fit"])
        if "state_space_gaussian" not in constructor:
            descriptor = state.get("fit", {}).get("configuration", {}).get("state_space", {})
            if descriptor.get("response_approximation") == (
                "qualified_Gaussian_Laguerre_and_restored_gamma_tails"
            ):
                constructor["state_space_gaussian"] = "laguerre"
        model = BayesianMultimodalSRM(**constructor)
        fit = state["fit"]
        _fit_metadata(fit, len(state["parameter_names"]), state["phase_seconds"])
        if model.inference != "map" or fit["inference"] != "map":
            raise ValueError("model archive requires training MAP inference")
        model._config()
        seed = fit["configuration"]["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid original training seed")
        model.seed_ = seed
        adapter, model.problem_ = _prepare(model, state["training"])
        model.adapter_ = adapter
        _parameters(model, state["parameter_names"], fit["map_parameters"], fit)
        model.specification_ = model._specification()
        model.training_data_ = adapter._training_data
        model.prediction_runs_, model.targets_ = dict(adapter.domains_), None
        model.map_parameters_ = readonly_array(fit["map_parameters"])
        model.parameter_names_ = list(model.problem_.names)
        model.map_diagnostics_ = copy.deepcopy(fit["map_diagnostics"])
        model.restart_diagnostics_ = copy.deepcopy(fit["restart_diagnostics"])
        model.objective_ = fit["objective"]
        model.parameter_draws_ = readonly_array(model.map_parameters_[None, None, :])
        model.sampling_diagnostics_, model.sample_stats_ = None, {}
        model.phase_seconds_ = copy.deepcopy(state["phase_seconds"])
        if state["log_likelihood_draws"] is not None:
            loglik = state["log_likelihood_draws"]
            if (
                not isinstance(loglik, np.ndarray)
                or loglik.shape != (1, 1)
                or not np.isfinite(loglik).all()
            ):
                raise ValueError("invalid MAP log likelihood record")
            model.log_likelihood_draws_ = readonly_array(loglik)
        if model.features > 1:
            expected = orientation(model.problem_, model.map_parameters_, model.factor_anchors)
            _orientation(expected, fit["configuration"]["factor_orientation"])
            model.factor_orientation_ = copy.deepcopy(fit["configuration"]["factor_orientation"])
        model._set_configuration()
        saved_configuration = dict(fit["configuration"])
        saved_configuration["search"] = _search_for_validation(saved_configuration.get("search"))
        saved_configuration["sampler"] = _sampler_for_validation(saved_configuration.get("sampler"))
        if not _archive.same(model.configuration_, saved_configuration):
            raise ValueError("saved configuration differs from reconstructed model")
        # Preserve original reporting coordinates, rather than replacing their
        # provenance with the reconstruction used above to validate the archive.
        model.configuration_ = copy.deepcopy(fit["configuration"])
        if model.features > 1:
            model.factor_orientation_ = copy.deepcopy(fit["configuration"]["factor_orientation"])
        model.training_fit_ = copy.deepcopy(fit)
        standardizer = state["standardizer"]
        _validate_training_model(model, standardizer, rebuild=False)
        return model, standardizer
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid fitted-model state: {exc}") from exc
