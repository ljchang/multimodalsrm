"""Version 3 posterior training archives, without inference or sampler continuation."""

import copy
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..data import readonly_array
from . import _archive
from .fitting import SamplerConfig, SearchConfig
from .model import BayesianMultimodalSRM
from .multifactor import validate_anchors
from .persistence import (
    _fit_metadata,
    _parameters,
    _prepare,
    _sampler_for_validation,
    _scaler,
    _search_for_validation,
    _training_problem,
)
from .posterior_coordinates import metadata

STATE_KEYS = {
    "kind",
    "constructor",
    "training",
    "fit",
    "original_training_fit",
    "parameter_names",
    "phase_seconds",
    "parameter_draws",
    "log_likelihood_draws",
    "sample_stats",
    "sampling_diagnostics",
    "factor_anchor_keys",
    "standardizer",
    "draw_axes",
    "parameter_coordinates",
    "posterior_provenance",
    "writer_provenance",
    "training_data_sha256",
}


def _provenance(value, depth=0):
    """Only finite non-executable JSON-like caller/writer metadata is accepted."""
    if depth > 32:
        raise ValueError("posterior provenance exceeds maximum nesting")
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float and np.isfinite(value):
        return
    if isinstance(value, dict) and all(type(k) is str for k in value):
        for child in value.values():
            _provenance(child, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _provenance(child, depth + 1)
        return
    raise ValueError("posterior provenance must contain finite JSON-like values")


def _writer(model):
    packages = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "numpyro", "arviz"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    folder = Path(__file__).parent
    return dict(
        role="archive_writer_not_training",
        python=platform.python_version(),
        packages=packages,
        source_scope="archive_and_model_modules_at_save_time_not_historical_training",
        source_sha256={
            name: _archive.digest(folder / name)
            for name in (
                "_archive.py",
                "persistence.py",
                "posterior_persistence.py",
                "model.py",
                "problem.py",
                "fitting.py",
                "priors.py",
                "posterior_coordinates.py",
            )
        },
        previous_archive_writer=copy.deepcopy(getattr(model, "archive_writer_provenance_", None)),
    )


def _training_digest(training):
    arrays = {}
    encoded = _archive.encode(training, arrays)
    digest = hashlib.sha256(json.dumps(encoded, sort_keys=True, allow_nan=False).encode())
    for name, value in arrays.items():
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _fit(model):
    return dict(
        inference=model.inference,
        map_parameters=model.map_parameters_,
        map_diagnostics=model.map_diagnostics_,
        restart_diagnostics=model.restart_diagnostics_,
        objective=model.objective_,
        configuration=model.configuration_,
    )


def _numeric(array, shape, label, *, finite=True, floating=True):
    if (
        not isinstance(array, np.ndarray)
        or array.shape != shape
        or (array.dtype.kind != "f" if floating else array.dtype.kind not in "biuf")
        or (finite and not np.isfinite(array).all())
    ):
        raise ValueError(f"invalid posterior {label} shape, dtype or values")


def _sampler_identity(model, dimension):
    """Validate saved public-fit identities without consulting the loader host."""
    config, diagnostic = model.sampler_config_, model.sampling_diagnostics_
    seed = (model.seed_ + 1) % 2**32
    for key in ("sampler_seed", "initialization_seed"):
        # These fields were not recorded by older public fits.
        if key in diagnostic and (type(diagnostic[key]) is not int or diagnostic[key] != seed):
            raise ValueError(f"posterior {key} differs from original public fit")
    execution = diagnostic.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("invalid posterior recorded execution")
    devices = execution.get("devices")
    count = execution.get("local_device_count")
    if (
        type(execution.get("chains")) is not int
        or execution["chains"] != config.chains
        or execution.get("requested_chain_method") != config.chain_method
        or execution.get("effective_chain_method") != config.chain_method
        or type(count) is not int
        or count < 1
        or not isinstance(devices, list)
        or len(devices) != count
        or any(type(device) is not str or not device for device in devices)
        or len(set(devices)) != count
        or type(execution.get("backend")) is not str
        or not execution["backend"]
        or (config.chain_method == "parallel" and count < config.chains)
        or execution.get("parallel_verified")
        is not (config.chains > 1 and config.chain_method == "parallel")
    ):
        raise ValueError("posterior recorded execution contradicts sampler config")
    records = model.restart_diagnostics_
    ids = [record.get("start") for record in records]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid posterior restart identities")
    eligible = {record["start"] for record in records if record["objective"] is not None}
    starts = diagnostic.get("start_indices")
    if (
        not isinstance(starts, list)
        or len(starts) != config.chains
        or any(type(i) is not int or i not in eligible for i in starts)
    ):
        raise ValueError("posterior chain starts differ from recorded restarts")
    _numeric(
        np.asarray(diagnostic.get("initial_unconstrained")),
        (config.chains, dimension),
        "initial unconstrained coordinates",
    )


def _adaptation_identity(diagnostic, metric_shape, metric_dtype, chain_axes):
    """Check retained adaptation descriptors/values without estimating a metric."""
    metric = diagnostic["metric"]
    stored = metric.get("adapted_values_stored")
    if type(stored) is not bool:
        raise ValueError("invalid posterior adapted-value storage flag")
    if "adaptation" not in diagnostic:
        if stored:
            raise ValueError("posterior claims adapted values without adaptation")
        return  # Older samplers retained only metric descriptors.
    adaptation = diagnostic["adaptation"]
    if not isinstance(adaptation, dict) or adaptation.get("values_stored") is not stored:
        raise ValueError("posterior adaptation storage flags differ")
    step_dtype = np.dtype(adaptation.get("step_size_dtype", "invalid"))
    if (
        adaptation.get("inverse_mass_matrix_shape") != metric_shape
        or np.dtype(adaptation.get("inverse_mass_matrix_dtype", "invalid")) != metric_dtype
        or adaptation.get("step_size_shape") != chain_axes
        or step_dtype.kind != "f"
    ):
        raise ValueError("posterior adaptation descriptors differ from metric")
    if not stored:
        if "step_size" in adaptation or "inverse_mass_matrix" in adaptation:
            raise ValueError("posterior adaptation values contradict omission flag")
        return
    step = np.asarray(adaptation.get("step_size"))
    inverse = np.asarray(adaptation.get("inverse_mass_matrix"))
    _numeric(step, tuple(chain_axes), "adapted step size")
    _numeric(inverse, tuple(metric_shape), "adapted inverse metric")
    if np.any(step <= 0):
        raise ValueError("posterior adapted step size must be positive")
    if metric["actual"] == "diagonal":
        if np.any(inverse <= 0):
            raise ValueError("posterior adapted diagonal metric must be positive")
    else:
        if not np.allclose(inverse, inverse.swapaxes(-1, -2), rtol=1e-7, atol=1e-12):
            raise ValueError("posterior adapted dense metric must be symmetric")
        try:
            np.linalg.cholesky(inverse)
        except np.linalg.LinAlgError as exc:
            raise ValueError("posterior adapted dense metric must be positive definite") from exc


def _diagnostics(model):
    """Check coordinate/shape identity while retaining original convergence values."""
    config = model.sampler_config_
    shape = (config.chains, config.draws)
    names = model.parameter_names_
    xs, diagnostic = model.parameter_draws_, model.sampling_diagnostics_
    _numeric(xs, (*shape, len(names)), "raw draws")
    _numeric(model.log_likelihood_draws_, shape, "likelihood draws")
    bounds = np.asarray(model.problem_.bounds)
    if np.any(xs < bounds[:, 0]) or np.any(xs > bounds[:, 1]):
        raise ValueError("posterior draws lie outside physical parameter support")
    noise = [i for i, name in enumerate(names) if name[0] == "noise"]
    if np.any(xs[..., noise] <= 0):
        raise ValueError("posterior noise variance must be positive")
    if not isinstance(diagnostic, dict) or type(diagnostic.get("passes")) is not bool:
        raise ValueError("invalid posterior sampling diagnostics")
    if any(
        type(diagnostic.get(k)) is not int or diagnostic[k] != value
        for k, value in zip(("chains", "draws"), shape)
    ):
        raise ValueError("posterior diagnostic chain/draw identity differs")
    active = [
        i
        for i, name in enumerate(names)
        if model.sample_blocks is None or name[0] in model.sample_blocks
    ]
    active_set = set(active)
    fixed = [i for i in range(len(names)) if i not in active_set]
    if diagnostic.get("active_parameters") != [list(names[i]) for i in active]:
        raise ValueError("posterior active parameter order differs")
    reference = np.asarray(diagnostic.get("reference_parameters"), dtype=float)
    if reference.shape != (len(names),) or not np.isfinite(reference).all():
        raise ValueError("invalid posterior fixed reference")
    if not np.array_equal(reference, model.map_parameters_):
        raise ValueError("posterior reference differs from training MAP")
    expected_fixed = [dict(name=list(names[i]), value=float(reference[i])) for i in fixed]
    if not _archive.same(diagnostic.get("fixed_parameters"), expected_fixed):
        raise ValueError("posterior fixed parameter records differ")
    if fixed and not np.all(xs[..., fixed] == reference[fixed]):
        raise ValueError("posterior fixed physical coordinates changed")
    stats = model.sample_stats_
    required = {"energy", "potential_energy", "accept_prob", "num_steps", "diverging"}
    if not isinstance(stats, dict) or not required <= stats.keys():
        raise ValueError("posterior sample statistics are missing")
    for name, array in stats.items():
        if type(name) is not str:
            raise ValueError("posterior sample statistic names must be strings")
        _numeric(array, shape, f"sample statistic {name}", finite=False, floating=False)
    if stats["diverging"].dtype.kind != "b":
        raise ValueError("posterior divergence statistic must be Boolean")
    if stats["num_steps"].dtype.kind not in "iu" or np.any(stats["num_steps"] < 1):
        raise ValueError("posterior step counts must be positive integers")
    acceptance = stats["accept_prob"]
    if not np.isfinite(acceptance).all() or np.any((acceptance < 0) | (acceptance > 1)):
        raise ValueError("invalid posterior acceptance probabilities")
    for key, value in (
        ("divergences", int(stats["diverging"].sum())),
        (
            "tree_depth_saturations",
            int((stats["num_steps"] >= 2**config.max_tree_depth - 1).sum()),
        ),
    ):
        if type(diagnostic.get(key)) is not int or diagnostic[key] != value:
            raise ValueError(f"posterior {key} differs from saved statistics")
    if not np.isclose(
        diagnostic.get("mean_accept", np.nan), acceptance.mean(), rtol=1e-12, atol=1e-14
    ):
        raise ValueError("posterior mean acceptance differs from saved statistics")
    rows = diagnostic.get("parameters")
    if not isinstance(rows, list) or len(rows) != len(active) + 1:
        raise ValueError("invalid posterior coordinate diagnostic records")
    for row, name in zip(rows, [names[i] for i in active] + [("log_likelihood",)]):
        if not isinstance(row, dict) or row.get("name") != list(name):
            raise ValueError("posterior coordinate diagnostic order differs")
        for field in ("r_hat", "ess_bulk", "ess_tail"):
            if type(row.get(field)) not in (int, float):
                raise ValueError("invalid posterior coordinate diagnostic value")
    for field in ("max_rank_rhat", "min_bulk_ess", "min_tail_ess", "min_bfmi"):
        if type(diagnostic.get(field)) not in (int, float):
            raise ValueError("invalid posterior aggregate diagnostic value")
    bfmi = np.asarray(diagnostic.get("bfmi"))
    if bfmi.shape != (config.chains,) or bfmi.dtype.kind not in "fi":
        raise ValueError("invalid posterior per-chain BFMI")
    for field, values, reduction in (
        ("max_rank_rhat", [r["r_hat"] for r in rows], np.max),
        ("min_bulk_ess", [r["ess_bulk"] for r in rows], np.min),
        ("min_tail_ess", [r["ess_tail"] for r in rows], np.min),
    ):
        values = np.asarray(values)
        present = values[~np.isnan(values)]
        expected = float(reduction(present)) if len(present) else np.nan
        if np.any(values < 0) or not np.array_equal(diagnostic[field], expected, equal_nan=True):
            raise ValueError("posterior aggregate diagnostics differ from coordinate records")
    if np.any(bfmi < 0) or not np.array_equal(diagnostic["min_bfmi"], bfmi.min(), equal_nan=True):
        raise ValueError("posterior aggregate BFMI differs from chain records")
    limits = dict(
        rank_rhat=1.01,
        bulk_ess=400,
        tail_ess=200,
        bfmi=0.2,
        divergences=0,
        depth_saturations=0,
    )
    if diagnostic.get("limits") != limits:
        raise ValueError("posterior diagnostic thresholds differ")
    if diagnostic["passes"] and not (
        config.chains >= 2
        and all(
            np.isfinite([row[k] for k in ("r_hat", "ess_bulk", "ess_tail")]).all() for row in rows
        )
        and np.isfinite(bfmi).all()
        and diagnostic["max_rank_rhat"] <= 1.01
        and diagnostic["min_bulk_ess"] >= 400
        and diagnostic["min_tail_ess"] >= 200
        and diagnostic["min_bfmi"] >= 0.2
        and diagnostic["divergences"] == diagnostic["tree_depth_saturations"] == 0
        and all(np.isfinite(x).all() for x in stats.values())
    ):
        raise ValueError("posterior passing flag contradicts diagnostic records")
    metric = diagnostic.get("metric", {})
    dimension = len(active)
    axes = [config.chains] if config.chains > 1 else []
    metric_shape = axes + [dimension] * (2 if config.mass_matrix == "dense" else 1)
    if (
        metric.get("requested") != config.mass_matrix
        or metric.get("actual") != config.mass_matrix
        or metric.get("active_dimension") != dimension
        or metric.get("adapted_inverse_mass_matrix_shape") != metric_shape
    ):
        raise ValueError("posterior metric dimensions or kind differ")
    dtype = np.dtype(metric.get("adapted_inverse_mass_matrix_dtype", "invalid"))
    entries = dimension**2 if config.mass_matrix == "dense" else dimension
    if dtype.kind != "f" or metric.get("estimated_bytes_per_chain") != entries * dtype.itemsize:
        raise ValueError("posterior metric storage metadata differs")
    _sampler_identity(model, dimension)
    _adaptation_identity(diagnostic, metric_shape, dtype, axes)
    from .orientation_persistence import validate_orientation_record

    validate_orientation_record(model)


def _original_fit(model):
    original = model.training_fit_
    _fit_metadata(
        original,
        len(model.parameter_names_),
        model.phase_seconds_,
        inference="posterior",
    )
    if not _archive.same(original, _fit(model)):
        raise ValueError("posterior state differs from original public training fit")


def _validate(model, standardizer, *, rebuild=True, updated=False):
    if type(model) is not BayesianMultimodalSRM:
        raise ValueError("posterior archive requires a built-in BayesianMultimodalSRM")
    check_is_fitted(model, ["training_fit_", "parameter_draws_", "configuration_"])
    if model.inference != "posterior" or model.configuration_.get("inference") != "posterior":
        raise ValueError("posterior archive requires fitted posterior inference")
    if not updated and (
        model.targets_ is not None
        or "conditioning_mode" in model.configuration_
        or model.prediction_runs_ != model.adapter_.domains_
        or set(model.problem_.systems) != set(model.adapter_.domains_)
    ):
        raise ValueError("save the original posterior training model, not a conditioned copy")
    if not _archive.same(model._specification(), model.specification_):
        raise ValueError("posterior model specification changed after fitting")
    if (
        not _archive.same(model.search or SearchConfig(), model.search_config_)
        or not _archive.same(model.sampler or SamplerConfig(), model.sampler_config_)
        or (model.random_state is not None and model.random_state != model.seed_)
        or asdict(model.search_config_)
        != _search_for_validation(model.configuration_.get("search"))
        or asdict(model.sampler_config_)
        != _sampler_for_validation(model.configuration_.get("sampler"))
        or model.seed_ != model.configuration_.get("seed")
    ):
        raise ValueError("posterior fitted search, sampler or seed configuration changed")
    fit = _fit(model)
    _fit_metadata(fit, len(model.parameter_names_), model.phase_seconds_, inference="posterior")
    _parameters(model, model.parameter_names_, model.map_parameters_, fit)
    if updated:
        from .updated_posterior_persistence import validate_target_state

        validate_target_state(model)
    else:
        _original_fit(model)
    _diagnostics(model)
    if model.features > 1:
        keys = tuple(validate_anchors(model.problem_, model.factor_anchors))
        if getattr(model, "_factor_anchor_keys_", None) != keys:
            raise ValueError("posterior fitted factor anchors differ")
        if not _archive.same(model.factor_orientation_, metadata(keys)):
            raise ValueError("posterior factor orientation metadata differs")
    expected = copy.copy(model)
    expected._set_configuration()
    if updated:
        from .posterior_updates import update_configuration

        expected.configuration_.update(update_configuration(model.posterior_update_))
    comparable = copy.deepcopy(model.configuration_)
    comparable["sampler"] = _sampler_for_validation(comparable.get("sampler"))
    comparable["search"] = _search_for_validation(comparable.get("search"))
    if not _archive.same(expected.configuration_, comparable):
        raise ValueError("posterior saved configuration differs from reconstructed model")
    _scaler(standardizer, model.problem_)
    provenance = getattr(model, "posterior_provenance_", None)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("posterior provenance must be a mapping or None")
    _provenance(provenance)
    if rebuild and not updated:
        _training_problem(model)


def state(model, standardizer, *, updated=False):
    try:
        _validate(model, standardizer, updated=updated)
        writer = _writer(model)
        _provenance(writer)
        return dict(
            kind="posterior_training",
            constructor=model.get_params(deep=False),
            training=model.training_data_,
            training_data_sha256=_training_digest(model.training_data_),
            fit=_fit(model),
            original_training_fit=model.training_fit_,
            parameter_names=model.parameter_names_,
            phase_seconds=model.phase_seconds_,
            parameter_draws=model.parameter_draws_,
            log_likelihood_draws=model.log_likelihood_draws_,
            sample_stats=model.sample_stats_,
            sampling_diagnostics=model.sampling_diagnostics_,
            factor_anchor_keys=getattr(model, "_factor_anchor_keys_", None),
            standardizer=standardizer,
            draw_axes=["chain", "draw", "parameter"],
            parameter_coordinates="raw_physical",
            posterior_provenance=getattr(model, "posterior_provenance_", None),
            writer_provenance=writer,
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid posterior training state: {exc}") from exc


def _copy_readonly(value):
    if isinstance(value, np.ndarray):
        return readonly_array(value, value.dtype)
    if isinstance(value, dict):
        return {k: _copy_readonly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_readonly(v) for v in value)
    return copy.deepcopy(value)


def restore(saved, *, updated=False):
    try:
        expected_keys = STATE_KEYS
        expected_kind = "posterior_training"
        if updated:
            from .updated_posterior_persistence import STATE_KEYS as UPDATED_KEYS

            expected_keys = UPDATED_KEYS
            expected_kind = "posterior_updated"
        if (
            not isinstance(saved, dict)
            or set(saved) != expected_keys
            or saved["kind"] != expected_kind
        ):
            raise ValueError("invalid posterior training archive fields")
        if (
            saved["draw_axes"] != ["chain", "draw", "parameter"]
            or saved["parameter_coordinates"] != "raw_physical"
        ):
            raise ValueError("invalid posterior raw coordinate or axis identity")
        if saved["training_data_sha256"] != _training_digest(saved["training"]):
            raise ValueError("posterior native training data digest differs")
        from .persistence import _restore_search_default

        constructor = _restore_search_default(_copy_readonly(saved["constructor"]), saved["fit"])
        model = BayesianMultimodalSRM(**constructor)
        if model.inference != "posterior":
            raise ValueError("posterior archive requires posterior constructor")
        model._config()
        fit = saved["fit"]
        _fit_metadata(
            fit,
            len(saved["parameter_names"]),
            saved["phase_seconds"],
            inference="posterior",
        )
        seed = fit["configuration"]["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid original posterior training seed")
        model.seed_ = seed
        if updated:
            from .posterior_updates import content_id, rebuild_target
            from .updated_posterior_persistence import fit_identity

            update = saved["posterior_update"]
            if content_id(update) != saved["posterior_update_sha256"]:
                raise ValueError("posterior update evidence digest differs")
            if fit_identity(saved) != saved["posterior_fit_sha256"]:
                raise ValueError("posterior update fit/draw identity differs")
            model.adapter_, model.problem_, _ = rebuild_target(update)
            model.posterior_update_ = _copy_readonly(update)
        else:
            model.adapter_, model.problem_ = _prepare(model, saved["training"])
        model.training_data_ = model.adapter_._training_data
        model.prediction_runs_, model.targets_ = dict(model.adapter_.domains_), None
        if updated:
            if not _archive.same(model.training_data_, saved["training"]):
                raise ValueError("posterior update training data differs from evidence union")
            model.prediction_runs_ = _copy_readonly(saved["prediction_runs"])
            model.targets_ = _copy_readonly(saved["targets"])
        model.specification_ = model._specification()
        model.parameter_names_ = _copy_readonly(saved["parameter_names"])
        model.map_parameters_ = readonly_array(fit["map_parameters"])
        model.map_diagnostics_ = _copy_readonly(fit["map_diagnostics"])
        model.restart_diagnostics_ = _copy_readonly(fit["restart_diagnostics"])
        model.objective_ = fit["objective"]
        # Preserve dtypes and axes rather than silently coercing damaged records.
        model.parameter_draws_ = readonly_array(
            saved["parameter_draws"], saved["parameter_draws"].dtype
        )
        model.log_likelihood_draws_ = readonly_array(
            saved["log_likelihood_draws"], saved["log_likelihood_draws"].dtype
        )
        model.sample_stats_ = {
            k: readonly_array(v, v.dtype) for k, v in saved["sample_stats"].items()
        }
        model.sampling_diagnostics_ = _copy_readonly(saved["sampling_diagnostics"])
        model.configuration_ = _copy_readonly(fit["configuration"])
        model.training_fit_ = _copy_readonly(saved["original_training_fit"])
        model.phase_seconds_ = _copy_readonly(saved["phase_seconds"])
        if model.features > 1:
            model._factor_anchor_keys_ = saved["factor_anchor_keys"]
            model.factor_orientation_ = _copy_readonly(model.configuration_["factor_orientation"])
        elif saved["factor_anchor_keys"] is not None:
            raise ValueError("scalar posterior cannot have factor anchors")
        model.posterior_provenance_ = _copy_readonly(saved["posterior_provenance"])
        model.archive_writer_provenance_ = _copy_readonly(saved["writer_provenance"])
        _provenance(model.archive_writer_provenance_)
        if (
            not isinstance(model.archive_writer_provenance_, dict)
            or model.archive_writer_provenance_.get("role") != "archive_writer_not_training"
        ):
            raise ValueError("invalid posterior archive writer provenance")
        standardizer = saved["standardizer"]
        _validate(model, standardizer, rebuild=False, updated=updated)
        return model, standardizer
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid posterior training archive: {exc}") from exc
