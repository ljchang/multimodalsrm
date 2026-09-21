"""Schema 4 for explicitly reconstructed joint posterior update targets."""

from . import _archive, posterior_persistence
from .posterior_updates import (
    adapter_state,
    content_id,
    rebuild_target,
    validate_update_target,
)

STATE_KEYS = posterior_persistence.STATE_KEYS | {
    "posterior_update",
    "posterior_update_sha256",
    "posterior_fit_sha256",
    "prediction_runs",
    "targets",
}


def fit_identity(saved):
    """Bind numeric arrays to the retained fit and its diagnostic records."""
    return content_id(
        {
            key: saved[key]
            for key in (
                "fit",
                "original_training_fit",
                "parameter_names",
                "phase_seconds",
                "parameter_draws",
                "log_likelihood_draws",
                "sample_stats",
                "sampling_diagnostics",
                "factor_anchor_keys",
            )
        }
    )


def validate_target_state(model):
    """Bind saved fit, query scope and source to the same reconstructed target."""
    update = model.posterior_update_
    validate_update_target(model.problem_)
    if not _archive.same(update, model.problem_._posterior_update):
        raise ValueError("posterior update evidence differs from fitted target")
    adapter, problem, prediction_runs = rebuild_target(update)
    if (
        not _archive.same(update["specification"], model.specification_)
        or not _archive.same(update["priors"], model.priors)
        or update["linear_algebra"] != model.linear_algebra
        or not _archive.same(model.training_data_, adapter._training_data)
        or not _archive.same(adapter_state(model.adapter_), adapter_state(adapter))
        or not _archive.same(model.prediction_runs_, prediction_runs)
        or not _archive.same(model.targets_, update["targets"])
        or model.parameter_names_ != problem.names
        or getattr(model, "_factor_anchor_keys_", None) != update["reference"]["factor_anchor_keys"]
    ):
        raise ValueError("posterior update model/query state differs from evidence target")
    expected_fit = (
        posterior_persistence._fit(model)
        if update["kind"] == "calibration"
        else update["reference"]["training_fit"]
    )
    if not _archive.same(model.training_fit_, expected_fit):
        raise ValueError("posterior update reference fit changed")


def state(model, standardizer):
    saved = posterior_persistence.state(model, standardizer, updated=True)
    saved.update(
        kind="posterior_updated",
        posterior_update=model.posterior_update_,
        posterior_update_sha256=content_id(model.posterior_update_),
        prediction_runs=model.prediction_runs_,
        targets=model.targets_,
    )
    saved["posterior_fit_sha256"] = fit_identity(saved)
    from pathlib import Path

    folder = Path(__file__).parent
    saved["writer_provenance"]["source_sha256"].update(
        {
            name: _archive.digest(folder / name)
            for name in (
                "posterior_updates.py",
                "posterior_participants.py",
                "updated_posterior_persistence.py",
            )
        }
    )
    return saved


def restore(saved):
    return posterior_persistence.restore(saved, updated=True)
