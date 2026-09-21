"""Parallel execution must preserve fit identities and validation boundaries."""

import copy
import os
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import (
    Gaussian,
    MultimodalSRM,
    Response,
    TimeSeries,
    cross_validate,
    nested_cross_validate,
    select_model,
)


def data():
    t = np.arange(48.0)
    return {
        s: {
            r: {
                "a": TimeSeries(
                    np.column_stack([np.sin(t / 4) + offset, np.cos(t / 5) - offset]), t
                ),
                "b": TimeSeries((np.sin(t / 4 + 0.3) + offset)[:, None], t),
            }
            for r, offset in (("first", 0.0), ("second", 12.0), ("third", -5.0))
        }
        for s in ("s1", "s2")
    }


def model(**kwargs):
    return MultimodalSRM(
        features=2,
        latent_dt=2.0,
        max_iter=3,
        kernel_max_iter=2,
        random_state=23,
        n_init=3,
        init=kwargs.pop("init", "hybrid"),
        responses={
            m: Response(
                Gaussian(0.4),
                pooling="shared",
                fixed={"width": 0.4},
                bounds={"lag": (-0.5, 0.5)},
            )
            for m in ("a", "b")
        },
        **kwargs,
    )


def fit(estimator, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(x)


def same_fit(a, b):
    assert a.best_restart_ == b.best_restart_
    assert a.converged_ == b.converged_
    np.testing.assert_allclose(a.objective_history_, b.objective_history_, rtol=1e-12, atol=1e-13)
    assert [r["seed"] for r in a.restart_diagnostics_] == [
        r["seed"] for r in b.restart_diagnostics_
    ]
    assert [r["initialization"] for r in a.restart_diagnostics_] == [
        r["initialization"] for r in b.restart_diagnostics_
    ]
    for s in a.loadings_:
        for m in a.loadings_[s]:
            np.testing.assert_allclose(a.loadings_[s][m], b.loadings_[s][m], rtol=1e-11, atol=1e-12)
            assert a.subject_kernels_[s][m].parameters == pytest.approx(
                b.subject_kernels_[s][m].parameters, rel=1e-11, abs=1e-12
            )


@pytest.mark.parametrize("init", ["random", "spectral", "hybrid"])
def test_parallel_restarts_preserve_seeds_initialization_selection_and_maps(init):
    x = data()
    serial = fit(model(n_jobs=1, init=init), x)
    parallel = fit(model(n_jobs=2, init=init), x)
    same_fit(serial, parallel)
    assert [r["seed"] for r in parallel.restart_diagnostics_] == [
        74293843,
        726266598,
        1919689768,
    ]
    assert len({r["worker_pid"] for r in parallel.restart_diagnostics_}) == 2
    assert os.getpid() not in {r["worker_pid"] for r in parallel.restart_diagnostics_}
    assert parallel.execution_["restart_workers"] == 2


def test_old_pickles_default_to_serial_and_clone_preserves_parallel_option():
    state = vars(model()).copy()
    state.pop("n_jobs", None)
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert clone(restored).n_jobs == 1
    assert clone(model(n_jobs=3)).n_jobs == 3


@pytest.mark.parametrize("bad", [0, -2, True, 1.5])
def test_invalid_worker_counts_fail_before_fitting(bad):
    with pytest.raises(ValueError, match="n_jobs"):
        fit(model(n_jobs=bad), data())


def test_parallel_cv_preserves_fold_order_and_training_only_preprocessing():
    x = data()
    template = model(n_jobs=2)
    serial = cross_validate(template, x, targets={"s1": ["a"]}, n_jobs=1)
    parallel = cross_validate(template, x, targets={"s1": ["a"]}, n_jobs=2)
    assert not hasattr(template, "loadings_")
    assert serial.folds == parallel.folds
    for a, b, fold in zip(serial.models, parallel.models, parallel.folds):
        same_fit(a, b)
        assert b.execution_["restart_workers"] == 1
        expected = np.concatenate([x["s1"][r]["a"].values for r in fold["train_runs"]]).mean(0)
        np.testing.assert_allclose(b.preprocessing_["s1"]["a"]["mean"], expected)
    for a, b in zip(serial.scores, parallel.scores):
        for metric in ("r2", "mse", "correlation"):
            assert a[metric] == pytest.approx(b[metric], abs=1e-11)


def test_parallel_selection_retains_failed_candidates_and_common_support():
    candidates = {
        "good": model(n_jobs=2),
        "bad": model(n_jobs=2).set_params(loading_ridge=-1),
    }
    serial = select_model(candidates, data(), targets={"s1": ["a"]}, n_jobs=1)
    parallel = select_model(candidates, data(), targets={"s1": ["a"]}, n_jobs=2)
    assert serial.best_name == parallel.best_name == "good"
    assert parallel.status == "ok"
    bad = [r for r in parallel.folds if r["candidate"] == "bad"]
    assert [r["fold"] for r in bad] == [0, 1, 2]
    assert all(r["status"] == "fit_failed" for r in bad)
    for a, b in zip(serial.predictions, parallel.predictions):
        np.testing.assert_array_equal(a["common_mask"], b["common_mask"])
        np.testing.assert_allclose(a["values"], b["values"], rtol=1e-11, atol=1e-11, equal_nan=True)


def test_parallel_nested_cv_never_learns_from_outer_test_targets():
    x = data()
    candidates = {"one": model(n_jobs=2).set_params(n_init=1)}
    original = nested_cross_validate(candidates, x, targets={"s1": ["a"]}, n_jobs=2)
    poisoned = copy.deepcopy(x)
    ts = x["s1"]["first"]["a"]
    poisoned["s1"]["first"]["a"] = TimeSeries(ts.values * 100 + 300, ts.times)
    changed = nested_cross_validate(candidates, poisoned, targets={"s1": ["a"]}, n_jobs=2)
    assert original.folds[0]["test_runs"] == ["first"]
    same_fit(original.models[0], changed.models[0])
    assert original.selections[0].best_name == changed.selections[0].best_name
    for a, b in zip(original.selections[0].scores, changed.selections[0].scores):
        assert a == b
    for a, b in zip(original.predictions, changed.predictions):
        if a["fold"] == 0:
            np.testing.assert_array_equal(a["values"], b["values"])


def test_parallel_cv_forwards_warnings_to_parent_and_honors_error_filters():
    template = model().set_params(n_init=1, max_iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cross_validate(template, data(), targets={"s1": ["a"]}, n_jobs=2)
    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(convergence) == 3
    assert all(w.filename.endswith("validation.py") for w in convergence)
    assert all(not fitted.converged_ for fitted in result.models)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning):
            cross_validate(template, data(), targets={"s1": ["a"]}, n_jobs=2)


def test_parallel_workers_return_fit_state_without_full_observation_arrays():
    # A local helper travels by value to isolated workers, which need only the
    # installed package and do not import this repository's test package.
    class InspectRestartPayload(MultimodalSRM):
        def _select_fit(self, candidates):
            self.returned_observation_blocks_ = [candidate[-1] for candidate in candidates]
            return super()._select_fit(candidates)

    fitted = fit(InspectRestartPayload(**model(n_jobs=2).get_params()), data())
    assert fitted.returned_observation_blocks_ == [None, None, None]
    assert fitted.observation_blocks_
    assert all(b.H.shape[0] == len(b.values) for b in fitted.observation_blocks_)
