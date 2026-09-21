"""MAP scheduling must preserve starts, selection, callbacks and qualification."""

import threading
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .test_bayesian_multifactor import fixture


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_scalar_parallel_map_preserves_positive_anchor_and_physical_density(algebra):
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    from .test_bayesian_problem import api, problem_fixture

    original, adapter, _ = problem_fixture(True)
    p = api().BayesianProblem(
        adapter, original.priors, anchor=original.anchor, linear_algebra=algebra
    )
    config = SearchConfig(starts=2, maxiter=15)
    serial, _ = search(p, config, 24)
    parallel, records = search(p, replace(config, n_jobs=2), 24)
    assert_allclose(parallel["parameters"], serial["parameters"], atol=1e-10, rtol=1e-10)
    assert parallel["objective"] == serial["objective"]
    assert all(r["parameters"][p.indices[("loading", *p.anchor)]] > 0 for r in records)


@pytest.mark.parametrize("learned", [False, True])
def test_parallel_search_matches_serial_and_runs_callbacks_on_caller(learned, monkeypatch):
    from multimodalsrm.bayesian import fitting

    p, _, _ = fixture(algebra="grouped", gaussian=learned)
    config = fitting.SearchConfig(starts=3, maxiter=12, n_jobs=1)
    serial_best, serial = fitting.search(p, config, 71)
    caller = threading.get_ident()
    worker_ids = set()
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    minimize = fitting.minimize
    calls = 0

    def concurrent_minimize(*args, **kwargs):
        nonlocal calls
        with lock:
            worker_ids.add(threading.get_ident())
            calls += 1
            first_pair = calls <= 2
        # Only the first two tasks rendezvous; a third reuses either worker.
        if first_pair:
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                raise AssertionError("MAP restarts did not execute concurrently")
        return minimize(*args, **kwargs)

    monkeypatch.setattr(fitting, "minimize", concurrent_minimize)
    events = []

    def progress(event):
        assert threading.get_ident() == caller
        events.append((event["phase"], event["record"]["start"]))
        # A consumer must not be able to mutate the retained fit records.
        event["record"]["parameters"] = [999.0]

    best, parallel = fitting.search(p, replace(config, n_jobs=2), 71, progress=progress)
    assert caller not in worker_ids and len(worker_ids) == 2
    assert [r["start"] for r in parallel] == [0, 1, 2]
    assert best["start"] == serial_best["start"]
    for expected, actual in zip(serial, parallel):
        assert_array_equal(actual["initial_parameters"], expected["initial_parameters"])
        assert_allclose(actual["parameters"], expected["parameters"], atol=1e-10, rtol=1e-10)
        for key in ("objective", "physical_projected_gradient"):
            assert_allclose(actual[key], expected[key], atol=1e-9, rtol=1e-10)
        for key in ("status", "iterations", "meets_gradient_tolerance"):
            assert actual[key] == expected[key]
        phases = [phase for phase, start in events if start == actual["start"]]
        assert phases[0] == "started" and phases[-1] == "finished"
        assert "iteration" in phases


def test_callback_failure_propagates_and_joins_parallel_workers():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    p, _, _ = fixture(algebra="grouped")
    failure = OSError("cannot save checkpoint")

    def progress(event):
        if event["phase"] == "iteration":
            raise failure

    with pytest.raises(OSError) as caught:
        search(p, SearchConfig(starts=4, maxiter=100, n_jobs=2), 7, progress=progress)
    assert caught.value is failure
    assert not any(t.name.startswith("gp-map") for t in threading.enumerate())


def test_unexpected_worker_failure_propagates_and_joins_workers():
    from multimodalsrm.bayesian.map_parallel import ordered_restarts

    failure = RuntimeError("worker failed unexpectedly")
    barrier = threading.Barrier(2)

    def run_start(index, point, emit, check_cancelled):
        barrier.wait(timeout=10)
        if index == 0:
            raise failure
        return point

    with pytest.raises(RuntimeError) as caught:
        ordered_restarts(run_start, [1, 2], 2, None)
    assert caught.value is failure
    assert not any(t.name.startswith("gp-map") for t in threading.enumerate())


def test_failed_start_retained_and_equal_objectives_select_first_start(monkeypatch):
    from multimodalsrm.bayesian import fitting

    p, _, _ = fixture(algebra="grouped")
    points = np.stack([p.initial, p.initial, np.full_like(p.initial, np.nan)])
    monkeypatch.setattr(fitting, "initial_points", lambda *args: points.copy())
    best, records = fitting.search(p, fitting.SearchConfig(starts=3, maxiter=3, n_jobs=2), 2)
    assert [r["start"] for r in records] == [0, 1, 2]
    assert best["start"] == 0
    assert records[0]["objective"] == records[1]["objective"]
    assert records[2]["objective"] is None
    assert "failure" in records[2]


def test_parallel_refinement_matches_serial_and_stays_on_caller():
    from multimodalsrm.bayesian import fitting

    p, _, _ = fixture(algebra="grouped")
    config = fitting.SearchConfig(starts=2, maxiter=3, refine_maxiter=3, n_jobs=1)
    serial, _ = fitting.search(p, config, 14)
    phases = []
    caller = threading.get_ident()

    def progress(event):
        assert threading.get_ident() == caller
        phases.append(event["phase"])

    parallel, _ = fitting.search(p, replace(config, n_jobs=2), 14, progress=progress)
    assert phases[-1] == "refined"
    assert phases.index("refining") < phases.index("refinement_iteration")
    assert phases.index("refinement_iteration") < phases.index("refinement_optimized")
    assert parallel["refinement"]["accepted"] == serial["refinement"]["accepted"]
    assert_allclose(parallel["parameters"], serial["parameters"], atol=1e-10, rtol=1e-10)
    assert parallel["meets_gradient_tolerance"] == serial["meets_gradient_tolerance"]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(True)])
def test_invalid_worker_counts_rejected(value):
    from multimodalsrm.bayesian.fitting import SearchConfig

    with pytest.raises(ValueError, match="n_jobs"):
        SearchConfig(n_jobs=value)


@pytest.mark.parametrize("missing", [{"n_jobs"}, {"n_jobs", "polish_max_parameters"}])
def test_legacy_search_configuration_keeps_serial_default(missing):
    from dataclasses import asdict

    from multimodalsrm.bayesian import SearchConfig, _archive
    from multimodalsrm.bayesian.persistence import _search_for_validation

    config = SearchConfig(starts=2)
    encoded = _archive.encode(config, {})
    for field in missing:
        del encoded["fields"][field]
    restored = _archive.decode(encoded, {}, set())
    assert restored.n_jobs == 1
    legacy = {k: v for k, v in asdict(config).items() if k not in missing}
    assert _search_for_validation(legacy) == asdict(config)
    assert not missing.intersection(legacy)
