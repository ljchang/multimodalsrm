"""Dense polishing must not turn voxel-scale MAP into a quadratic allocation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


def test_voxel_scale_polishing_skips_dense_work_and_preserves_estimate(monkeypatch):
    from multimodalsrm.bayesian import polishing
    from multimodalsrm.bayesian.fitting import SearchConfig

    n = 72471
    probes = []

    def evaluate(x):
        probes.append(1)
        return float(x @ x / 2), x.copy()

    original_empty = np.empty

    def bounded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and min(shape) > 256:
            raise AssertionError("voxel-scale dense matrix allocation attempted")
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(polishing.np, "empty", bounded_empty)
    problem = SimpleNamespace(
        value_gradient=evaluate,
        names=[("loading", i) for i in range(n)],
        bounds=[(-np.inf, np.inf)] * n,
    )
    record = dict(
        parameters=[1.0] * n,
        objective=n / 2,
        physical_projected_gradient=1.0,
        meets_gradient_tolerance=False,
    )
    original = deepcopy(record)
    report = polishing.polish(problem, record, SearchConfig(), 3)
    assert record == original
    assert report["accepted_steps"] == 0
    assert report["skipped"] == "dense_parameter_limit"
    assert report["free_parameters"] == n
    assert report["dense_hessian_bytes"] == 42016366728
    assert len(probes) == 1


def test_polishing_limit_counts_free_coordinates_not_active_bounds():
    from multimodalsrm.bayesian.fitting import SearchConfig
    from multimodalsrm.bayesian.polishing import polish

    n = 1024

    def evaluate(x):
        gradient = np.ones(n)
        gradient[-1] = x[-1]
        return float(x[:-1].sum() + x[-1] ** 2 / 2), gradient

    problem = SimpleNamespace(
        value_gradient=evaluate,
        names=[("loading", i) for i in range(n)],
        bounds=[(0.0, np.inf)] * (n - 1) + [(-np.inf, np.inf)],
    )
    record = dict(
        parameters=[0.0] * (n - 1) + [0.02],
        objective=0.0002,
        physical_projected_gradient=0.02,
        meets_gradient_tolerance=False,
    )
    report = polish(problem, record, SearchConfig(), 2)
    assert record["meets_gradient_tolerance"]
    assert report["accepted_steps"] == 1
    np.testing.assert_allclose(record["parameters"], 0.0, atol=1e-9)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_polishing_limit_rejects_invalid_budgets(limit):
    from multimodalsrm.bayesian.fitting import SearchConfig

    with pytest.raises(ValueError, match="polish_max_parameters"):
        SearchConfig(polish_max_parameters=limit)


def test_iteration_checkpoints_hold_physical_parameters_without_changing_search():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    from .test_bayesian_problem import problem_fixture

    problem, _, _ = problem_fixture(gaussian=True)
    config = SearchConfig(starts=1, maxiter=2)
    expected, _ = search(problem, config, 314)
    events = []
    actual, _ = search(problem, config, 314, progress=events.append)
    iterations = [event["record"] for event in events if event["phase"] == "iteration"]
    assert iterations, "no in-flight parameters were checkpointed"
    point = iterations[0]
    assert point["iteration"] == 1
    np.testing.assert_allclose(
        problem.objective(np.asarray(point["parameters"])), point["objective"], rtol=1e-12
    )
    np.testing.assert_allclose(actual["parameters"], expected["parameters"], rtol=0, atol=0)
    assert actual["meets_gradient_tolerance"] == expected["meets_gradient_tolerance"]


def test_iteration_checkpoint_failure_propagates_instead_of_marking_restart_failed():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    from .test_bayesian_problem import problem_fixture

    problem, _, _ = problem_fixture(gaussian=True)

    def checkpoint(event):
        if event["phase"] == "iteration":
            raise ValueError("checkpoint storage unavailable")

    with pytest.raises(ValueError, match="checkpoint storage unavailable"):
        search(problem, SearchConfig(starts=1, maxiter=2), 314, progress=checkpoint)


def test_refinement_checkpoints_estimate_before_entering_polishing():
    from multimodalsrm.bayesian.fitting import SearchConfig, _refine

    curvature = np.array([1e8, 1.0])

    def evaluate(x):
        return float(1000 + 0.5 * np.sum(curvature * x**2)), curvature * x

    problem = SimpleNamespace(
        value_gradient=evaluate,
        to_unconstrained=lambda x: np.asarray(x),
        from_unconstrained=lambda x: (np.asarray(x), 0.0),
        names=[("stiff",), ("soft",)],
        bounds=[(-np.inf, np.inf)] * 2,
    )
    record = dict(
        start=0,
        parameters=[1e-10, 0.002],
        objective=1000.000002,
        elapsed_seconds=0.0,
        physical_projected_gradient=0.01,
        meets_gradient_tolerance=False,
    )
    events = []

    def checkpoint(event):
        events.append(event)
        if event["phase"] == "refinement_optimized":
            raise RuntimeError("simulated interruption before polishing")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _refine(problem, record, SearchConfig(refine_maxiter=100), evaluate, progress=checkpoint)
    saved = events[-1]["record"]
    assert np.isfinite(saved["parameters"]).all()
    np.testing.assert_allclose(evaluate(np.asarray(saved["parameters"]))[0], saved["objective"])
    assert not saved["meets_gradient_tolerance"]
