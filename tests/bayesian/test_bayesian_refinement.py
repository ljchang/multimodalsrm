"""Refinement keeps the same physical objective and records its original fit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .test_bayesian_problem import api, problem_fixture


@pytest.mark.parametrize(
    "boundary,optimum",
    [(0.0, 0.0), (0.0, 0.3), (1.0, 1.0), (1.0, 0.7)],
)
@pytest.mark.parametrize("refine_budget", [3, 200])
def test_refinement_uses_unspent_budget_after_boundary_conversion_failure(
    boundary, optimum, refine_budget, monkeypatch
):
    from types import SimpleNamespace

    from multimodalsrm.bayesian import polishing
    from multimodalsrm.bayesian.fitting import SearchConfig, _refine

    original_polish = polishing.polish

    def capped_polish(problem, record, config, max_steps):
        assert max_steps == 3
        return original_polish(problem, record, config, max_steps)

    monkeypatch.setattr(polishing, "polish", capped_polish)

    def physical(x):
        assert 0 <= x[0] <= 1
        # At the first and last cases the bound is a genuine constrained optimum.
        target = -1.0 if optimum == 0 else 2.0 if optimum == 1 else optimum
        delta = x - np.array([target, 0.0])
        return float(0.5 * delta @ delta), delta

    def inverse(x):
        if not 0 < x[0] < 1:
            raise ValueError("physical parameters must lie strictly inside prior support")
        return np.array([np.log(x[0] / (1 - x[0])), x[1]])

    def transformed(z):
        pytest.fail("conversion failed before the optimizer could consume iterations")

    problem = SimpleNamespace(
        value_gradient=physical,
        to_unconstrained=inverse,
        from_unconstrained=lambda z: (np.array([1 / (1 + np.exp(-z[0])), z[1]]), 0),
        names=[("bounded",), ("free",)],
        bounds=[(0.0, 1.0), (-np.inf, np.inf)],
    )
    x = np.array([boundary, 0.02])
    record = dict(
        parameters=x.tolist(),
        objective=physical(x)[0],
        elapsed_seconds=0.0,
        physical_projected_gradient=max(0.02, abs(boundary - optimum)),
        meets_gradient_tolerance=False,
    )
    _refine(problem, record, SearchConfig(refine_maxiter=refine_budget), transformed)
    assert record["meets_gradient_tolerance"]
    assert_allclose(record["parameters"], [optimum, 0.0], atol=1e-8)
    assert record["refinement"]["iterations"] <= 3
    assert record["refinement"]["failure_phase"] == "initialization"
    assert record["refinement"]["polish"]["accepted_steps"] > 0
    assert record["objective"] <= record["pre_refinement"]["objective"]


def test_optional_refinement_improves_real_map_and_retains_original():
    from multimodalsrm.bayesian.fitting import search

    b = api()
    p, _, _ = problem_fixture(gaussian=True)
    original, _ = search(p, b.SearchConfig(starts=1, maxiter=2), 413)
    assert not original["meets_gradient_tolerance"]
    best, records = search(p, b.SearchConfig(starts=1, maxiter=2, refine_maxiter=200), 413)
    assert len(records) == 1
    assert best["objective"] < original["objective"]
    assert best["physical_projected_gradient"] < original["physical_projected_gradient"]
    assert best["meets_gradient_tolerance"]
    assert best["refinement"]["accepted"]
    assert_allclose(best["pre_refinement"]["objective"], original["objective"], atol=1e-10)
    assert_allclose(best["objective"], p.objective(best["parameters"]), atol=1e-10)


@pytest.mark.parametrize("invalid", [-1, 0.5, True])
def test_refinement_budget_validation(invalid):
    b = api()
    with pytest.raises(ValueError, match="refine_maxiter"):
        b.SearchConfig(refine_maxiter=invalid)
    assert b.SearchConfig().refine_maxiter == 0


def test_refinement_retains_original_if_budget_cannot_converge():
    from multimodalsrm.bayesian.fitting import search

    b = api()
    p, _, _ = problem_fixture(gaussian=True)
    best, records = search(p, b.SearchConfig(starts=1, maxiter=2, refine_maxiter=1), 413)
    assert np.isfinite(best["objective"])
    assert best["objective"] <= best["pre_refinement"]["objective"]
    assert not best["meets_gradient_tolerance"]
    assert len(records) == 1 and best["refinement"]["iterations"] <= 1


def test_refinement_polishes_stiff_quadratic_after_objective_stopping():
    from types import SimpleNamespace

    from multimodalsrm.bayesian.fitting import SearchConfig, _refine

    curvature = np.diag([1e8, 1.0])

    def evaluate(x):
        return float(1000.0 + 0.5 * x @ curvature @ x), curvature @ x

    problem = SimpleNamespace(
        value_gradient=evaluate,
        to_unconstrained=lambda x: np.asarray(x),
        from_unconstrained=lambda x: (np.asarray(x), 0.0),
        names=[("stiff",), ("soft",)],
        bounds=[(-np.inf, np.inf)] * 2,
    )
    x = np.array([1e-10, 0.002])
    record = dict(
        parameters=x.tolist(),
        objective=evaluate(x)[0],
        elapsed_seconds=0.0,
        physical_projected_gradient=0.01,
        meets_gradient_tolerance=False,
    )
    _refine(problem, record, SearchConfig(refine_maxiter=100), evaluate)
    assert record["meets_gradient_tolerance"]
    assert record["refinement"]["polish"]["steps"] >= 1
    assert_allclose(record["parameters"], np.zeros(2), atol=1e-6)
    for field in ("optimizer_success", "status", "message", "boundary_parameters"):
        assert record["refinement"][field] == record[field]


def test_polishing_fixes_active_bound_and_probes_free_coordinates_only():
    from types import SimpleNamespace

    from multimodalsrm.bayesian.fitting import SearchConfig
    from multimodalsrm.bayesian.polishing import polish

    probes = []

    def evaluate(x):
        probes.append(x.copy())
        assert x[0] == 0.0
        return 0.5 * ((x[0] + 1) ** 2 + x[1] ** 2), np.array([x[0] + 1, x[1]])

    problem = SimpleNamespace(
        value_gradient=evaluate,
        bounds=[(0, np.inf), (-np.inf, np.inf)],
        names=[("bound",), ("free",)],
    )
    record = dict(
        parameters=[0.0, 0.02],
        objective=evaluate(np.array([0.0, 0.02]))[0],
        physical_projected_gradient=0.02,
        meets_gradient_tolerance=False,
    )
    report = polish(problem, record, SearchConfig(), 2)
    assert record["meets_gradient_tolerance"]
    assert report["accepted_steps"] == 1
    assert report["attempts"][0]["free_coordinates"] == [1]
    assert_allclose(record["parameters"], [0.0, 0.0], atol=1e-8)
    assert all(p[0] == 0.0 for p in probes)


def test_polishing_preserves_small_positive_curvature_in_mixed_units():
    from types import SimpleNamespace

    from multimodalsrm.bayesian.fitting import SearchConfig
    from multimodalsrm.bayesian.polishing import polish

    # Positive definite with a known minimum at zero. A physical eigenvalue
    # floor tied to 1e12 replaces the soft curvature (~4) by 100 and stalls it.
    curvature = np.array([[1e12, 1e5], [1e5, 4.0]])

    def evaluate(x):
        return float(0.5 * x @ curvature @ x), curvature @ x

    problem = SimpleNamespace(
        value_gradient=evaluate,
        bounds=[(-np.inf, np.inf)] * 2,
        names=[("stiff",), ("soft",)],
    )
    x = np.array([1e-9, 0.02])
    record = dict(
        parameters=x.tolist(),
        objective=evaluate(x)[0],
        physical_projected_gradient=float(np.max(np.abs(evaluate(x)[1]))),
        meets_gradient_tolerance=False,
    )
    report = polish(problem, record, SearchConfig(), 3)
    assert record["meets_gradient_tolerance"], record["physical_projected_gradient"]
    assert record["objective"] < 1e-16
    assert_allclose(record["parameters"], [0.0, 0.0], atol=1e-8)
    assert report["steps"] <= 3


def test_nonfinite_polishing_probe_preserves_original_point():
    from types import SimpleNamespace

    from multimodalsrm.bayesian.fitting import SearchConfig
    from multimodalsrm.bayesian.polishing import polish

    x = np.array([0.01])

    def evaluate(point):
        if abs(point[0] - x[0]) > 1e-8:
            return np.nan, np.array([np.nan])
        return float(point[0] ** 2), 2 * point

    problem = SimpleNamespace(value_gradient=evaluate, bounds=[(-np.inf, np.inf)], names=[("x",)])
    record = dict(
        parameters=x.tolist(),
        objective=evaluate(x)[0],
        physical_projected_gradient=0.02,
        meets_gradient_tolerance=False,
    )
    original = dict(record)
    report = polish(problem, record, SearchConfig(), 2)
    assert record == original
    assert report["accepted_steps"] == 0
    assert "nonfinite" in report["attempts"][0]["failure"]


@pytest.mark.parametrize("failure", ["nonfinite_result", "optimizer_exception"])
def test_failed_refinement_cannot_reuse_consumed_or_unknown_budget(monkeypatch, failure):
    from types import SimpleNamespace

    from scipy.optimize import OptimizeResult

    from multimodalsrm.bayesian import fitting

    def evaluate(x):
        return float(x @ x), 2 * x

    problem = SimpleNamespace(
        value_gradient=evaluate,
        to_unconstrained=lambda x: np.asarray(x),
        from_unconstrained=lambda x: (np.asarray(x), 0),
        names=[("x",)],
        bounds=[(-np.inf, np.inf)],
    )

    def fail(*args, **kwargs):
        if failure == "optimizer_exception":
            raise ValueError("optimizer failed after unknown work")
        return OptimizeResult(
            x=np.array([np.nan]), nit=1, success=False, status=1, message="failed"
        )

    monkeypatch.setattr(fitting, "minimize", fail)
    record = dict(
        parameters=[0.01],
        objective=0.0001,
        elapsed_seconds=0.0,
        physical_projected_gradient=0.02,
        meets_gradient_tolerance=False,
    )
    fitting._refine(problem, record, fitting.SearchConfig(refine_maxiter=1), evaluate)
    assert "polish" not in record["refinement"]
    assert not record["refinement"]["accepted"]
    assert record["parameters"] == [0.01]
    assert record["refinement"]["iterations"] == (1 if failure == "nonfinite_result" else None)
