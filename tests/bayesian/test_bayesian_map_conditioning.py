"""Conditioning changes solver coordinates, never the MAP density or its gate."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .test_bayesian_multifactor import fixture


def test_conditioning_is_opt_in_and_legacy_archives_keep_original_path():
    from multimodalsrm.bayesian import _archive
    from multimodalsrm.bayesian.fitting import SearchConfig
    from multimodalsrm.bayesian.persistence import _search_for_validation

    config = SearchConfig()
    assert config.conditioning == "none"
    for invalid in (None, True, "auto", "scaled"):
        with pytest.raises(ValueError, match="conditioning"):
            SearchConfig(conditioning=invalid)
    encoded = _archive.encode(config, {})
    del encoded["fields"]["conditioning"]
    assert _archive.decode(encoded, {}, set()).conditioning == "none"
    legacy = asdict(config)
    del legacy["conditioning"]
    assert _search_for_validation(legacy) == asdict(config)
    assert "conditioning" not in legacy


def test_information_scales_use_observed_native_feature_counts():
    from multimodalsrm.bayesian.map_conditioning import InformationScaling

    p, x, _ = fixture(algebra="grouped", gaussian=False)
    scaling = InformationScaling(p)
    scales, report = scaling.at(x)
    # Count independently from the observation keys, including the masked voxel
    # and the participant with no auxiliary modality.
    expected = np.array(
        [sum(system.keys.count(key) for system in p.systems.values()) for key in p.keys]
    )
    assert_array_equal(scaling.feature_counts, expected)
    assert expected.min() < expected.max()
    assert np.isfinite(scales).all() and (scales > 0).all()
    for key, count in zip(p.keys, expected):
        j = p.indices[("offset", *key)]
        variance = x[p.indices[("noise", *key[:2])]]
        precision = count / variance + 1 / p.parameter_priors[j].scale ** 2
        assert_allclose(scales[j], 1 / np.sqrt(precision), rtol=1e-14)
    assert report["gradient_evaluations"] == 0


def test_filter_curvature_probes_stay_in_declared_bounds():
    from multimodalsrm.bayesian.map_conditioning import InformationScaling

    p, x, _ = fixture(algebra="grouped")
    indices = [i for i, name in enumerate(p.names) if name[0] == "filter"]
    x[indices] = [p.bounds[i][0] for i in indices]
    calls = []
    original = p.value_gradient

    def checked(point):
        assert all(p.bounds[i][0] <= point[i] <= p.bounds[i][1] for i in indices)
        calls.append(point.copy())
        return original(point)

    p.value_gradient = checked
    scales, report = InformationScaling(p).at(x)
    assert np.isfinite(scales).all() and (scales > 0).all()
    assert report["gradient_evaluations"] == len(calls) == 2 * len(indices)


@pytest.mark.parametrize("learned", [False, True])
def test_conditioned_search_preserves_density_starts_budget_and_parallel_results(
    learned,
):
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        initial_points,
        search,
    )

    p, _, _ = fixture(algebra="grouped", gaussian=learned)
    config = SearchConfig(starts=2, maxiter=60, conditioning="diagonal")
    best, serial = search(p, config, 721)
    other, parallel = search(p, replace(config, n_jobs=2), 721)
    for initial, left, right in zip(initial_points(p, 2, 721), serial, parallel):
        assert_array_equal(left["initial_parameters"], initial)
        assert left["iterations"] <= config.maxiter
        assert_allclose(left["objective"], p.objective(left["parameters"]), atol=1e-8)
        assert_allclose(left["parameters"], right["parameters"], atol=1e-9)
        assert left["conditioning"]["method"] == "diagonal"
        assert left["conditioning"]["gradient_evaluations"] > 0
        assert left["meets_gradient_tolerance"] == (
            left["physical_projected_gradient"] <= config.physical_gradient_tolerance
        )
    assert best["start"] == other["start"]


def test_unsupported_conditioning_fails_before_search():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    p, _, _ = fixture(features=1, algebra="grouped")
    with pytest.raises(ValueError, match="multiple factors"):
        search(p, SearchConfig(conditioning="diagonal"), 7)


@pytest.mark.parametrize("setting", ["noise_timescales", "run_baseline_sd", "joint"])
def test_conditioning_rejects_other_unsupported_models_before_search(setting):
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    p, _, _ = fixture(algebra="grouped")
    setattr(p, setting, {"brain": 1.0})
    with pytest.raises(ValueError, match="full dense/grouped model"):
        search(p, SearchConfig(conditioning="diagonal"), 7)


def test_conditioned_refinement_retains_original_and_uses_total_budget():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    p, _, _ = fixture(algebra="grouped")
    best, _ = search(
        p,
        SearchConfig(starts=1, maxiter=2, refine_maxiter=30, conditioning="diagonal"),
        4,
    )
    assert best["objective"] <= best["pre_refinement"]["objective"]
    assert best["refinement"]["iterations"] <= 30
    assert best["refinement"]["conditioning"]["method"] == "diagonal"
    assert best["refinement"]["conditioning"]["gtol"] == best["refinement"]["gtol"]


def test_conditioned_callback_error_is_not_swallowed():
    from multimodalsrm.bayesian.fitting import SearchConfig, search

    p, _, _ = fixture(algebra="grouped")
    failure = OSError("checkpoint unavailable")

    def progress(event):
        if event["phase"] == "iteration":
            raise failure

    with pytest.raises(OSError) as caught:
        search(
            p,
            SearchConfig(starts=2, maxiter=10, n_jobs=2, conditioning="diagonal"),
            2,
            progress=progress,
        )
    assert caught.value is failure


def test_affine_gradient_matches_finite_differences_and_zero_noise_is_rejected(
    monkeypatch,
):
    from scipy.optimize import OptimizeResult

    from multimodalsrm.bayesian import map_conditioning as c
    from multimodalsrm.bayesian.fitting import SearchConfig

    p, x, _ = fixture(algebra="grouped")
    expected, _ = p.value_gradient(x)
    noise = next(i for i, n in enumerate(p.names) if n[0] == "noise")
    width = next(i for i, n in enumerate(p.names) if n[0] == "filter")

    def inspect(fun, initial, *, bounds, **kwargs):
        value, gradient = fun(initial)
        assert_allclose(value, expected, atol=1e-10)
        for j in (0, noise, width):
            plus, minus = initial.copy(), initial.copy()
            plus[j], minus[j] = 1e-5, -1e-5
            numerical = (fun(plus)[0] - fun(minus)[0]) / 2e-5
            assert_allclose(gradient[j], numerical, atol=1e-6, rtol=1e-5)
        invalid = initial.copy()
        invalid[noise] = bounds[noise][0]
        assert np.isinf(fun(invalid)[0])
        return OptimizeResult(
            x=initial, nit=0, nfev=8, success=False, status=1, message="inspection only"
        )

    monkeypatch.setattr(c, "minimize", inspect)
    result, report = c.minimize_conditioned(p, x, SearchConfig(), maxiter=20)
    assert result.x[noise] > 0
    assert result.fun <= expected
    assert result.nit == 0
    assert report["stages"][0]["optimizer_success"] is False


def test_conditioned_model_round_trip_preserves_solver_and_physical_result(tmp_path):
    from sklearn.exceptions import ConvergenceWarning

    from multimodalsrm.bayesian import BayesianMultimodalSRM, SearchConfig
    from multimodalsrm.bayesian.persistence import load_model, save_model

    p, _, data = fixture(algebra="grouped")
    model = BayesianMultimodalSRM(
        features=2,
        priors=p.priors,
        responses=p.responses,
        inference="map",
        length_scale=p.length_scale,
        linear_algebra="grouped",
        max_observations=500,
        random_state=31,
        search=SearchConfig(starts=1, maxiter=10, conditioning="diagonal"),
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(data)
    save_model(tmp_path / "conditioned", model)
    restored, _ = load_model(tmp_path / "conditioned")
    assert restored.search_config_.conditioning == "diagonal"
    assert restored.map_diagnostics_ == model.map_diagnostics_
    assert_array_equal(restored.map_parameters_, model.map_parameters_)


def test_filter_probe_respects_open_lognormal_support_near_zero():
    from multimodalsrm.bayesian import Prior
    from multimodalsrm.bayesian.map_conditioning import InformationScaling

    p, x, _ = fixture(algebra="grouped")
    j = p.indices[("filter", "aux", "width")]
    p.parameter_priors[j] = Prior.lognormal(-2, 0.7).bounded(0, 0.8)
    p.bounds[j] = (0, 0.8)
    x[j] = 1e-8
    probes = []

    def supported(point):
        assert point[j] > 0, "a curvature probe left open prior support"
        probes.append(point[j])
        return 0.5 * float(point @ point), point.copy()

    scales, _ = InformationScaling(p).at(x, supported)
    assert min(probes) > 0 and np.isfinite(scales).all()


def test_failed_scale_refresh_retains_finite_best_with_failure_diagnostic(monkeypatch):
    from scipy.optimize import OptimizeResult

    from multimodalsrm.bayesian import map_conditioning as c
    from multimodalsrm.bayesian.fitting import SearchConfig

    p, x, _ = fixture(algebra="grouped")
    original = c.InformationScaling.at
    calls = 0

    def fail_second(self, point, evaluate=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("curvature unavailable")
        return original(self, point, evaluate)

    def one_step(fun, initial, *, bounds, callback, **kwargs):
        value, gradient = fun(initial)
        candidate = np.clip(
            initial - 1e-3 * gradient,
            np.asarray(bounds)[:, 0],
            np.asarray(bounds)[:, 1],
        )
        improved, _ = fun(candidate)
        assert improved < value
        callback(OptimizeResult(x=candidate, fun=improved))
        return OptimizeResult(
            x=candidate, nit=1, nfev=2, success=False, status=1, message="one step"
        )

    monkeypatch.setattr(c.InformationScaling, "at", fail_second)
    monkeypatch.setattr(c, "minimize", one_step)
    result, report = c.minimize_conditioned(p, x, SearchConfig(), maxiter=20)
    assert result.fun < p.value_gradient(x)[0]
    assert np.isfinite(result.x).all() and not result.success
    assert "curvature unavailable" in report["stages"][-1]["failure"]


def test_equal_rounded_objective_keeps_more_stationary_point(monkeypatch):
    from scipy.optimize import OptimizeResult

    from multimodalsrm.bayesian import map_conditioning as c
    from multimodalsrm.bayesian.fitting import SearchConfig

    p, x, _ = fixture(algebra="grouped")
    target = x.copy()
    target[0] -= 0.002

    def objective(point):
        delta = point - target
        return float(1e12 + 0.5 * delta @ delta), delta

    p.value_gradient = objective
    monkeypatch.setattr(
        c.InformationScaling,
        "at",
        lambda self, point, evaluate: (np.ones_like(point), {}),
    )

    def one_step(fun, initial, *, callback, **kwargs):
        value, gradient = fun(initial)
        point = initial - 0.5 * gradient
        improved, _ = fun(point)
        # Both totals round to the same float, while the gradient improves.
        assert improved == value
        callback(OptimizeResult(x=point, fun=improved))
        return OptimizeResult(x=point, nit=1, nfev=2, success=True, status=0, message="step")

    monkeypatch.setattr(c, "minimize", one_step)
    result, _ = c.minimize_conditioned(p, x, SearchConfig(), maxiter=1)
    assert result.fun == objective(x)[0]
    assert np.max(np.abs(result.jac)) < np.max(np.abs(objective(x)[1]))
