"""Independent Gaussian conditional densities and stiff variance regressions."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult

from multimodalsrm import TimeSeries

from .test_bayesian_problem import api, independent_parameters, problem_fixture


def conditional_change(p, x, j, delta):
    """Independent conditional covariance eigenvalues, with decimal arithmetic."""
    with localcontext() as context:
        context.prec = 60
        D = Decimal.from_float
        total = Decimal(0)
        _, offsets, _, _, _ = p.arrays(x)
        for run, system in p.systems.items():
            selected = np.array([k[:2] == p.names[j][1:] for k in system.keys])
            if not selected.any():
                continue
            C = np.asarray(p.covariance(x, run))
            r = system.values - np.asarray(offsets)[p._packed[run][0]]
            S, u = C[np.ix_(selected, selected)], r[selected]
            if (~selected).any():
                cross = C[np.ix_(selected, ~selected)]
                other = C[np.ix_(~selected, ~selected)]
                S = S - cross @ np.linalg.solve(other, cross.T)
                u = u - cross @ np.linalg.solve(other, r[~selected])
            eigenvalues, vectors = np.linalg.eigh(S)
            for v, r2 in zip(eigenvalues, (vectors.T @ u) ** 2):
                v, r2, d = D(float(v)), D(float(r2)), D(float(delta))
                total += ((1 + d / v).ln() - d * r2 / (v * (v + d))) / 2
        prior = p.parameter_priors[j]
        v, d = D(float(x[j])), D(float(delta))
        loc, scale = D(prior.loc), D(prior.scale)
        if prior.family == "lognormal":
            logchange = (1 + d / v).ln()
            total += logchange * (1 + (v.ln() - loc) / scale**2)
            total += logchange**2 / (2 * scale**2)
        elif prior.family == "normal":
            total += d * (v - loc) / scale**2 + d**2 / (2 * scale**2)
        return float(total)


def multi_fixture(mode, baseline, family, features=2):
    b = api()
    p, adapter, data = problem_fixture(True, two_runs=True)
    for run in data["a"]:
        ts = data["a"][run]["ref"]
        data["a"][run]["ref"] = TimeSeries(
            np.column_stack([ts.values[:, 0] * 0.7**i for i in range(features)]),
            ts.times,
        )
    adapter._prepare(data)
    noise = {
        "normal": b.Prior.normal(0.2, 0.4).bounded(0.01, 0.9),
        "lognormal": b.Prior.lognormal(-2.0, 0.7).bounded(0.01, 0.9),
        "uniform": b.Prior.uniform(0.01, 0.9),
    }[family]
    p = b.BayesianProblem(
        adapter,
        b.BayesianPriors(noise=noise, filters=p.priors.filters),
        anchor=("a", "ref", 0),
        linear_algebra=mode,
        run_baseline_sd={"ref": 0.3, "signal": 0.2} if baseline else None,
    )
    return p, independent_parameters(p), p.indices[("noise", "a", "ref")]


@pytest.mark.parametrize("mode", ["dense", "grouped"])
@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("family", ["normal", "lognormal", "uniform"])
def test_profile_matches_independent_conditional_density(mode, baseline, family):
    from multimodalsrm.bayesian.noise_profile import variance_profile

    p, x, j = multi_fixture(mode, baseline, family)
    profile = variance_profile(p, x, j)
    assert profile.available
    assert_allclose(profile.gradient, p.value_gradient(x)[1][j], atol=2e-10)
    h = 1e-5
    plus, minus = x.copy(), x.copy()
    plus[j] += h
    minus[j] -= h
    expected = (p.value_gradient(plus)[1][j] - p.value_gradient(minus)[1][j]) / (2 * h)
    assert_allclose(profile.curvature, expected, rtol=2e-8)
    for delta in [-0.03, -1e-10, 1e-10, 0.02]:
        assert_allclose(
            profile.change(delta),
            conditional_change(p, x, j, delta),
            atol=1e-20,
            rtol=2e-10,
        )
    assert np.isinf(profile.change(-x[j]))
    assert np.isinf(profile.change(1.0))


def stiff_fixture(mode="dense", v=1e-4):
    b = api()
    _, adapter, data = problem_fixture()
    for subject in data:
        for modality, ts in data[subject]["train"].items():
            data[subject]["train"][modality] = TimeSeries(
                np.tile([-np.sqrt(v), np.sqrt(v)], 4)[:, None], ts.times[:8]
            )
    adapter._prepare(data)
    p = b.BayesianProblem(
        adapter,
        b.BayesianPriors(noise=b.Prior.uniform(1e-8, 1.0)),
        anchor=("a", "ref", 0),
        linear_algebra=mode,
    )
    x = np.zeros(len(p.names))
    for i, name in enumerate(p.names):
        if name[0] == "noise":
            x[i] = v
    j = p.indices[("noise", "a", "ref")]
    x[j] += 1e-10
    return p, x, j


@pytest.mark.parametrize("mode", ["dense", "grouped"])
def test_tiny_variance_and_offset_reach_stationarity_in_one_polishing_step(mode):
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import polish

    # Both coordinates violate stationarity, requiring the general Hessian
    # sweep. Unit-scaled probes span 40% of this variance and leave it far
    # from stationarity; relative probes resolve this local Gaussian problem.
    p, x, j = stiff_fixture(mode, v=1.5e-5)
    x[j] += 1e-10
    x[p.indices["offset", "a", "ref", 0]] = 1e-7
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    assert not record["meets_gradient_tolerance"]
    report = polish(p, record, config, 1)
    assert record["meets_gradient_tolerance"], record["physical_projected_gradient"]
    assert report["steps"] == 1
    delta = record["parameters"][j] - x[j]
    assert conditional_change(p, x, j, delta) < 0
    assert_allclose(record["parameters"][j], 1.5e-5, rtol=1e-8)


def test_isolated_variance_repair_avoids_a_full_hessian_sweep():
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import polish

    # Only one variance violates stationarity. The real Gaussian coordinate
    # update should resolve it with bounded gradient work even in a larger fit.
    p, x, j = stiff_fixture()
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    report = polish(p, record, config, 1)
    assert record["meets_gradient_tolerance"]
    assert conditional_change(p, x, j, record["parameters"][j] - x[j]) < 0
    assert report["gradient_evaluations"] <= 3


def test_unavailable_priority_profile_still_uses_general_polishing(monkeypatch):
    from multimodalsrm.bayesian import noise_profile
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import polish

    p, x, _ = stiff_fixture()
    monkeypatch.setattr(noise_profile, "MAX_SELECTED_ROWS", 1)
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    report = polish(p, record, config, 1)
    assert record["meets_gradient_tolerance"]
    assert report["accepted_steps"] == 1
    assert "capacity" in report["attempts"][0]["variance_profile"]["failure"]


@pytest.mark.parametrize("mode", ["dense", "grouped"])
def test_polish_accepts_real_profile_decrease_despite_upward_rounded_total(mode):
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import polish

    p, x, j = stiff_fixture(mode)
    actual = p.value_gradient
    # Reproduce the observed rounding disagreement without perturbing gradients.
    value = actual(x)[0]
    p.value_gradient = lambda y: (
        value if np.array_equal(y, x) else np.nextafter(value, np.inf),
        actual(y)[1],
    )
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    report = polish(p, record, config, 1)
    assert report["accepted_steps"] == 1
    assert record["meets_gradient_tolerance"]
    assert record["objective"] > value
    delta = record["parameters"][j] - x[j]
    assert conditional_change(p, x, j, delta) < 0
    detail = report["attempts"][0]["variance_profile"]
    assert detail["stable_change"] < -detail["cancellation_bound"]
    assert detail["rounded_absolute_change"] > 0


def test_profile_rejects_worsening_step_even_with_smaller_gradient():
    from multimodalsrm.bayesian.noise_profile import variance_profile

    p, x, j = stiff_fixture()
    # Far below a Gaussian variance optimum, an overshoot can reduce |gradient|
    # while increasing the density objective. It must never count as descent.
    x[j] = 1e-4 * 0.8
    profile = variance_profile(p, x, j)
    delta = 1e-4 * 0.5
    y = x.copy()
    y[j] += delta
    assert abs(p.value_gradient(y)[1][j]) < abs(p.value_gradient(x)[1][j])
    assert conditional_change(p, x, j, delta) > 0
    assert not profile.is_descent(delta)


def test_worsening_proposal_cannot_replace_record(monkeypatch):
    from multimodalsrm.bayesian import noise_profile
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import _variance_repair

    p, x, j = stiff_fixture()
    x[j] = 0.95e-4
    delta = 0.105e-4
    y = x.copy()
    y[j] += delta
    assert conditional_change(p, x, j, delta) > 0
    assert abs(p.value_gradient(y)[1][j]) < abs(p.value_gradient(x)[1][j])
    profile = noise_profile.variance_profile(p, x, j)
    # Force a deliberately bad proposal; its density calculation remains real.
    profile.curvature = -profile.gradient / delta
    monkeypatch.setattr(noise_profile, "variance_profile", lambda *args: profile)
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    original = dict(record)
    report, attempt = dict(gradient_evaluations=0, accepted_steps=0), {}
    assert not _variance_repair(
        p, record, config, x, p.value_gradient(x)[1], np.arange(len(x)), report, attempt
    )
    assert record == original
    assert report["gradient_evaluations"] == 0
    assert attempt["variance_profile"]["stable_change"] > 0


def test_grouped_profile_chunks_selected_rhs_without_dense_observation_array(
    monkeypatch,
):
    from multimodalsrm.bayesian import noise_profile

    p, x, j = multi_fixture("grouped", True, "lognormal", features=5)
    expected = conditional_change(p, x, j, -1e-7)
    monkeypatch.setattr(p, "covariance", lambda *args: pytest.fail("observation covariance"))
    observation_count = len(p.systems["train"].values)
    original_zeros, original_empty, original_eye = np.zeros, np.empty, np.eye

    def checked_allocate(original):
        def allocate(shape, *args, **kwargs):
            assert shape != (observation_count, observation_count)
            if isinstance(shape, tuple) and shape[0] == observation_count:
                assert shape[1] <= 32
            return original(shape, *args, **kwargs)

        return allocate

    monkeypatch.setattr(noise_profile.np, "zeros", checked_allocate(original_zeros))
    monkeypatch.setattr(noise_profile.np, "empty", checked_allocate(original_empty))
    monkeypatch.setattr(
        noise_profile.np,
        "eye",
        lambda n, *a, **kw: (
            pytest.fail("observation identity")
            if n == observation_count
            else original_eye(n, *a, **kw)
        ),
    )
    profile = noise_profile.variance_profile(p, x, j)
    assert profile.available, profile.reason
    assert_allclose(profile.change(-1e-7), expected, rtol=1e-10)
    assert all(r["maximum_rhs_columns"] == 32 for r in profile.diagnostics["runs"])
    assert all(r["rhs_batches"] == 3 for r in profile.diagnostics["runs"])


def test_capacity_and_unsupported_mode_are_explicit_without_factorization(monkeypatch):
    from multimodalsrm.bayesian import noise_profile

    p, x, j = multi_fixture("grouped", False, "uniform")
    assert noise_profile.MAX_SELECTED_ROWS == 1024
    monkeypatch.setattr(noise_profile, "MAX_SELECTED_ROWS", 4)
    monkeypatch.setattr(
        noise_profile,
        "_grouped_solver",
        lambda *args: pytest.fail("capacity must precede factorization"),
    )
    profile = noise_profile.variance_profile(p, x, j)
    assert not profile.available and "capacity" in profile.reason
    p.linear_algebra = "spectral"
    profile = noise_profile.variance_profile(p, x, j)
    assert not profile.available and "dense/grouped" in profile.reason


def test_profile_ignores_runs_without_selected_group():
    from multimodalsrm.bayesian.noise_profile import variance_profile

    p, adapter, data = problem_fixture(two_runs=True)
    del data["a"]["second"]
    adapter._prepare(data)
    p = api().BayesianProblem(adapter, p.priors, anchor=p.anchor, linear_algebra="grouped")
    x = independent_parameters(p)
    j = p.indices[("noise", "a", "ref")]
    profile = variance_profile(p, x, j)
    assert profile.available
    assert len(profile.diagnostics["runs"]) == 1
    assert_allclose(profile.change(-1e-7), conditional_change(p, x, j, -1e-7), rtol=1e-10)


def test_grouped_profile_supports_run_without_configured_baseline_modality():
    from multimodalsrm.bayesian.noise_profile import variance_profile

    p, adapter, data = problem_fixture(two_runs=True)
    for runs in data.values():
        runs["second"].pop("signal", None)
        if not runs["second"]:
            del runs["second"]
    adapter._prepare(data)
    p = api().BayesianProblem(
        adapter,
        p.priors,
        anchor=p.anchor,
        linear_algebra="grouped",
        run_baseline_sd={"signal": 0.2},
    )
    assert p.baseline_designs["train"][0].shape == (18, 1)
    assert p.baseline_designs["second"][0].shape == (9, 0)
    x = independent_parameters(p)
    j = p.indices[("noise", "a", "ref")]
    value, gradient = p.value_gradient(x)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    profile = variance_profile(p, x, j)
    assert profile.available, profile.reason
    assert len(profile.diagnostics["runs"]) == 2
    assert_allclose(profile.gradient, gradient[j], atol=2e-10)
    for delta in (-0.03, -1e-10, 1e-10, 0.02):
        assert_allclose(
            profile.change(delta),
            conditional_change(p, x, j, delta),
            atol=1e-20,
            rtol=2e-10,
        )


@pytest.mark.parametrize("failure", ["nonpositive_curvature", "no_gradient_improvement"])
def test_failed_profile_candidate_retains_original_record(failure):
    from multimodalsrm.bayesian.fitting import (
        SearchConfig,
        _map_diagnostics,
    )
    from multimodalsrm.bayesian.polishing import _variance_repair

    p, x, j = stiff_fixture()
    if failure == "nonpositive_curvature":
        x[j] = 3e-4
    value, gradient = p.value_gradient(x)
    config = SearchConfig()
    record = _map_diagnostics(
        p, x, config, OptimizeResult(success=False, status=1, nit=0, message="test")
    )
    original = dict(record)
    if failure == "no_gradient_improvement":
        # A real profile decrease is insufficient without the fresh gradient gate.
        p.value_gradient = lambda y: (value, gradient.copy())
    report, attempt = dict(gradient_evaluations=0, accepted_steps=0), {}
    assert not _variance_repair(p, record, config, x, gradient, np.arange(len(x)), report, attempt)
    assert record == original and report["accepted_steps"] == 0
    expected = "curvature" if failure == "nonpositive_curvature" else "fresh projected gradient"
    assert expected in attempt["variance_profile"]["failure"]
