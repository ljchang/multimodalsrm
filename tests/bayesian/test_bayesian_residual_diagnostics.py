"""Native-time residual summaries and fixed-parameter donor interventions."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .test_bayesian_problem import independent_parameters, problem_fixture


def api():
    try:
        return importlib.import_module("multimodalsrm.bayesian.diagnostics")
    except ModuleNotFoundError:
        pytest.fail("residual and donor diagnostics are not implemented")


def test_lagged_pairs_use_actual_times_without_compressing_gaps_or_masks():
    times = np.array([0.0, 1.0, 3.0, 4.0, 6.0, 7.0])
    y = np.array([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])[:, None]
    mask = np.ones_like(y, bool)
    q = api().residual_summary(
        times,
        y,
        np.zeros_like(y),
        np.ones_like(y),
        mask,
        lags=[1.0, 2.0],
        tolerance=1e-6,
    )
    first = q["features"][0]
    assert first["acf"][0]["n_pairs"] == 3
    assert_allclose(first["acf"][0]["correlation"], -1.0)
    assert first["acf"][1]["n_pairs"] == 2
    assert first["acf"][1]["correlation"] is None  # Fewer than 3 pairs.
    mask[2] = False
    q = api().residual_summary(
        times, y, np.zeros_like(y), np.ones_like(y), mask, lags=[1.0], tolerance=1e-6
    )
    assert q["features"][0]["acf"][0]["n_pairs"] == 2
    assert q["features"][0]["n_values"] == 5


def test_residual_scale_and_constant_series_are_explicit():
    y = np.ones((6, 2)) * [2.0, 3.0]
    mask = np.ones_like(y, bool)
    mask[:, 1] = False
    q = api().residual_summary(
        np.arange(6.0), y, np.zeros_like(y), np.full_like(y, 4.0), mask, lags=[1.0]
    )
    a, b = q["features"]
    assert a["rmse"] == 2.0
    assert a["standardized_rms"] == 1.0
    assert a["acf"][0]["correlation"] is None
    assert b["n_values"] == 0
    assert b["rmse"] is None


def test_time_translation_and_small_timestamp_roundoff_preserve_pairs():
    t = np.arange(10.0)
    y = np.sin(t)[:, None]
    args = (y, y * 0.1, np.ones_like(y), np.ones_like(y, bool))
    a = api().residual_summary(t, *args, lags=[1.0, 2.0])
    b = api().residual_summary(t + 252 + np.arange(10) * 1e-10, *args, lags=[1.0, 2.0])
    assert a == b


@pytest.mark.parametrize("bad", ["time_order", "variance", "lag", "tolerance"])
def test_invalid_inputs_cannot_be_reported_as_white_residuals(bad):
    t = np.arange(6.0)
    y = t[:, None]
    v = np.ones_like(y)
    lags, tol = [1.0], 1e-6
    if bad == "time_order":
        t[1] = 0
    elif bad == "variance":
        v[2] = -1
    elif bad == "lag":
        lags = [-1.0]
    else:
        tol = 1.0
    with pytest.raises(ValueError):
        api().residual_summary(t, y, y * 0, v, np.ones_like(y, bool), lags=lags, tolerance=tol)


def diagnostic_problem(linear_algebra):
    from multimodalsrm._observation_preparation import ObservationSystem
    from multimodalsrm.bayesian import BayesianProblem

    original, adapter, _ = problem_fixture()
    # Two ref donors; signal is the held-out target. No target observations.
    system = ObservationSystem(np.array([1.0, 4.0]), [("a", "ref", 0)] * 2, np.array([2.0, -1.0]))
    p = BayesianProblem(
        adapter,
        original.priors,
        anchor=original.anchor,
        systems={"test": system},
        linear_algebra=linear_algebra,
    )
    return p, independent_parameters(p)


@pytest.mark.parametrize("linear_algebra", ["dense", "grouped"])
def test_donor_removal_matches_direct_conditioning_and_empty_donors_give_prior(
    linear_algebra,
):
    p, x = diagnostic_problem(linear_algebra)
    before = p.systems["test"].values.copy()
    t = np.array([2.0, 3.0])
    result = api().fixed_prediction(p, x, "test", t, target=("b", "signal"))

    def matern(a, b):
        d = np.sqrt(3) * np.abs(a[:, None] - b[None, :]) / 3
        return (1 + d) * np.exp(-d)

    C = 1.44 * matern(np.array([1.0, 4.0]), np.array([1.0, 4.0])) + 0.15 * np.eye(2)
    cross = -0.84 * matern(t, np.array([1.0, 4.0]))
    expected = cross @ np.linalg.solve(C, np.array([1.88, -1.12])) - 0.2
    assert_allclose(result["mean"][:, 0], expected, atol=1e-10)
    removed = api().fixed_prediction(
        p, x, "test", t, target=("b", "signal"), drop_modalities=["ref"]
    )
    assert_allclose(removed["mean"], -0.2)
    assert_allclose(removed["variance"], 0.79)  # loading squared + measurement variance
    assert removed["remaining_observations"] == 0
    assert removed["removed_observations"] == 2
    assert np.all(removed["variance"] >= result["variance"] - 1e-12)
    assert_allclose(p.systems["test"].values, before, atol=0, rtol=0)
    assert len(p.systems["test"].times) == 2


def test_zero_loading_donor_has_no_influence():
    p, x = diagnostic_problem("grouped")
    x[p.indices[("loading", "a", "ref", 0)]] = 0.0
    full = api().fixed_prediction(p, x, "test", np.array([2.0]), target=("b", "signal"))
    removed = api().fixed_prediction(
        p, x, "test", np.array([2.0]), target=("b", "signal"), drop_modalities=["ref"]
    )
    assert_allclose(full["mean"], removed["mean"], atol=1e-12)
    assert_allclose(full["variance"], removed["variance"], atol=1e-12)


def test_target_in_conditioning_system_is_rejected_even_if_it_would_be_dropped():
    p, x = diagnostic_problem("dense")
    with pytest.raises(ValueError, match="target"):
        api().fixed_prediction(
            p, x, "test", np.array([2.0]), target=("a", "ref"), drop_modalities=["ref"]
        )


def test_empty_donor_prior_rejects_invalid_noise_parameters():
    p, x = diagnostic_problem("dense")
    x[p.indices[("noise", "b", "signal")]] = -1.0
    with pytest.raises(ValueError, match="parameter"):
        api().fixed_prediction(
            p,
            x,
            "test",
            np.array([2.0]),
            target=("b", "signal"),
            drop_modalities=["ref"],
        )


def test_prediction_comparison_uses_identical_finite_rows_and_reports_harm():
    y = np.array([[0.0], [1.0], [2.0], [1000.0]])
    full = np.array([[0.0], [0.0], [0.0], [0.0]])
    removed = np.array([[0.0], [1.0], [2.0], [np.nan]])
    q = api().compare_predictions(y, full, removed, np.ones_like(y, bool))
    assert q[0]["n_values"] == 3
    assert q[0]["mse_full"] == pytest.approx(5 / 3)
    assert q[0]["mse_removed"] == 0
    assert q[0]["delta_mse"] == pytest.approx(-5 / 3)
    assert q[0]["prediction_change_rms"] == pytest.approx(np.sqrt(5 / 3))
