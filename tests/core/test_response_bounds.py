"""Convergence at response bounds is reported and warned about."""

import warnings

import numpy as np
import pytest

from multimodalsrm import Gaussian, Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.kernel_optimization import kernel_stationarity, response_bound_report


def recording():
    """Three participants; ``signal`` lags ``reference`` by 0.9 s through a 0.7 s Gaussian."""
    rng = np.random.default_rng(11)
    fine = np.arange(0, 80, 0.25)
    latent = np.convolve(rng.normal(size=len(fine) + 40), np.hanning(41), mode="valid")[: len(fine)]
    latent /= latent.std()
    shifted = np.interp(fine - 0.9, fine, latent)
    smooth = np.convolve(shifted, np.exp(-0.5 * (np.arange(-12, 13) * 0.25 / 0.7) ** 2), "same")
    smooth /= smooth.std()
    data = {}
    for index, subject in enumerate("abc"):
        local = np.random.default_rng(index)
        reference = latent[::4, None] * local.normal(size=(1, 3)) + local.normal(size=(80, 3)) * 0.2
        signal = smooth[::6, None] * local.normal(size=(1, 4))
        signal = signal + local.normal(size=signal.shape) * 0.2
        data[subject] = {
            "run": {
                "reference": TimeSeries(reference, fine[::4]),
                "signal": TimeSeries(signal, fine[::6]),
            }
        }
    return data


def fit(lag_bounds, width_bounds=(0.2, 2.0), **settings):
    responses = {
        "reference": Response(Identity(), estimate=False),
        "signal": Response(
            Gaussian(width=0.5, lag=0.2),
            pooling="shared",
            bounds={"width": width_bounds, "lag": lag_bounds},
        ),
    }
    options = dict(
        features=1,
        latent_dt=0.5,
        responses=responses,
        latent_timescale=1.0,
        max_iter=400,
        random_state=0,
    )
    options.update(settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MultimodalSRM(**options).fit(recording())
    return model, [w for w in caught if "held at their bounds" in str(w.message)]


def test_interior_fit_reports_no_bound_and_does_not_warn():
    model, caught = fit((-1.0, 2.5))
    report = model.optimization_diagnostics_["response_bounds"]
    assert model.converged_ and not caught
    assert report["at_bounds"] == [] and report["limited_by_bounds"] == []
    assert report["free_parameters"] == 2


def test_convergence_at_a_bound_with_an_outward_gradient_warns():
    # The delay the data support lies above the upper lag bound.
    model, caught = fit((-1.0, 0.4))
    report = model.optimization_diagnostics_["response_bounds"]
    assert model.converged_
    assert model.group_kernels_["signal"].lag == pytest.approx(0.4)
    rows = {r["parameter"]: r for r in report["at_bounds"]}
    assert rows["lag"]["bound"] == "upper" and rows["lag"]["gradient"] < 0
    assert rows["lag"]["bounds"] == (-1.0, 0.4) and rows["lag"]["value"] == pytest.approx(0.4)
    assert rows["lag"]["objective_decreases_beyond"]
    assert rows["lag"]["projected_decrease"] > report["tolerance"]
    assert "signal.lag" in report["limited_by_bounds"]
    assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
    assert "signal.lag" in str(caught[0].message)
    # Every restart keeps its own report.
    assert model.restart_diagnostics_[0]["response_bounds"] == report
    # Widening the bound lowers the objective, as the gradient says.
    wider, quiet = fit((-1.0, 2.5))
    assert not quiet
    assert wider.objective_history_[-1] < model.objective_history_[-1]


def test_every_free_parameter_at_a_bound_still_counts_as_converged():
    # Both coordinates pinned: no response step is left, so the stopping rule passes.
    model, caught = fit((-1.0, 0.4), width_bounds=(0.2, 0.55))
    report = model.optimization_diagnostics_["response_bounds"]
    steps = [d["kernel_step"] for d in model.restart_diagnostics_[0]["diagnostics"]]
    assert model.converged_ and steps[-1] == 0.0
    assert len(report["at_bounds"]) == report["free_parameters"] == 2
    assert set(report["limited_by_bounds"]) == {"signal.lag", "signal.width"}
    assert len(caught) == 1


def quadratic_rows(lag, target):
    kernels = {s: {"brain": Gaussian(width=0.25, lag=lag)} for s in "ab"}
    responses = {
        "brain": Response(
            Gaussian(width=0.25), pooling="shared", fixed={"width": 0.25}, bounds={"lag": (0, 2)}
        )
    }
    return kernel_stationarity(
        kernels, responses, lambda k: sum((m["brain"].lag - target) ** 2 for m in k.values())
    )


@pytest.mark.parametrize(
    "lag,target,bound,beyond",
    [(2, 3, "upper", True), (0, -1, "lower", True), (2, 1, "upper", False), (0, 1, "lower", False)],
)
def test_bound_report_separates_outward_from_inward_gradients(lag, target, bound, beyond):
    report = response_bound_report(quadratic_rows(lag, target), 1e-6)
    (row,) = report["at_bounds"]
    assert row["bound"] == bound and row["subjects"] == ["a", "b"]
    assert row["objective_decreases_beyond"] is beyond
    # Two participants, slope 2 * (lag - target) each, over a range of two seconds.
    assert row["projected_decrease"] == pytest.approx(8.0 if beyond else 0.0, rel=1e-4)
    assert report["limited_by_bounds"] == (["brain.lag"] if beyond else [])


def test_bound_report_ignores_interior_parameters_and_small_slopes():
    assert response_bound_report(quadratic_rows(0.5, 1), 1e-6)["at_bounds"] == []
    # A minimum on the bound: the one-sided difference leaves a slope of order its step.
    flat = response_bound_report(quadratic_rows(2, 2), 1e-3)
    assert 0 < flat["at_bounds"][0]["projected_decrease"] < 1e-3
    assert len(flat["at_bounds"]) == 1 and flat["limited_by_bounds"] == []


def test_extrapolation_records_the_distance_covered_not_only_its_scale():
    model, _ = fit((-1.0, 2.5), tol=1e-9, kernel_tol=1e-6)
    records = model.restart_diagnostics_[0]["diagnostics"]
    accepted = [d for d in records if d["kernel_extrapolation"]["accepted_steps"]]
    assert accepted
    for d in records:
        e = d["kernel_extrapolation"]
        assert (e["displacement"] > 0) == (e["accepted_steps"] > 0)
        # An accepted scale s extends a step d to at most (1 + s) d.
        assert e["scale"] == 2 ** e["accepted_steps"] - 1 <= 4095
        assert e["displacement"] <= d["kernel_step"] * e["scale"] / (1 + e["scale"]) + 1e-12
