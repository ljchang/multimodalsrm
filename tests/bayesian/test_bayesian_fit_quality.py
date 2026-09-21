"""Fit reports must distinguish kernel shape, timing convention and convergence."""

import copy
import importlib

import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Gaussian, Identity, Response


def quality():
    try:
        return importlib.import_module("multimodalsrm.bayesian.quality")
    except ModuleNotFoundError:
        pytest.fail("fit-quality summaries are not implemented")


def record():
    responses = {
        "reference": Response(Gaussian(1.0, 1.0), estimate=False, pooling="shared"),
        "signal": Response(
            Gaussian(1.0, 0.0),
            pooling="shared",
            bounds={"width": (0.5, 2.0), "lag": (-4.0, 4.0)},
        ),
        "instant": Response(Identity(), estimate=False, pooling="shared"),
    }
    names = [
        ("loading", "s", "signal", 0),
        ("filter", "signal", "width"),
        ("filter", "signal", "lag"),
    ]
    best = dict(
        parameters=[0.8, 2.0, 3.5],
        objective=12.0,
        physical_projected_gradient=0.0001,
        meets_gradient_tolerance=True,
        boundary_parameters=[("filter", "signal", "width")],
    )
    other = dict(
        best,
        parameters=[0.6, 1.0, 2.5],
        objective=15.0,
        physical_projected_gradient=0.2,
        meets_gradient_tolerance=False,
        boundary_parameters=[],
    )
    return dict(
        configuration=dict(
            response_metadata={m: r.metadata for m, r in responses.items()},
            reference_convention=dict(
                reference_modality="reference",
                reference_lag_seconds=1.0,
                absolute_physiological_timing_established=False,
            ),
            inference="map",
            uncertainty="conditional_on_MAP",
            search={"physical_gradient_tolerance": 0.001},
            linear_system_sizes={"train": {"observations": 100, "functionals": 40}},
        ),
        parameter_names=names,
        map=best,
        restarts=[best, other, {"objective": None, "failure": "nonfinite"}],
        sampling=None,
    )


def test_gaussian_peak_fwhm_and_support_use_fitted_values_and_reference_clock():
    q = quality().summarize_fit(record())
    rows = {r["modality"]: r for r in q["filters"]}
    s = rows["signal"]
    assert s["peak_delay_seconds"] == 2.5
    assert_allclose(s["fwhm_seconds"], 4.709640090061899, rtol=1e-14)
    assert_allclose(s["support_relative_seconds"], [-9.5, 14.5])
    assert s["boundary_parameters"] == ["width"]
    assert s["estimated_parameters"] == ["width", "lag"]
    assert rows["reference"]["estimated_parameters"] == []
    assert rows["instant"]["peak_delay_seconds"] == -1.0
    assert rows["instant"]["fwhm_seconds"] is None
    assert rows["instant"]["shape_type"] == "impulse"
    assert q["map_passed"] is True  # A bound does not imply failed stationarity.
    assert q["calibration_established"] is False


def test_global_clock_shift_changes_neither_relative_peak_nor_shape_width():
    r = record()
    a = quality().summarize_fit(r)
    # Gaussian-only case: Identity is fixed at zero and cannot be shifted.
    del r["configuration"]["response_metadata"]["instant"]
    r["configuration"]["reference_convention"]["reference_lag_seconds"] += 4.0
    for response in r["configuration"]["response_metadata"].values():
        response["parameters"]["lag"] += 4.0
    r["map"]["parameters"][-1] += 4.0
    b = quality().summarize_fit(r)
    for x, y in zip(a["filters"][:2], b["filters"]):
        assert x["peak_delay_seconds"] == y["peak_delay_seconds"]
        assert x["fwhm_seconds"] == y["fwhm_seconds"]


def test_restart_spread_is_descriptive_and_failed_sampling_stays_failed():
    r = record()
    before = copy.deepcopy(r)
    q = quality().summarize_fit(r)
    assert q["restarts"] == {"total": 3, "finite": 2, "stationary": 1}
    s = next(x for x in q["filters"] if x["modality"] == "signal")
    assert s["restart_ranges"]["lag"] == {"min": 2.5, "max": 3.5, "n": 2}
    assert q["sampling_passed"] is None
    assert r == before
    r["configuration"]["inference"] = "posterior"
    r["sampling"] = {"passes": False}
    q = quality().summarize_fit(r)
    assert q["sampling_passed"] is False
    assert q["calibration_established"] is False


def test_unfinished_fit_retains_diagnostics_without_inventing_shapes():
    r = record()
    r["configuration"] = None
    r["map"] = None
    q = quality().summarize_fit(r)
    assert q["filters"] == []
    assert q["map_passed"] is None
    assert q["restarts"]["total"] == 3
    assert q["availability"] == "incomplete"


def test_corrupt_parameter_vector_cannot_produce_a_filter_report():
    r = record()
    r["map"]["parameters"] = [1.0]
    with pytest.raises(ValueError, match="parameter"):
        quality().summarize_fit(r)


@pytest.mark.parametrize("corruption", ["duplicate", "nonfinite", "missing_filter"])
def test_corrupt_filter_metadata_cannot_silently_use_initial_values(corruption):
    r = record()
    if corruption == "duplicate":
        r["parameter_names"][-1] = r["parameter_names"][-2]
    elif corruption == "nonfinite":
        r["map"]["parameters"][-1] = float("nan")
    else:
        r["parameter_names"][-1] = ("noise", "s", "signal")
    with pytest.raises(ValueError, match="parameter"):
        quality().summarize_fit(r)


def test_pairwise_delay_does_not_turn_reference_convention_into_absolute_timing():
    q = quality().summarize_fit(record())
    pair = next(
        p
        for p in q["relative_delays"]
        if p["from_modality"] == "reference" and p["to_modality"] == "signal"
    )
    assert pair["delay_seconds"] == 2.5
    assert q["absolute_physiological_timing_established"] is False


def test_unsupported_family_is_explicitly_unavailable():
    r = record()
    r["configuration"]["response_metadata"]["instant"]["family"] = "FutureKernel"
    q = quality().summarize_fit(r)
    row = next(f for f in q["filters"] if f["modality"] == "instant")
    assert row["shape_type"] == "unsupported"
    assert row["peak_delay_seconds"] is None
    assert q["availability"] == "incomplete"


@pytest.mark.parametrize("corruption", ["lag_mismatch", "learned_lag", "family_mismatch"])
def test_reference_must_match_the_named_response_and_have_a_fixed_lag(corruption):
    r = record()
    ref = r["configuration"]["reference_convention"]
    if corruption == "lag_mismatch":
        ref["reference_lag_seconds"] = 10.0
    elif corruption == "learned_lag":
        ref.update(reference_modality="signal", reference_lag_seconds=3.5)
    else:
        ref["reference_family"] = "Identity"
    with pytest.raises(ValueError, match="reference"):
        quality().summarize_fit(r)


def test_reference_can_have_a_learned_width_while_its_lag_is_fixed():
    r = record()
    r["configuration"]["response_metadata"]["reference"] = Response(
        Gaussian(1.0, 1.0), pooling="shared", fixed={"lag": 1.0}
    ).metadata
    r["parameter_names"].append(("filter", "reference", "width"))
    for start in r["restarts"]:
        if start.get("parameters") is not None:
            start["parameters"].append(2.0)
    q = quality().summarize_fit(r)
    ref = next(row for row in q["filters"] if row["modality"] == "reference")
    assert q["availability"] == "complete"
    assert ref["peak_delay_seconds"] == 0.0
    assert_allclose(ref["fwhm_seconds"], 4.709640090061899)


def test_identity_cannot_carry_a_learned_gaussian_parameter_schema():
    r = record()
    r["configuration"]["response_metadata"]["signal"]["family"] = "Identity"
    with pytest.raises(ValueError, match="Identity"):
        quality().summarize_fit(r)
