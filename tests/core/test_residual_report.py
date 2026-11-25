"""Residual diagnostics respond to what they are meant to detect and to nothing else."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.residuals import (
    ResidualSeries,
    autocorrelation,
    exponential_timescale,
    residual_report,
)

from ..reference.residual_scenarios import recording

SCENARIOS = {
    "clean": (dict(), 2),
    "underfit": (dict(), 1),
    "private": (dict(private=1.0, delay=3), 2),
    "ar": (dict(ar=0.7), 2),
}


def ar1(rng, shape, rho):
    white = rng.normal(size=shape)
    out = np.empty(shape)
    out[0] = white[0]
    for i in range(1, shape[0]):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho**2) * white[i]
    return out


def series(values, times, subject="s", modality="m", run="run"):
    return ResidualSeries(subject, run, modality, np.asarray(times, float), values, 0 * values)


def fit(data, factors):
    model = MultimodalSRM(
        features=factors,
        latent_dt=1.0,
        responses={m: Response(Identity(), estimate=False) for m in "ab"},
        random_state=0,
        max_iter=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(data)


@pytest.fixture(scope="module")
def reports():
    output = {}
    for name, (departure, factors) in SCENARIOS.items():
        data = recording(participants=8, runs=2, samples=200, seed=1, **departure)
        output[name] = fit(data, factors).residual_report(data)
    return output


def test_autocorrelation_pairs_actual_timestamps_and_skips_missing_values():
    rng = np.random.default_rng(0)
    x = ar1(rng, (4000, 2), 0.6)
    times = np.arange(4000.0)
    assert_allclose(autocorrelation(times, x, [1.0, 2.0]), [[0.6] * 2, [0.36] * 2], atol=0.05)
    # Samples one step apart in the array but two seconds apart in time are lag two.
    assert_allclose(autocorrelation(times[::2], x[::2], [2.0]), [[0.36] * 2], atol=0.06)
    assert np.isnan(autocorrelation(times[::2], x[::2], [1.0])).all()
    # A gap is not bridged: no pair straddles it.
    kept = np.r_[0:10, 500:510]
    assert np.isnan(autocorrelation(times[kept], x[kept], [20.0])).all()
    masked = x.copy()
    masked[::3, 0] = np.nan
    assert_allclose(autocorrelation(times, masked, [1.0])[0], [0.6, 0.6], atol=0.06)


def test_exponential_timescale_is_exact_and_says_when_it_is_undefined():
    lags = np.arange(1.0, 9.0)
    timescale, slow, status = exponential_timescale(lags, 0.5 * np.exp(-lags / 4.0))
    assert status == "ok"
    assert_allclose([timescale, slow], [4.0, 0.5])
    for acf, reason in (
        (np.full(8, 0.01), "lag-one autocorrelation below"),
        (np.r_[0.3, np.zeros(7)], "fewer than two lags"),
        (np.r_[0.2, 0.3, 0.4, np.zeros(5)], "does not decay"),
        (np.full(8, np.nan), "too few timestamp pairs"),
    ):
        timescale, slow, status = exponential_timescale(lags, acf)
        assert timescale is None and slow is None and reason in status


def test_variance_shares_and_slow_residual_are_reported_per_modality_and_feature():
    rng = np.random.default_rng(1)
    times = np.arange(600.0)
    records = []
    for subject in "abc":
        fitted = rng.normal(size=(600, 3))
        noise = ar1(rng, (600, 3), 0.8) * [1.0, 1.0, 2.0]
        records.append(ResidualSeries(subject, "run", "m", times, fitted + noise, fitted))
    entry = residual_report(records, n_shifts=5)["variance"]["m"]
    assert_allclose(entry["per_feature"]["shared"], [0.5, 0.5, 0.2], atol=0.08)
    assert_allclose(entry["per_feature"]["residual"], [0.5, 0.5, 0.8], atol=0.08)
    assert_allclose(entry["residual"], 0.6, atol=0.08)
    assert_allclose(entry["acf_lag1"], 0.8, atol=0.05)
    assert_allclose(entry["acf"][2.0], 0.64, atol=0.07)
    assert entry["timescale_status"] == "ok"
    assert_allclose(entry["timescale"], -1 / np.log(0.8), rtol=0.3)
    assert entry["slow_fraction"] > 0.8
    assert entry["sample_interval"] == 1.0 and entry["n_participants"] == 3


def test_participant_correlation_detects_a_common_component_whatever_its_sign():
    rng = np.random.default_rng(2)
    times = np.arange(400.0)
    common = ar1(rng, (400, 4), 0.5)
    signs = rng.choice([-1.0, 1.0], size=(6, 4))
    independent = [series(ar1(rng, (400, 4), 0.5), times, s) for s in range(6)]
    leaking = [
        series(ar1(rng, (400, 4), 0.5) + sign * common, times, s) for s, sign in enumerate(signs)
    ]
    null = residual_report(independent, n_shifts=50)["participant_correlation"]["m"]
    found = residual_report(leaking, n_shifts=50)["participant_correlation"]["m"]
    assert null["status"] == "ok"
    assert null["rms"] < null["rms_null_q95"] + 0.03
    assert abs(null["correlation"]) < 0.08
    # Mixed signs largely cancel in the signed mean but not in the root mean square.
    assert found["rms"] > found["rms_null_q95"] + 0.1
    assert found["rms_p_shift"] <= 0.05
    assert abs(found["correlation"]) < found["rms"] / 2
    aligned = [series(ar1(rng, (400, 4), 0.5) + common, times, s) for s in range(6)]
    signed = residual_report(aligned, n_shifts=50)["participant_correlation"]["m"]
    assert signed["correlation"] > signed["null_q95"] + 0.2 and signed["p_shift"] <= 0.05
    assert signed["per_feature"].shape == (4,) and np.all(signed["per_feature"] > 0.2)


def test_participant_correlation_needs_common_timestamps_and_says_so():
    rng = np.random.default_rng(3)
    common = rng.normal(size=(300, 2))
    times = np.arange(300.0)
    offset = [
        series(common + rng.normal(size=(300, 2)), times + 0.5 * (s % 2), s) for s in range(4)
    ]
    entry = residual_report(offset, n_shifts=5)["participant_correlation"]["m"]
    assert entry["correlation"] is None and entry["rms"] is None
    assert "do not lie on a common regular grid" in entry["status"]
    jittered = [series(common, times + rng.uniform(-0.3, 0.3, size=300), s) for s in range(4)]
    entry = residual_report(jittered, n_shifts=5)["participant_correlation"]["m"]
    assert entry["correlation"] is None and entry["rms"] is None
    assert "do not lie on a common regular grid" in entry["status"]
    alone = residual_report([series(common, times)], n_shifts=5)
    assert "fewer than two participants" in alone["participant_correlation"]["m"]["status"]
    assert alone["modality_cross_correlation"] == {}
    # Masked stretches and a participant missing the end are handled on the common grid.
    partial = []
    for s in range(4):
        values = common + rng.normal(size=(300, 2))
        values[40 * s : 40 * s + 20] = np.nan
        partial.append(series(values[: 300 - 10 * s], times[: 300 - 10 * s], s))
    entry = residual_report(partial, n_shifts=20)["participant_correlation"]["m"]
    assert entry["status"] == "ok" and entry["rms"] > entry["rms_null_q95"] + 0.2
    # Four participants share 300 s: each is shifted inside its own 75 s slot.
    assert entry["min_shift_used"] == 30.0
    crowded = [series(common + rng.normal(size=(300, 2)), times, s) for s in range(20)]
    entry = residual_report(crowded, n_shifts=5)["participant_correlation"]["m"]
    assert entry["min_shift_used"] == 7.0


def test_cross_correlation_finds_the_delay_between_modalities_on_different_clocks():
    rng = np.random.default_rng(4)
    fine = np.arange(-20.0, 420.0, 0.5)
    fast, slow = np.arange(0.0, 400.0, 1.0), np.arange(0.5, 400.0, 2.0)
    records = []
    for s in range(5):
        own = np.convolve(rng.normal(size=len(fine) + 8), np.ones(9) / 3.0, "valid")
        for modality, times, delay, count in (("x", fast, 0.0, 4), ("y", slow, 4.0, 3)):
            signal = np.interp(times - delay, fine, own)
            values = np.outer(signal, rng.normal(size=count)) + rng.normal(size=(len(times), count))
            records.append(series(values, times, s, modality))
    report = residual_report(records, max_lag=10.0, n_shifts=100)
    entry = report["modality_cross_correlation"]["x", "y"]
    assert entry["status"] == "ok" and entry["resampled"] and entry["step"] == 2.0
    assert_allclose(entry["lags"], np.arange(-10.0, 11.0, 2.0))
    # Positive lag: the second modality follows the first.
    assert entry["peak_lag"] == 4.0
    assert entry["peak"] > entry["null_peak_q95"] + 0.1 and entry["p_shift"] <= 0.02
    assert entry["null_mean"] < entry["null_q95"] < entry["null_peak_q95"]
    assert entry["n_participants"] == 5
    # The same series against unrelated ones stay at the reference level.
    unrelated = [
        r
        if r.modality == "x"
        else series(rng.normal(size=r.observed.shape), r.times, r.subject, "y")
        for r in records
    ]
    entry = residual_report(unrelated, n_shifts=100)["modality_cross_correlation"]["x", "y"]
    assert entry["peak"] < entry["null_peak_q95"] + 0.02
    short = residual_report(records, max_lag=1.0, n_shifts=5)["modality_cross_correlation"]
    assert "max_lag is shorter than the common sampling step" in short["x", "y"]["status"]


def test_report_is_reproducible_validates_settings_and_keeps_detail_optional():
    rng = np.random.default_rng(5)
    times = np.arange(200.0)
    records = [series(rng.normal(size=(200, 2)), times, s, m) for s in range(3) for m in ("x", "y")]
    first = residual_report(records, n_shifts=20, random_state=3)
    again = residual_report(records, n_shifts=20, random_state=3)
    other = residual_report(records, n_shifts=20, random_state=4)
    key = ("x", "y")
    for name in ("correlation", "null_q95", "p_shift", "rms", "rms_null_q95", "rms_p_shift"):
        assert (
            first["participant_correlation"]["x"][name]
            == (again["participant_correlation"]["x"][name])
        )
    assert (
        first["participant_correlation"]["x"]["rms_null_q95"]
        != (other["participant_correlation"]["x"]["rms_null_q95"])
    )
    assert (
        first["modality_cross_correlation"][key]["null_q95"]
        == (again["modality_cross_correlation"][key]["null_q95"])
    )
    assert (
        first["modality_cross_correlation"][key]["null_q95"]
        != (other["modality_cross_correlation"][key]["null_q95"])
    )
    assert set(first) == {
        "variance",
        "participant_correlation",
        "modality_cross_correlation",
        "settings",
    }
    assert first["settings"]["n_shifts"] == 20 and first["settings"]["random_state"] == 3
    assert "participants" not in first["variance"]["x"]
    detailed = residual_report(records, n_shifts=5, detail=True)
    assert set(detailed["variance"]["x"]["participants"]) == {0, 1, 2}
    assert set(detailed["participant_correlation"]["x"]["participants"]) == {0, 1, 2}
    assert detailed["modality_cross_correlation"][key]["participants"][0].shape == (21,)
    assert detailed["residuals"][0]["run"]["x"]["residual"].shape == (200, 2)
    for bad in (
        dict(acf_lags=[]),
        dict(acf_lags=[1.0, -2.0]),
        dict(max_lag=0.0),
        dict(min_shift=np.inf),
        dict(n_shifts=0),
        dict(n_shifts=2.5),
    ):
        with pytest.raises(ValueError):
            residual_report(records, **bad)
    with pytest.raises(ValueError, match="no residuals"):
        residual_report([])


def test_residual_is_observed_minus_the_reconstruction_from_all_observations():
    data = recording(participants=4, samples=80, seed=3)
    mask = np.ones((80, 5), bool)
    mask[10:20, 1] = False
    mask[30] = False
    values = np.array(data["p0"]["run0"]["a"].values)
    values[~mask] = 1e9
    data["p0"]["run0"]["a"] = TimeSeries(values, data["p0"]["run0"]["a"].times, mask)
    model = fit(data, 2)
    before = copy.deepcopy({k: v for k, v in vars(model).items() if k.endswith("_")})
    report = model.residual_report(data, n_shifts=10, detail=True)
    latent = model.infer_latent(data, times=np.arange(80.0))
    for subject in data:
        for modality in "ab":
            stats = model.preprocessing_[subject][modality]
            series_ = data[subject]["run0"][modality]
            observed = (series_.values - stats["mean"]) / stats["scale"]
            fitted = latent[subject]["run0"].values @ model.loadings_[subject][modality].T
            expected = np.where(series_.mask, observed - fitted, np.nan)
            actual = report["residuals"][subject]["run0"][modality]["residual"]
            assert_allclose(actual, expected, atol=1e-9)
    assert np.isnan(report["residuals"]["p0"]["run0"]["a"]["residual"][30]).all()
    # Reading the residuals leaves the fit untouched.
    for name, value in before.items():
        assert repr(value) == repr(getattr(model, name)), name
    assert report["settings"]["units"].startswith("standardized")


def test_clean_fit_stays_at_the_reference_level(reports):
    report = reports["clean"]
    for entry in report["participant_correlation"].values():
        assert entry["status"] == "ok"
        assert entry["rms"] - entry["rms_null_mean"] < 0.05
    cross = report["modality_cross_correlation"]["a", "b"]
    assert cross["peak"] - cross["null_peak_q95"] < 0.02
    for entry in report["variance"].values():
        assert 0.3 < entry["shared"] < 0.7 and 0.3 < entry["residual"] < 0.7
        assert abs(entry["acf_lag1"]) < 0.1 and entry["timescale"] is None
        assert "lag-one autocorrelation below" in entry["timescale_status"]


def test_too_few_factors_leave_residuals_correlated_across_participants(reports):
    for entry in reports["underfit"]["participant_correlation"].values():
        assert entry["rms"] > entry["rms_null_q95"] + 0.02
        assert entry["rms_p_shift"] <= 0.02
    assert (
        reports["underfit"]["variance"]["a"]["shared"] < reports["clean"]["variance"]["a"]["shared"]
    )


def test_participant_specific_signal_links_modalities_at_its_delay(reports):
    cross = reports["private"]["modality_cross_correlation"]["a", "b"]
    assert cross["peak_lag"] == 3.0
    assert cross["peak"] > cross["null_peak_q95"] + 0.05 and cross["p_shift"] <= 0.02
    assert not cross["resampled"]
    for entry in reports["private"]["participant_correlation"].values():
        assert entry["rms"] - entry["rms_null_mean"] < 0.05


def test_autocorrelated_noise_shows_as_slow_residual_only(reports):
    report = reports["ar"]
    for entry in report["variance"].values():
        assert entry["acf_lag1"] > 0.5 and entry["timescale_status"] == "ok"
        assert 1.0 < entry["timescale"] < 6.0 and entry["slow_fraction"] > 0.7
    for entry in report["participant_correlation"].values():
        assert entry["rms"] - entry["rms_null_mean"] < 0.05
    cross = report["modality_cross_correlation"]["a", "b"]
    assert cross["peak"] - cross["null_peak_q95"] < 0.02
