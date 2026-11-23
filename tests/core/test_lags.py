"""Lag scans, response seeding and reference-free timing reports."""

import pickle
import warnings

import numpy as np
import pytest

from multimodalsrm import (
    BatemanSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    LagScan,
    MultimodalSRM,
    Response,
    SampledKernel,
    TimeSeries,
    peak_delay,
    scan_lags,
)

WIDTHS = {"A": 0.4, "B": 0.8, "C": 1.2}
CLOCKS = {"A": (0.0, 0.5), "B": (0.25, 0.5), "C": (0.0, 1.0)}


def recording(seed, lags, *, noise=1.0, duration=120.0, period=None, subjects=3, skip=()):
    """A two-factor latent seen with per-modality delays on unequal clocks."""
    rng = np.random.default_rng(seed)
    if period is None:
        frequency = rng.uniform(0.02, 0.3, (2, 12))
    else:
        frequency = rng.uniform(0.95 / period, 1.05 / period, (2, 2))
    amplitude, phase = rng.normal(size=frequency.shape), rng.uniform(0, 6.28, frequency.shape)

    def latent(times):
        angle = 2 * np.pi * frequency * np.asarray(times)[:, None, None] + phase
        return (amplitude * np.sin(angle)).sum(-1)

    data = {}
    for index in range(subjects):
        subject = f"s{index}"
        data[subject] = {}
        for run in range(2):
            streams = {}
            for modality in lags:
                if (subject, modality) in skip:
                    continue
                start, step = CLOCKS[modality]
                times = np.arange(start, duration, step)
                loading = np.random.default_rng([seed, index, ord(modality)]).normal(size=(4, 2))
                values = latent(times - lags[modality] + 1000 * run) @ loading.T
                streams[modality] = TimeSeries(
                    values + rng.normal(size=values.shape) * noise, times
                )
            data[subject][f"run{run}"] = streams
    return data


def gaussian_responses(lag=0.0, **settings):
    return {
        m: Response(Gaussian(w, lag), pooling="shared", fixed={"width": w}, **settings)
        for m, w in WIDTHS.items()
    }


def fit(data, **settings):
    options = dict(features=2, latent_dt=0.5, max_iter=300, random_state=0)
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultimodalSRM(**options).fit(data)


TRUE = {"A": 0.0, "B": 10.0, "C": 4.0}


def test_peak_delay_by_family():
    assert peak_delay(Identity()) == 0.0
    assert peak_delay(Gaussian(0.7, -1.3)) == -1.3
    assert peak_delay(Gamma(3.0, 1.5, 0.4)) == pytest.approx(0.4 + 2 * 1.5)
    assert peak_delay(Gamma(1.0, 2.0, 0.4)) == pytest.approx(0.4)
    double = DoubleGamma(lag=0.5)
    grid = np.linspace(*double.support, 200_001)
    assert peak_delay(double) == pytest.approx(grid[np.argmax(double.evaluate(grid))], abs=1e-3)
    rise, decay = 0.7, 3.0
    expected = np.log(decay / rise) * rise * decay / (decay - rise)
    assert peak_delay(BatemanSCR(rise, decay, 1.0)) == pytest.approx(1.0 + expected, abs=1e-5)
    sampled = SampledKernel(np.array([-1.0, 0.0, 0.5, 2.0]), np.array([0.0, 0.2, 1.0, -3.0]))
    assert peak_delay(sampled) == 0.5
    with pytest.raises(ValueError, match="kernel"):
        peak_delay("gaussian")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scan_recovers_relative_delays_far_outside_default_bounds(seed):
    scan = scan_lags(recording(seed, TRUE))
    assert scan.reliable and scan.notes == ()
    relative = scan.relative_to("A")
    assert relative["B"] == pytest.approx(10.0, abs=0.1)
    assert relative["C"] == pytest.approx(4.0, abs=0.1)
    assert sum(scan.delays.values()) == pytest.approx(0.0, abs=1e-9)
    assert scan.closure_error < 0.25
    assert scan.pairwise["A", "B"]["delay"] == pytest.approx(10.0, abs=0.1)
    assert all(pair["prominence"] > 0.15 for pair in scan.pairwise.values())


def test_scan_flags_noise_and_near_periodic_signals():
    rng = np.random.default_rng(0)
    noise = {
        s: {"run": {m: TimeSeries(rng.normal(size=(200, 3)), np.arange(200) * 0.5) for m in "AB"}}
        for s in ("s0", "s1")
    }
    scan = scan_lags(noise)
    assert not scan.reliable and any("no lag stands out" in note for note in scan.notes)
    assert scan.closure_error is None
    periodic = scan_lags(recording(0, {"A": 0.0, "B": 5.0, "C": 2.0}, period=10.0))
    assert not periodic.reliable
    assert any("near-periodic" in note for note in periodic.notes)
    assert periodic.closure_error > 1.0


def test_scan_handles_masks_and_participants_missing_a_modality():
    data = recording(1, TRUE, skip={("s0", "C"), ("s1", "A")})
    series = data["s2"]["run0"]["B"]
    mask = np.ones_like(series.values, dtype=bool)
    mask[30:50, 1] = False
    mask[:, 3] = False
    values = np.where(mask, series.values, 1e20)
    data["s2"]["run0"]["B"] = TimeSeries(values, series.times, mask)
    scan = scan_lags(data)
    relative = scan.relative_to("A")
    assert relative["B"] == pytest.approx(10.0, abs=0.15)
    assert relative["C"] == pytest.approx(4.0, abs=0.15)
    assert scan.pairwise["A", "C"]["participants"] == 1


def test_scan_reports_modalities_it_cannot_link():
    data = recording(0, {"A": 0.0, "B": 2.0})
    rng = np.random.default_rng(3)
    data["solo"] = {"other": {"C": TimeSeries(rng.normal(size=(100, 2)), np.arange(100.0))}}
    scan = scan_lags(data)
    assert np.isnan(scan.delays["C"]) and not scan.reliable
    assert scan.relative_to("A")["B"] == pytest.approx(2.0, abs=0.1)
    responses = gaussian_responses()
    assert scan.seed_responses(responses)["C"] is responses["C"]


@pytest.mark.parametrize(
    "settings, message",
    [
        (dict(max_lag=0.1, step=0.25), "max_lag"),
        (dict(step=0.0), "max_lag"),
        (dict(components=0), "components"),
        (dict(modalities=["A"]), "at least two"),
        (dict(modalities=["A", "missing"]), "observed"),
    ],
)
def test_scan_validation(settings, message):
    with pytest.raises(ValueError, match=message):
        scan_lags(recording(0, {"A": 0.0, "B": 1.0}, duration=40.0), **settings)


def scan(delays):
    return LagScan(dict(delays), {}, None, 15.0, 0.25, 2)


def test_seeding_takes_the_clock_from_modalities_without_an_estimated_lag():
    responses = {
        "A": Response(Identity(), estimate=False),
        "B": Response(Gaussian(0.8), pooling="shared", bounds={"width": (0.5, 1.0)}),
        "C": Response(Gamma(3.0, 1.0), pooling="shared", fixed={"shape": 3.0}),
        "D": Response(Gaussian(1.0, 0.3), pooling="shared", fixed={"lag": 0.3}),
    }
    seeded = scan({"A": -2.0, "B": 3.0, "C": 0.0, "D": -1.0}).seed_responses(responses, margin=1.5)
    # A peaks at 0 with scan -2; D at 0.3 with scan -1: origin = mean(2.0, 1.3).
    origin = 1.65
    assert seeded["A"] is responses["A"] and seeded["D"] is responses["D"]
    b = seeded["B"].initial_kernel()
    assert b.lag == pytest.approx(3.0 + origin)
    assert seeded["B"].bounds["lag"] == pytest.approx((b.lag - 1.5, b.lag + 1.5))
    assert seeded["B"].bounds["width"] == (0.5, 1.0)
    c = seeded["C"].initial_kernel()
    assert peak_delay(c) == pytest.approx(0.0 + origin)
    assert c.lag == pytest.approx(origin - 2.0)
    assert seeded["C"].fixed == {"shape": 3.0}


def test_seeding_with_every_lag_free_keeps_the_mean_peak_delay():
    responses = gaussian_responses(lag=1.5)
    seeded = scan({"A": -4.0, "B": 6.0, "C": -2.0}).seed_responses(responses)
    lags = {m: r.initial_kernel().lag for m, r in seeded.items()}
    assert lags == pytest.approx({"A": -2.5, "B": 7.5, "C": -0.5})
    assert seeded["B"].bounds["lag"] == pytest.approx((5.5, 9.5))
    with pytest.raises(ValueError, match="margin"):
        scan({"A": 0.0}).seed_responses(responses, margin=0)
    with pytest.raises(ValueError, match="Response"):
        scan({"A": 0.0}).seed_responses({"A": Gaussian()})


def test_scan_initialization_reaches_a_delay_the_default_start_misses():
    data = recording(0, TRUE)
    default = fit(data, responses=gaussian_responses()).timing_report(reference="A")
    assert abs(default["relative_peak_delay"]["B"] - 10.0) > 5.0
    assert default["lags_at_bounds"] == []  # The miss is silent.
    model = fit(data, responses=gaussian_responses(), lag_init="scan")
    report = model.timing_report(reference="A")
    assert report["relative_peak_delay"]["B"] == pytest.approx(10.0, abs=0.1)
    assert report["relative_peak_delay"]["C"] == pytest.approx(4.0, abs=0.1)
    assert max(abs(v) for v in report["scan_disagreement"].values()) < 0.1
    assert report["scan_reliable"] and report["lags_at_bounds"] == []
    assert isinstance(model.lag_scan_, LagScan)
    provenance = model.configuration_["lag_init"]
    assert provenance["method"] == "scan" and provenance["reliable"] and provenance["applied"]
    assert set(provenance["seeded_lags"]) == set("ABC")
    # The constructor's responses are not modified.
    assert model.responses["B"].initial_kernel().lag == 0.0
    assert pickle.loads(pickle.dumps(model)).lag_scan_.delays == model.lag_scan_.delays


def test_supplied_scans_are_applied_only_when_reliable():
    data = recording(0, TRUE)
    delays = {"A": -4.7, "B": 5.3, "C": -0.6}
    model = fit(
        data,
        responses=gaussian_responses(),
        lag_init=LagScan(delays, {}, 0.0, 15.0, 0.25, 2),
    )
    provenance = model.configuration_["lag_init"]
    assert provenance["method"] == "supplied_scan" and provenance["applied"]
    assert model.timing_report(reference="A")["relative_peak_delay"]["B"] == pytest.approx(
        10.0, abs=0.1
    )
    unreliable = LagScan(delays, {}, 0.0, 15.0, 0.25, 2, ("made up",))
    with pytest.warns(UserWarning, match="no reliable relative delays.*made up"):
        model = MultimodalSRM(
            features=2,
            latent_dt=0.5,
            responses=gaussian_responses(),
            lag_init=unreliable,
            max_iter=5,
            random_state=0,
        ).fit(data)
    provenance = model.configuration_["lag_init"]
    assert not provenance["applied"] and provenance["seeded_lags"] == {}
    # Lags and bounds stay as configured rather than narrowing around a guess.
    assert model.responses_["B"].initial_kernel().lag == 0.0
    assert model.responses_["B"].parameter_bounds()["lag"] == (-2.0, 2.0)
    assert model.lag_scan_ is unreliable


def test_scan_initialization_is_skipped_when_it_cannot_apply():
    single = {
        s: {r: {"A": mods["A"]} for r, mods in runs.items()}
        for s, runs in recording(0, TRUE).items()
    }
    model = fit(single, responses={"A": gaussian_responses()["A"]}, lag_init="scan", max_iter=5)
    assert model.configuration_["lag_init"]["method"] == "skipped"
    assert not hasattr(model, "lag_scan_")
    fixed = {m: Response(Identity(), estimate=False) for m in "ABC"}
    model = fit(recording(0, TRUE), responses=fixed, lag_init="scan", max_iter=5)
    assert model.configuration_["lag_init"] is None
    with pytest.raises(ValueError, match="lag_init"):
        MultimodalSRM(latent_dt=0.5, lag_init="xcorr").fit(recording(0, TRUE))
    state = dict(fit(single, responses={"A": gaussian_responses()["A"]}, max_iter=3).__getstate__())
    state.pop("lag_init")
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert restored.lag_init is None


def test_timing_report_is_invariant_to_the_arbitrary_common_offset():
    data = recording(2, {"A": 0.0, "B": 1.0, "C": 0.4})
    first = fit(data, responses=gaussian_responses(lag=0.0)).timing_report()
    second = fit(data, responses=gaussian_responses(lag=1.5)).timing_report()
    assert first["convention"] == "across_modality_mean" and first["reference"] is None
    for name in ("relative_peak_delay", "pairwise_difference"):
        assert first[name] == pytest.approx(second[name], abs=0.02)
    assert first["pairwise_difference"]["B - A"] == pytest.approx(1.0, abs=0.1)
    assert sum(first["relative_peak_delay"].values()) == pytest.approx(0.0, abs=1e-9)
    # Raw lags follow the start; the reported clock origin absorbs the shift.
    shift = second["clock_origin"] - first["clock_origin"]
    assert shift == pytest.approx(1.5, abs=0.05)
    assert second["peak_delay_model_clock"]["A"] - first["peak_delay_model_clock"]["A"] == (
        pytest.approx(shift, abs=0.02)
    )


def test_timing_report_reference_conventions():
    data = recording(2, {"A": 0.0, "B": 1.0, "C": 0.4})
    responses = gaussian_responses()
    responses["A"] = Response(Identity(), estimate=False)
    model = fit(data, responses=responses)
    automatic = model.timing_report()
    assert automatic["reference"] == "A"
    assert automatic["convention"] == "only_modality_without_an_estimated_lag"
    assert automatic["relative_peak_delay"]["A"] == 0.0 and automatic["clock_origin"] == 0.0
    assert automatic["lag_estimated"] == {"A": False, "B": True, "C": True}
    chosen = model.timing_report(reference="C")
    assert chosen["convention"] == "requested_reference"
    assert chosen["relative_peak_delay"]["C"] == 0.0
    assert chosen["pairwise_difference"] == pytest.approx(automatic["pairwise_difference"])
    with pytest.raises(ValueError, match="reference"):
        model.timing_report(reference="missing")
    # Two modalities without an estimated lag already fix the model clock.
    responses["C"] = Response(Identity(), estimate=False)
    report = fit(data, responses=responses).timing_report()
    assert report["reference"] is None and report["clock_origin"] == 0.0
    assert report["convention"] == "model_clock_set_by_modalities_without_estimated_lags"
    assert report["relative_peak_delay"] == report["peak_delay_model_clock"]
    assert report["relative_peak_delay"]["A"] == report["relative_peak_delay"]["C"] == 0.0


def test_timing_report_uses_peaks_and_lists_participants_when_not_shared():
    data = recording(2, {"A": 0.0, "B": 1.0})
    responses = {
        "A": Response(Gamma(3.0, 0.5, -1.0), estimate=False),
        "B": Response(Gaussian(0.8), pooling="none", fixed={"width": 0.8}),
    }
    report = fit(data, responses=responses, max_iter=60).timing_report()
    assert report["reference"] == "A"
    assert report["peak_delay_model_clock"]["A"] == pytest.approx(0.0)  # onset -1, peak +1
    individual = report["participant_relative_peak_delay"]["B"]
    assert set(individual) == {"s0", "s1", "s2"}
    assert np.mean(list(individual.values())) == pytest.approx(report["relative_peak_delay"]["B"])
    assert "A" not in report["participant_relative_peak_delay"]
