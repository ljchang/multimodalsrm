"""Scan-seeded GP lag priors and the shared timing report."""

import warnings

import numpy as np
import pytest

from multimodalsrm import Gamma, Gaussian, Identity, Response, TimeSeries, peak_delay, scan_lags
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    Prior,
    SearchConfig,
    lag_priors,
)
from multimodalsrm.data import readonly_array

WIDTHS = {"A": 0.4, "B": 0.8, "C": 1.2}
TRUE = {"A": 0.0, "B": 8.0, "C": 3.0}


def recording(seed=0, duration=60.0):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.02, 0.3, 12)
    amplitude, phase = rng.normal(size=12), rng.uniform(0, 6.28, 12)
    clocks = {"A": (0.0, 0.5), "B": (0.25, 0.5), "C": (0.0, 1.0)}
    data = {}
    for index in range(2):
        streams = {}
        for modality, (start, step) in clocks.items():
            times = np.arange(start, duration, step)
            angle = 2 * np.pi * frequency * (times - TRUE[modality])[:, None] + phase
            signal = (amplitude * np.sin(angle)).sum(-1)
            loading = np.random.default_rng([seed, index, ord(modality)]).normal(size=3)
            values = signal[:, None] * loading + rng.normal(size=(len(times), 3))
            streams[modality] = TimeSeries(values, times)
        data[f"s{index}"] = {"run": streams}
    return data


def fit(responses, filters):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BayesianMultimodalSRM(
            features=1,
            responses=responses,
            priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0), loading_sd=2.0, filters=filters),
            length_scale=2.0,
            inference="map",
            linear_algebra="grouped",
            max_observations=5000,
            search=SearchConfig(starts=1, maxiter=600),
            random_state=0,
        ).fit(recording())


def gaussian(**settings):
    return {
        m: Response(Gaussian(w), pooling="shared", fixed={"width": w}, **settings)
        for m, w in WIDTHS.items()
    }


def test_lag_priors_follow_the_seeded_lags():
    responses = gaussian()
    responses["A"] = Response(Identity(), estimate=False, pooling="shared")
    responses["C"] = Response(Gamma(3.0, 1.0, 2.5), pooling="shared", fixed={"shape": 3.0})
    priors = lag_priors(responses, sd=0.7)
    assert set(priors) == {"B", "C"}
    assert priors["B"]["lag"] == Prior.normal(0.0, 0.7)
    assert priors["C"]["lag"] == Prior.normal(2.5, 0.7)
    with pytest.raises(ValueError, match="sd"):
        lag_priors(responses, sd=0.0)


@pytest.fixture(scope="module")
def seeded_fit():
    responses = scan_lags(recording()).seed_responses(gaussian())
    return fit(responses, lag_priors(responses))


def test_single_start_reaches_the_scanned_basin_and_a_wide_start_does_not(seeded_fit):
    report = seeded_fit.timing_report(reference="A")
    assert report["relative_peak_delay"]["B"] == pytest.approx(8.0, abs=0.2)
    assert report["relative_peak_delay"]["C"] == pytest.approx(3.0, abs=0.2)
    assert report["lags_at_bounds"] == []
    wide = gaussian(bounds={"lag": (-15.0, 15.0)})
    unseeded = fit(wide, {m: {"lag": Prior.normal(0.0, 5.0)} for m in wide})
    missed = unseeded.timing_report(reference="A")["relative_peak_delay"]
    assert abs(missed["B"] - 8.0) > 1.0 or abs(missed["C"] - 3.0) > 1.0


def test_map_report_matches_relative_lags_for_symmetric_responses(seeded_fit):
    report = seeded_fit.timing_report()
    assert report["convention"] == "first_modality" and report["reference"] == "A"
    assert report["uncertainty"] == "conditional_on_MAP"
    # Both summaries show the first modality at zero.
    lags = {m: float(v[0, 0]) for m, v in seeded_fit.relative_lag_draws().items()}
    assert lags["A"] == 0.0
    assert report["relative_peak_delay"] == pytest.approx(lags)
    assert report["pairwise_difference"]["B - A"] == pytest.approx(lags["B"] - lags["A"])
    with pytest.raises(ValueError, match="reference"):
        seeded_fit.timing_report(reference="missing")


def test_posterior_draws_are_summarized_per_draw(seeded_fit):
    model = seeded_fit
    original = model.parameter_draws_, model.configuration_["inference"]
    rng = np.random.default_rng(4)
    draws = np.tile(np.asarray(model.map_parameters_), (2, 6, 1))
    index = {m: model.problem_.indices[("filter", m, "lag")] for m in WIDTHS}
    for column in index.values():
        draws[..., column] += rng.normal(scale=0.1, size=(2, 6))
    try:
        model.parameter_draws_ = readonly_array(draws)
        model.configuration_["inference"] = "posterior"
        report = model.timing_report(reference="A", max_draws=None)
    finally:
        model.parameter_draws_, model.configuration_["inference"] = original
    flat = draws.reshape(-1, draws.shape[-1])
    expected = flat[:, index["B"]] - flat[:, index["A"]]
    summary = report["relative_peak_delay"]["B"]
    np.testing.assert_allclose(summary["draws"], expected)
    assert summary["mean"] == pytest.approx(expected.mean())
    assert summary["sd"] == pytest.approx(expected.std())
    assert report["relative_peak_delay"]["A"]["sd"] == 0.0
    np.testing.assert_allclose(report["clock_origin"]["draws"], flat[:, index["A"]])


def test_causal_responses_report_their_peak_not_their_onset():
    kernel = Gamma(3.0, 0.8, 1.2)
    assert peak_delay(kernel) == pytest.approx(1.2 + 2 * 0.8)
    assert peak_delay(kernel.with_parameters(scale=1.0, lag=0.8)) == pytest.approx(2.8)


def test_fits_saved_with_mean_centred_lags_still_load_and_keep_that_reference(seeded_fit):
    import copy

    from multimodalsrm.bayesian.persistence import _model_state, _restore_model

    state = copy.deepcopy(_model_state(seeded_fit))
    convention = state["fit"]["configuration"]["reference_convention"]
    assert convention.pop("relative_lag_reference") == "A"
    convention["relative_lag_definition"] = "modality_lag_minus_mean_lag"
    restored, _ = _restore_model(state)
    lags = {m: float(v[0, 0]) for m, v in restored.relative_lag_draws().items()}
    assert sum(lags.values()) == pytest.approx(0.0, abs=1e-9)
    current = {m: float(v[0, 0]) for m, v in seeded_fit.relative_lag_draws().items()}
    assert lags["B"] - lags["A"] == pytest.approx(current["B"] - current["A"])
    # The report still needs one modality at zero and falls back to the first.
    assert restored.timing_report()["reference"] == "A"


def test_saved_alphabetical_fixed_reference_is_kept():
    from multimodalsrm.bayesian.reference import keep_saved_reference

    current = dict(
        mode="automatic_fixed_lag",
        reference_modality="rating",
        reference_family="Identity",
        reference_lag_seconds=0.0,
        reference_width="fixed",
        relative_lag_definition="modality_lag_minus_reference_lag",
    )
    saved = dict(current, reference_modality="brain", reference_family="Gaussian")
    saved.update(reference_lag_seconds=1.0, reference_shape_parameters_estimated=[])
    convention = dict(current)
    keep_saved_reference(convention, saved)
    assert convention == saved
    # An explicit reference, or a matching one, is never replaced.
    for other in (dict(saved, mode="explicit_fixed_lag"), dict(current), None):
        convention = dict(current)
        keep_saved_reference(convention, other)
        assert convention == current
