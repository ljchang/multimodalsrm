"""GP residual diagnostics reuse prediction and respond to the right departures."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.prediction import result, selected_draws
from multimodalsrm.bayesian.residuals import residual_series

from ..reference.residual_scenarios import recording

SCENARIOS = {
    "clean": (dict(), 2),
    "underfit": (dict(), 1),
    "private": (dict(private=1.0, delay=3), 2),
    "ar": (dict(ar=0.7), 2),
}


def fit(data, factors, *, linear_algebra="grouped", maxiter=300, **kwargs):
    model = BayesianMultimodalSRM(
        features=factors,
        anchor=("p0", "a", 0) if factors == 1 else None,
        responses={m: Response(Identity(), estimate=False) for m in "ab"},
        priors=BayesianPriors(
            noise=Prior.lognormal(np.log(0.5), 1.0), loading_sd=1.0, offset_sd=1.0
        ),
        length_scale=3.0,
        inference="map",
        linear_algebra=linear_algebra,
        max_observations=100_000,
        search=SearchConfig(starts=1, maxiter=maxiter),
        random_state=0,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(data)


@pytest.fixture(scope="module")
def reports():
    # Two modalities of 100 samples: 200 time nodes in each run.
    return {
        name: fit(
            recording(participants=6, runs=2, samples=100, seed=1, **departure), factors
        ).residual_report()
        for name, (departure, factors) in SCENARIOS.items()
    }


def small(seed=3):
    data = recording(participants=2, samples=40, features=(3, 2), seed=seed)
    mask = np.ones((40, 3), bool)
    mask[5:9, 1] = False
    mask[20] = False
    values = np.array(data["p0"]["run0"]["a"].values)
    values[~mask] = 1e9
    data["p0"]["run0"]["a"] = TimeSeries(values, data["p0"]["run0"]["a"].times, mask)
    return data


@pytest.mark.parametrize(
    "linear_algebra, factors, extra",
    [
        ("dense", 2, {}),
        ("grouped", 2, {}),
        ("state_space", 2, {}),
        ("grouped", 2, dict(noise_timescales={"a": 3.0})),
        ("dense", 1, dict(run_baseline_sd={"a": 0.5})),
    ],
)
def test_reconstruction_is_the_filtered_signal_mean_of_prediction(linear_algebra, factors, extra):
    data = small()
    model = fit(data, factors, linear_algebra=linear_algebra, maxiter=40, **extra)
    draws = selected_draws(model, 200)
    series = residual_series(model)
    assert {(s.subject, s.modality) for s in series} == {(p, m) for p in data for m in "ab"}
    for s in series:
        source = data[s.subject][s.run][s.modality]
        rows = source.mask.any(axis=1)
        assert_allclose(s.times, source.times[rows])
        assert_allclose(s.observed, np.where(source.mask, source.values, np.nan)[rows])
        assert np.array_equal(np.isfinite(s.fitted), source.mask[rows])
        for f in (0, s.observed.shape[1] - 1):
            direct = result(
                model, s.run, s.times, draws, [(s.subject, s.modality, f)], include_noise=False
            ).values[:, 0]
            keep = source.mask[rows, f]
            assert_allclose(s.fitted[keep, f], direct[keep], atol=1e-10)


def test_several_parameter_draws_average_their_conditional_means():
    model = fit(small(), 2, maxiter=40)
    x = model.map_parameters_
    model.parameter_draws_ = np.stack([x, 1.02 * x])[None]
    draws = selected_draws(model, 200)
    assert len(draws) == 2
    for s in residual_series(model):
        direct = np.column_stack(
            [
                result(
                    model, s.run, s.times, draws, [(s.subject, s.modality, f)], include_noise=False
                ).values[:, 0]
                for f in range(s.observed.shape[1])
            ]
        )
        keep = np.isfinite(s.fitted)
        assert_allclose(s.fitted[keep], direct[keep], atol=1e-10)


def test_report_matches_the_regularized_estimator_in_structure_and_changes_nothing():
    data = small()
    model = fit(data, 2, maxiter=40)
    before = model.map_parameters_.copy()
    report = model.residual_report(n_shifts=10, detail=True)
    assert_allclose(model.map_parameters_, before, atol=0)
    regularized = MultimodalSRM(
        features=2,
        latent_dt=1.0,
        responses={m: Response(Identity(), estimate=False) for m in "ab"},
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = regularized.fit(data).residual_report(data, n_shifts=10, detail=True)
    assert set(report) == set(other)
    for section in ("variance", "participant_correlation", "modality_cross_correlation"):
        assert set(report[section]) == set(other[section])
        for name in report[section]:
            assert set(report[section][name]) == set(other[section][name])
    assert report["settings"]["units"].startswith("supplied units")
    for entry in report["variance"].values():
        assert 0.0 < entry["residual"] < 1.0 and 0.0 < entry["shared"] < 1.0
    with pytest.raises(Exception, match="not fitted"):
        BayesianMultimodalSRM(
            priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0))
        ).residual_report()


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


def test_too_few_factors_leave_residuals_correlated_across_participants(reports):
    for entry in reports["underfit"]["participant_correlation"].values():
        assert entry["rms"] > entry["rms_null_q95"] + 0.02
        assert entry["rms_p_shift"] <= 0.02


def test_participant_specific_signal_links_modalities_at_its_delay(reports):
    cross = reports["private"]["modality_cross_correlation"]["a", "b"]
    assert cross["peak_lag"] == 3.0
    assert cross["peak"] > cross["null_peak_q95"] + 0.05 and cross["p_shift"] <= 0.02
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
