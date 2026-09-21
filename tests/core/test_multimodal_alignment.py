"""Alignment scores need independent participants and honest denominators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import multimodalsrm as multimodal
from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import GaussianMixtureSeries


def series(values, masks=None, times=None):
    values = np.asarray(values, float)
    if values.ndim == 2:
        values = values[..., None]
    return {
        str(i): TimeSeries(
            v,
            np.arange(len(v)) if times is None else times,
            None if masks is None else masks[i],
        )
        for i, v in enumerate(values)
    }


def evaluator(name):
    assert callable(getattr(multimodal, name, None)), f"missing {name} evaluator"
    return getattr(multimodal, name)


def test_temporal_isc_uses_other_subjects_and_common_feature_masks():
    values = np.random.default_rng(91).normal(size=(3, 12, 2))
    masks = np.ones(values.shape, bool)
    masks[1, 2, 0] = False
    values[1, 2, 0] = np.nan
    report = evaluator("temporal_isc")(series(values, masks))
    assert report["n_common_valid_per_feature"] == [11, 12]
    expected = []
    for s in range(3):
        row = []
        for f in range(2):
            valid = masks[:, :, f].all(axis=0)
            other = np.mean(np.delete(values, s, axis=0)[:, valid, f], axis=0)
            row.append(np.corrcoef(values[s, valid, f], other)[0, 1])
        assert_allclose(report["subjects"][str(s)]["correlations"], row, atol=1e-14)
        expected.append(row)
    assert_allclose(
        report["mean_fisher_z_correlation_per_feature"],
        np.tanh(np.arctanh(expected).mean(axis=0)),
        atol=1e-14,
    )


def test_temporal_isc_constant_and_insufficient_are_not_successes():
    data = series([[1, 1, 1, 1], [0, 1, 2, 3], [3, 2, 1, 0]])
    report = evaluator("temporal_isc")(data)
    assert report["subjects"]["0"]["correlations"] == [None]
    assert report["subjects"]["0"]["reasons"] == ["constant_signal"]
    assert report["mean_fisher_z_correlation_per_feature"] == [None]
    short = evaluator("temporal_isc")(series([[1, 2], [2, 4]]))
    assert short["subjects"]["0"]["reasons"] == ["insufficient_support"]


def test_time_segment_scores_match_independent_brute_force():
    rng = np.random.default_rng(19)
    base = rng.normal(size=(25, 2))
    values = np.stack([base + rng.normal(scale=0.9, size=base.shape) for _ in range(3)])
    width = 4
    report = evaluator("time_segment_matching")(series(values), window_size=width)
    starts = np.arange(len(base) - width + 1)
    for s in range(3):
        reference = np.mean(np.delete(values, s, axis=0), axis=0)
        credits, chances = [], []
        for i, row in zip(starts, report["subjects"][str(s)]["queries"]):
            candidates = [j for j in starts if i == j or abs(i - j) >= width]
            scores = [
                np.corrcoef(values[s, i : i + width].ravel(), reference[j : j + width].ravel())[
                    0, 1
                ]
                for j in candidates
            ]
            winners = np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-12))
            correct = candidates.index(i)
            credit = float(correct in winners) / len(winners)
            assert row["n_candidates"] == len(candidates)
            assert row["credit"] == credit
            assert row["chance"] == 1 / len(candidates)
            credits.append(credit)
            chances.append(1 / len(candidates))
        assert_allclose(report["subjects"][str(s)]["accuracy"], np.mean(credits))
        assert_allclose(report["subjects"][str(s)]["chance"], np.mean(chances))


def test_matching_unique_windows_and_shift_control():
    base = np.random.default_rng(212).normal(size=70)
    score = evaluator("time_segment_matching")
    matched = score(series([base, base, base]), window_size=7)
    assert all(s["accuracy"] == 1.0 for s in matched["subjects"].values())
    shifted = score(series([np.roll(base, 23), base, base]), window_size=7)
    assert shifted["subjects"]["0"]["accuracy"] < 0.2


def test_matching_ties_receive_fractional_credit_not_first_index():
    # Every nonconstant linear window has Pearson correlation one.
    base = np.arange(14.0)
    report = evaluator("time_segment_matching")(series([base, base]), window_size=3)
    for subject in report["subjects"].values():
        assert subject["accuracy"] == pytest.approx(subject["chance"])
        assert all(row["n_ties"] == row["n_candidates"] for row in subject["queries"])


def test_matching_masks_do_not_stitch_across_missing_rows():
    values = np.random.default_rng(3).normal(size=(3, 15, 1))
    masks = np.ones(values.shape, bool)
    masks[0, 6] = False
    values[0, 6] = np.nan
    report = evaluator("time_segment_matching")(series(values, masks), window_size=4)
    assert report["window_starts"] == [0, 1, 2, 7, 8, 9, 10, 11]
    assert report["n_common_valid_rows"] == 14
    assert report["n_excluded_windows"] == 4


def test_matching_reports_undefined_candidates_and_short_support():
    score = evaluator("time_segment_matching")
    report = score(series([[1] * 10, list(range(10))]), window_size=3)
    assert all(
        s["accuracy"] is None and s["n_scored_queries"] == 0 for s in report["subjects"].values()
    )
    short = score(series([list(range(4)), list(range(4))]), window_size=4)
    assert short["subjects"]["0"]["queries"][0]["reason"] == "insufficient_candidates"
    empty = score(series([[1, 2], [3, 4]]), window_size=4)
    assert empty["subjects"]["0"]["accuracy"] is None
    assert empty["subjects"]["0"]["reason"] == "no_valid_windows"


@pytest.mark.parametrize("name", ["temporal_isc", "time_segment_matching"])
def test_evaluation_rejects_misaligned_inputs_and_pooled_latents(name):
    evaluate = evaluator(name)
    kwargs = {"window_size": 3} if name == "time_segment_matching" else {}
    data = series([list(range(12)), list(range(12))])
    with pytest.raises(ValueError, match="two"):
        evaluate({"0": data["0"]}, **kwargs)
    shifted = {**data, "1": TimeSeries(data["1"].values, data["1"].times + 0.1)}
    with pytest.raises(ValueError, match="clock"):
        evaluate(shifted, **kwargs)
    expanded = {
        **data,
        "1": TimeSeries(np.tile(data["1"].values, (1, 2)), data["1"].times),
    }
    with pytest.raises(ValueError, match="feature"):
        evaluate(expanded, **kwargs)
    pooled = GaussianMixtureSeries(
        np.ones((1, 12, 1)),
        np.ones((1, 12, 1)),
        np.arange(12),
        np.ones(12, bool),
        {"conditioning_mode": "frozen"},
    )
    with pytest.raises(ValueError, match="independent"):
        evaluate({"0": pooled, "1": pooled}, **kwargs)


def test_matching_requires_uniform_query_grid():
    data = series([list(range(8)), list(range(8))], times=[0, 1, 2, 3, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="regular"):
        evaluator("time_segment_matching")(data, window_size=3)


def test_large_finite_signals_do_not_overflow_reference_means():
    base = np.random.default_rng(91).uniform(0.6, 1.4, 24) * 1e308
    data = series([base, base, base])
    with np.errstate(over="raise", invalid="raise"):
        isc = evaluator("temporal_isc")(data)
        matching = evaluator("time_segment_matching")(data, window_size=4)
    for s in data:
        assert isc["subjects"][s]["correlations"][0] == pytest.approx(1)
        assert isc["subjects"][s]["reasons"] == [None]
        assert matching["subjects"][s]["accuracy"] == 1
