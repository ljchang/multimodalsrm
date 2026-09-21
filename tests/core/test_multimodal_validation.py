"""Run partitioning and scale-sensitive scoring must not leak target data."""

import numpy as np
import pytest


def test_metrics_flag_undefined_and_keep_scale_error():
    from multimodalsrm.validation import score_prediction

    score = score_prediction(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[2.0], [4.0], [6.0]]),
        np.ones((3, 1), dtype=bool),
    )
    assert score["correlation"] == pytest.approx(1.0)
    assert score["mse"] == pytest.approx(14 / 3)
    assert score["r2"] == pytest.approx(-6.0)
    constant = score_prediction(np.ones((3, 1)), np.zeros((3, 1)), np.ones((3, 1), bool))
    assert np.isnan(constant["correlation"])
    assert np.isnan(constant["r2"])
    assert set(constant["undefined"]) == {"correlation", "r2"}
    assert constant["mse"] == 1


def test_masked_nonfinite_values_do_not_enter_scores():
    from multimodalsrm.validation import score_prediction

    score = score_prediction(
        np.array([[1.0], [np.nan], [3.0]]),
        np.array([[1.0], [np.nan], [2.0]]),
        np.array([[True], [False], [True]]),
    )
    assert score["n_observations"] == 2
    assert score["mse"] == 0.5


def test_leave_one_run_out_uses_names_and_omits_empty_participants():
    from multimodalsrm.data import TimeSeries
    from multimodalsrm.validation import LeaveOneRunOut

    entry = TimeSeries([[1.0], [2.0]], times=[0, 1])
    data = {
        "a": {"movie-A": {"rating": entry}, "movie-B": {"rating": entry}},
        "b": {"movie-B": {"rating": entry}},
    }
    folds = list(LeaveOneRunOut().split(data))
    assert len(folds) == 2
    for train, test in folds:
        train_runs = {r for runs in train.values() for r in runs}
        test_runs = {r for runs in test.values() for r in runs}
        assert len(test_runs) == 1
        assert train_runs.isdisjoint(test_runs)
        assert all(train.values()) and all(test.values())
    with pytest.raises(ValueError, match="two|2"):
        list(LeaveOneRunOut().split({"a": {"rating": entry}}))


def test_common_prediction_mask_preserves_feature_missingness():
    from multimodalsrm.validation import common_valid_mask

    observed = np.array([[True, False], [True, True], [True, True]])
    valid = [
        np.array([True, True, False]),
        np.array([[True, True], [False, True], [True, True]]),
    ]
    np.testing.assert_array_equal(
        common_valid_mask(observed, valid),
        [[True, False], [False, True], [False, False]],
    )


def test_cross_validate_clones_and_freezes_fold_training_statistics():
    from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries
    from multimodalsrm.validation import cross_validate

    times = np.arange(12, dtype=float)
    data = {}
    for subject in ("a", "b"):
        data[subject] = {}
        for run, offset in (("film-A", 0), ("film-B", 100), ("film-C", -20)):
            latent = np.sin(times / 3) + offset
            data[subject][run] = {
                "brain": TimeSeries(np.column_stack([latent, 2 * latent]), times),
                "rating": TimeSeries(latent[:, None], times),
            }
    estimator = MultimodalSRM(
        features=1,
        latent_dt=1.0,
        latent_pooling="shared",
        responses={m: Response(Identity(), estimate=False) for m in ("brain", "rating")},
        max_iter=100,
        tol=1e-4,
        random_state=11,
    )
    result = cross_validate(estimator, data, targets={"a": ["brain"]})
    assert not hasattr(estimator, "loadings_")
    assert len(result.models) == 3
    assert len({id(model) for model in result.models}) == 3
    assert len(result.scores) == 9
    for fold, model in zip(result.folds, result.models):
        expected = np.concatenate([data["a"][r]["brain"].values for r in fold["train_runs"]]).mean(
            axis=0
        )
        np.testing.assert_allclose(model.preprocessing_["a"]["brain"]["mean"], expected)
        rows = [row for row in result.scores if row["fold"] == fold["fold"]]
        assert len({row["n_observations"] for row in rows}) == 1
        assert rows[0]["n_observations"] > 0
        assert {row["source"] for row in rows} == {"within", "across", "both"}
