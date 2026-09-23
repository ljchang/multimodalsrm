"""Batched grouped predictions agree with dense Gaussian conditioning."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm.bayesian.grouped_prediction import project as grouped_project
from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.problem import BayesianProblem

from .test_bayesian_multifactor import fixture


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("noise", [{}, {"brain": 3.0}, {"brain": 3.0, "aux": 1.2}])
def test_multi_feature_draw_and_query_batches_match_dense(features, noise):
    original, x, _ = fixture(features=features)
    kwargs = dict(anchor=original.anchor, noise_timescales=noise)
    dense = BayesianProblem(original.adapter, original.priors, **kwargs)
    grouped = BayesianProblem(original.adapter, original.priors, linear_algebra="grouped", **kwargs)
    other = x.copy()
    for i, name in enumerate(grouped.names):
        if name[:4] == ("loading", "b", "brain", 1):
            other[i] = 0.0
        elif name[0] == "noise":
            other[i] *= 1.7
        elif name[0] == "offset":
            other[i] += 0.2
        elif name == ("filter", "aux", "width"):
            other[i] = 0.7
    draws = np.stack([x, other])
    # Both the query-batch and native-time-chunk boundaries, plus ordering and
    # duplicate times. Distinct subject keys have different feature masks.
    query = np.r_[7.1, np.linspace(2.5, 10, 130), 4.0, 7.1]
    for modality in ("brain", "aux"):
        keys = [("a", modality, 0), ("b", modality, 1)]
        for noisy in (False, True):
            actual = grouped_project(grouped, draws, "train", query, keys=keys, include_noise=noisy)
            for i, key in enumerate(keys):
                expected = project(dense, draws, "train", query, key=key, include_noise=noisy)
                assert_allclose(actual[0][..., i], expected[0], atol=2e-9, rtol=1e-9)
                assert_allclose(actual[1][..., i], expected[1], atol=2e-9, rtol=1e-9)


def test_draw_aligned_rotations_retain_cross_factor_covariance():
    dense, x, _ = fixture(features=3)
    grouped = BayesianProblem(
        dense.adapter, dense.priors, anchor=dense.anchor, linear_algebra="grouped"
    )
    draws = np.stack([x, x.copy()])
    draws[1, 0] += 0.4
    rng = np.random.default_rng(820)
    directions = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0].T for _ in draws])
    query = np.array([3.2, 5.5, 7.0])
    actual = grouped_project(
        grouped,
        draws,
        "train",
        query,
        keys=[None] * 3,
        include_noise=False,
        directions=directions,
    )
    for i in range(3):
        expected = project(
            dense,
            draws,
            "train",
            query,
            key=None,
            include_noise=False,
            latent_loading=directions[:, i],
        )
        assert_allclose(actual[0][..., i], expected[0], atol=1e-9)
        assert_allclose(actual[1][..., i], expected[1], atol=1e-9)


@pytest.mark.parametrize("noise", [{}, {"ref": 2.0, "signal": 3.0}])
def test_conditioned_public_prediction_uses_new_observations_and_keeps_metadata(noise):
    from .test_bayesian_model import make_model

    model, data = make_model(linear_algebra="grouped", noise_timescales=noise)
    model.fit(data)
    donors = {s: {"new": copy.deepcopy(runs["train"])} for s, runs in data.items()}
    first = model.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    query = np.array([6.0, 9.0, 13.0])
    result = first.predict(times={"new": query}, include_noise=True)["b"]["new"]["signal"]
    dense = copy.copy(first)
    dense.problem_ = BayesianProblem(
        first.problem_.adapter,
        first.problem_.priors,
        systems=first.problem_.systems,
        anchor=first.problem_.anchor,
        noise_timescales=noise,
    )
    expected = dense.predict(times={"new": query}, include_noise=True)["b"]["new"]["signal"]
    assert_allclose(result.component_means, expected.component_means, atol=1e-9)
    assert_allclose(result.component_variances, expected.component_variances, atol=1e-9)
    assert result.metadata == expected.metadata
    ts = donors["a"]["new"]["ref"]
    from multimodalsrm import TimeSeries

    donors["a"]["new"]["ref"] = TimeSeries(ts.values + 4, ts.times, ts.mask)
    second = model.condition(donors, targets={"b": ["signal"]}, mode="frozen")
    changed = second.predict(times={"new": query}, include_noise=True)["b"]["new"]["signal"]
    assert np.max(abs(changed.component_means - result.component_means)) > 0.1
    repeated = first.predict(times={"new": query}, include_noise=True)["b"]["new"]["signal"]
    assert_allclose(repeated.component_means, result.component_means, atol=0, rtol=0)
