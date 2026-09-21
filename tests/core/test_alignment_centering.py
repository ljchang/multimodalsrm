"""The companion score removes each feature's mean within each window."""

import numpy as np
import pytest

from multimodalsrm import time_segment_matching

from .test_multimodal_alignment import series


def test_feature_centered_matching_agrees_with_direct_scores_and_rotations():
    rng = np.random.default_rng(915)
    x = rng.normal(size=(4, 22, 3))
    x += rng.normal(size=(1, 22, 3))
    mask = np.ones(x.shape, bool)
    mask[1, 8] = False
    width = 4
    expected = time_segment_matching(series(x, mask), window_size=width, centering="feature")
    assert expected["metric"] == "time_segment_matching_feature_centered"
    starts = expected["window_starts"]
    for s in range(len(x)):
        other = np.delete(x, s, axis=0).mean(axis=0)
        for record in expected["subjects"][str(s)]["queries"]:
            i = record["start_index"]
            q = x[s, i : i + width] - x[s, i : i + width].mean(axis=0)
            candidates = [j for j in starts if abs(j - i) >= width or j == i]
            scores = []
            for j in candidates:
                t = other[j : j + width] - other[j : j + width].mean(axis=0)
                scores.append(np.sum(q * t) / (np.linalg.norm(q) * np.linalg.norm(t)))
            winners = np.isclose(scores, max(scores), rtol=0, atol=1e-12)
            assert record["credit"] == winners[candidates.index(i)] / winners.sum()
    for _ in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = x @ rotation + rng.normal(size=(4, 1, 3))
        actual = time_segment_matching(
            series(rotated, mask), window_size=width, centering="feature"
        )
        assert actual == expected
    assert time_segment_matching(series(x), window_size=width) == time_segment_matching(
        series(x), window_size=width, centering="global"
    )


def test_feature_constants_remain_undefined_and_ties_fractional():
    constant = np.broadcast_to([1.0, 2.0], (3, 12, 2)).copy()
    report = time_segment_matching(series(constant), window_size=3, centering="feature")
    assert all(s["accuracy"] is None for s in report["subjects"].values())
    assert report["subjects"]["0"]["queries"][0]["reason"] == "constant_signal"
    ramp = constant + np.arange(12)[None, :, None]
    report = time_segment_matching(series(ramp), window_size=3, centering="feature")
    for s in report["subjects"].values():
        assert all(q["credit"] == q["chance"] for q in s["queries"])
    with pytest.raises(ValueError, match="centering"):
        time_segment_matching(series(ramp), window_size=3, centering="zscore")
