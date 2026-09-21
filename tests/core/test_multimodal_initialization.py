"""Training-only native-block informed starts and paired restart regression."""

import warnings

import numpy as np
import pytest
from scipy import sparse

from multimodalsrm import MultimodalSRM, TimeSeries
from multimodalsrm.initialization import spectral_latents
from multimodalsrm.objective import ObservationBlock


def fixture_blocks(rank=2):
    rng = np.random.RandomState(12)
    grids = {r: np.arange(24.0) for r in ["run-B", "run-A"]}
    truth = {r: rng.normal(size=(24, rank)) for r in grids}
    maps = {s: rng.normal(size=(4, rank)) for s in ["p2", "p1"]}
    blocks = [
        ObservationBlock(
            s,
            r,
            "brain",
            truth[r] @ maps[s].T,
            np.ones((24, 4), bool),
            t.copy(),
            sparse.eye(24, format="csr"),
            np.full((24, 4), 1 / 96),
            np.ones(24, bool),
        )
        for r, t in grids.items()
        for s in maps
    ]
    return blocks, grids, truth


def test_native_rank_two_multiple_runs_and_maps_recover_common_basis():
    blocks, grids, truth = fixture_blocks()
    z, diag = spectral_latents(blocks, grids, ["p2", "p1"], 2, "shared", None)
    observed = np.concatenate([z["p2"][r] for r in grids])
    target = np.concatenate(list(truth.values()))
    np.testing.assert_allclose(
        observed @ np.linalg.lstsq(observed, target, rcond=None)[0], target, atol=1e-12
    )
    z2, diag2 = spectral_latents(blocks, grids, ["p2", "p1"], 2, "shared", None)
    assert diag == diag2
    for r in grids:
        np.testing.assert_array_equal(z["p1"][r], z["p2"][r])
        np.testing.assert_array_equal(z["p1"][r], z2["p1"][r])
    assert diag["method"] == "spectral"


def test_masks_unequal_intervals_unsupported_cells_do_not_amplify():
    blocks, grids, truth = fixture_blocks()
    block = blocks[0]
    block.mask[::2, 0] = False
    block.values[::2, 0] = np.nan
    block.coefficients[::2, 0] = 0
    block.H = block.H * 1e-30
    other = blocks[1]
    other.times = other.times[::2]
    other.H = other.H[::2]
    other.values = other.values[::2]
    other.mask = other.mask[::2]
    other.coefficients = other.coefficients[::2]
    other.valid = other.valid[::2]
    before = block.values.copy()
    z, diag = spectral_latents(blocks, grids, ["p2", "p1"], 2, "population", None)
    assert all(np.isfinite(v).all() for runs in z.values() for v in runs.values())
    np.testing.assert_array_equal(block.values, before)
    block.values[:] = 1e20  # This entire operator has negligible supported mass.
    again, _ = spectral_latents(blocks, grids, ["p2", "p1"], 2, "population", None)
    for r in grids:
        np.testing.assert_array_equal(z["p2"][r], again["p2"][r])
    for r in grids:
        np.testing.assert_array_equal(z["p2"][r], z["p1"][r])


def test_numerical_rank_fallback_is_explicit():
    blocks, grids, _ = fixture_blocks(rank=1)
    z, diag = spectral_latents(blocks, grids, ["p2", "p1"], 2, "shared", None)
    assert z is None
    assert diag["method"] == "random"
    assert "rank" in diag["fallback_reason"]


def test_invalid_policy_and_paired_random_restarts():
    t = np.arange(16.0)
    data = {"a": {"train": {"x": TimeSeries(np.column_stack([np.sin(t), np.cos(t)]), t)}}}
    kwargs = dict(features=2, latent_dt=1, n_init=3, max_iter=2, random_state=42)
    with pytest.raises(ValueError, match="init"):
        MultimodalSRM(**kwargs, init="bad").fit(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random = MultimodalSRM(**kwargs).fit(data)
        hybrid = MultimodalSRM(**kwargs, init="hybrid").fit(data)
    assert random.configuration_["init"] == "random"
    assert [r["seed"] for r in random.restart_diagnostics_] == [
        r["seed"] for r in hybrid.restart_diagnostics_
    ]
    for a, b in zip(random.restart_diagnostics_[1:], hybrid.restart_diagnostics_[1:]):
        assert a["objective"] == b["objective"]
        assert b["initialization"]["method"] == "random"
    assert hybrid.restart_diagnostics_[0]["initialization"]["method"] == "spectral"


def test_components_use_independent_native_columns_and_no_unrelated_donor():
    blocks, grids, _ = fixture_blocks()
    affinity = {"p2": {}, "p1": {}}
    z, _ = spectral_latents(blocks, grids, ["p2", "p1"], 2, "components", affinity)
    for block in blocks:
        if block.subject == "p1":
            block.values[:] = np.random.RandomState(77).normal(size=block.values.shape)
    after, _ = spectral_latents(blocks, grids, ["p2", "p1"], 2, "components", affinity)
    for r in grids:
        np.testing.assert_array_equal(z["p2"][r], after["p2"][r])
    assert not np.allclose(z["p1"]["run-A"], after["p1"]["run-A"])


def test_rank_fallback_preserves_random_fit_exactly_and_spectral_repeats():
    t = np.arange(16.0)
    x = np.sin(t)[:, None]
    data = {"a": {"train": {"x": TimeSeries(x, t)}}}
    kwargs = dict(features=2, latent_dt=1, n_init=2, max_iter=3, random_state=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random = MultimodalSRM(**kwargs).fit(data)
        fallback = MultimodalSRM(**kwargs, init="spectral").fit(data)
        spectral = MultimodalSRM(**(kwargs | dict(features=1)), init="spectral").fit(data)
    for a, b in zip(random.restart_diagnostics_, fallback.restart_diagnostics_):
        assert a["seed"] == b["seed"]
        assert a["objective"] == b["objective"]
        assert "rank" in b["initialization"]["fallback_reason"]
    assert (
        spectral.restart_diagnostics_[0]["objective"]
        == spectral.restart_diagnostics_[1]["objective"]
    )
