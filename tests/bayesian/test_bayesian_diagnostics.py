"""Dependency upgrades must preserve the declared posterior diagnostic definitions."""

import numpy as np
from numpy.testing import assert_allclose


def test_diagnostics_preserve_legacy_tail_probabilities_and_coordinate_order():
    from multimodalsrm.bayesian.diagnostics import diagnostic_summary

    values = np.random.default_rng(5).normal(size=(4, 100, 2))
    # Recorded with ArviZ 0.22.0: bulk rank ESS, 5%/95% tail ESS,
    # rank-normalized split R-hat, and mean/SD Monte Carlo errors.
    expected = [
        [
            0.049411675762730,
            0.035402490766278,
            377.9446409589444,
            445.8086788784357,
            1.002403987632182,
        ],
        [
            0.045305837074329,
            0.031958387934991,
            498.2378640962900,
            357.4481726864719,
            1.010507062069358,
        ],
    ]
    result = diagnostic_summary(values)
    assert list(result.index) == ["parameter[0]", "parameter[1]"]
    columns = ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
    assert_allclose(result[columns], expected, rtol=1e-12, atol=1e-12)
    scalar = diagnostic_summary(values[..., 1], name="scalar")
    assert list(scalar.index) == ["scalar"]
    assert_allclose(scalar[columns], [expected[1]], rtol=1e-12, atol=1e-12)


def test_diagnostics_detect_chains_trapped_at_different_locations():
    from multimodalsrm.bayesian.diagnostics import diagnostic_summary

    values = np.random.default_rng(31).normal(size=(4, 400))
    values += np.arange(4)[:, None] * 5
    result = diagnostic_summary(values).iloc[0]
    assert result.r_hat > 1.1
    assert result.ess_bulk < 100


def _ar1(rng, chains, draws, count, phi):
    x = np.zeros((chains, draws, count))
    x[:, 0] = rng.normal(size=(chains, count))
    for t in range(1, draws):
        x[:, t] = phi * x[:, t - 1] + rng.normal(size=(chains, count))
    return x


def _arviz_reference(values):
    from arviz_stats.base import array_stats

    axes = dict(chain_axis=0, draw_axis=1)
    return dict(
        ess_bulk=array_stats.ess(values, method="bulk", **axes),
        ess_tail=array_stats.ess(values, method="tail", prob=(0.05, 0.95), **axes),
        r_hat=array_stats.rhat(values, **axes),
        mcse_mean=array_stats.mcse(values, method="mean", **axes),
        mcse_sd=array_stats.mcse(values, method="sd", **axes),
    )


def test_batched_diagnostics_match_arviz_per_quantity_on_edge_cases():
    import warnings

    from multimodalsrm.bayesian.rank_diagnostics import FIELDS, rank_diagnostics

    rng = np.random.default_rng(7)
    mixed = rng.normal(size=(4, 100, 5))
    mixed[..., 1] = 2.5  # constant quantity
    mixed[0, 3, 2] = np.nan  # one missing draw
    mixed[..., 3] = np.arange(4)[:, None] * 1.0  # each chain constant at its own value
    cases = [
        rng.normal(size=(4, 100, 6)),
        rng.normal(size=(3, 101, 5)),  # odd draws
        rng.normal(size=(2, 4, 5)),  # minimum draws
        rng.normal(size=(2, 5, 5)),
        rng.normal(size=(1, 80, 4)),  # single chain: R-hat undefined
        rng.normal(size=(8, 50, 4)),
        _ar1(rng, 4, 500, 5, 0.95),  # long positive autocorrelation sequence
        _ar1(rng, 2, 3000, 3, 0.995),
        _ar1(rng, 4, 200, 5, -0.9),  # antithetic: negative first pair sum
        rng.integers(0, 3, size=(4, 120, 5)).astype(float),  # rank ties
        (rng.random(size=(4, 120, 4)) < 0.1).astype(float),
        rng.normal(size=(4, 400, 3)) + np.arange(4)[:, None, None] * 5,  # trapped chains
        rng.standard_t(2, size=(4, 300, 4)),
        mixed,
    ]
    for values in cases:
        ours = rank_diagnostics(values, chunk=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = _arviz_reference(values)
        for field in FIELDS:
            mine, theirs = ours[field], np.asarray(reference[field], dtype=float)
            assert np.array_equal(np.isnan(mine), np.isnan(theirs)), (values.shape, field)
            finite = np.isfinite(theirs)
            # A chain constant at its own value has zero within-chain variance and an
            # arbitrarily large R-hat in both implementations; compare only its order.
            huge = finite & (theirs > 1e6)
            assert np.all(mine[huge] > 1e6)
            assert_allclose(mine[finite & ~huge], theirs[finite & ~huge], rtol=1e-10)


def test_batched_diagnostics_reject_bad_shapes_and_short_chains():
    import pytest

    from multimodalsrm.bayesian.diagnostics import diagnostic_summary
    from multimodalsrm.bayesian.rank_diagnostics import FIELDS, rank_diagnostics

    with pytest.raises(ValueError, match="chain, draw and quantity"):
        rank_diagnostics(np.zeros((4, 10)))
    short = rank_diagnostics(np.random.default_rng(1).normal(size=(4, 3, 2)))
    assert all(np.isnan(short[field]).all() for field in FIELDS)
    values = np.random.default_rng(2).normal(size=(4, 60, 3))
    columns = list(FIELDS)
    assert_allclose(
        diagnostic_summary(values[..., 2], name="one")[columns].to_numpy()[0],
        diagnostic_summary(values)[columns].to_numpy()[2],
    )


def test_average_ranks_match_scipy_with_ties_and_constants():
    from scipy.stats import rankdata

    from multimodalsrm.bayesian.rank_diagnostics import _average_ranks

    rng = np.random.default_rng(3)
    for flat in (
        rng.normal(size=(5, 40)),
        rng.integers(0, 3, size=(5, 40)).astype(float),
        np.ones((3, 10)),
        np.concatenate([np.zeros((2, 5)), np.ones((2, 6))], axis=1),
    ):
        assert np.array_equal(_average_ranks(flat), rankdata(flat, axis=1))


def test_bfmi_matches_arviz_per_chain():
    import pytest
    from arviz_stats.base import array_stats

    from multimodalsrm.bayesian.rank_diagnostics import bfmi

    energy = np.cumsum(np.random.default_rng(4).normal(size=(3, 300)), axis=1)
    assert_allclose(bfmi(energy), array_stats.bfmi(energy, chain_axis=0, draw_axis=1), rtol=1e-12)
    assert bfmi(energy).shape == (3,)
    with pytest.raises(ValueError, match="chain and draw"):
        bfmi(energy[0])
