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
