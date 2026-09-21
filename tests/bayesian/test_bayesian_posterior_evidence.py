"""Sampling evidence must preserve axes and reject misleading comparisons."""

import numpy as np
from numpy.testing import assert_allclose

from .test_bayesian_problem import api, problem_fixture


def test_likelihood_batches_preserve_chain_draw_order_and_partial_tail():
    api()
    from multimodalsrm.bayesian import fitting

    assert hasattr(fitting, "log_likelihood_draws"), "bounded likelihood evaluation missing"
    p, _, _ = problem_fixture(True, two_runs=True)
    xs = np.broadcast_to(p.initial, (3, 7, len(p.names))).copy()
    idx = p.indices[("loading", "b", "signal", 0)]
    xs[..., idx] = np.linspace(-1.2, 1.2, 21).reshape(3, 7)
    expected = np.array([[-float(p.nll(x)) for x in chain] for chain in xs])
    actual = fitting.log_likelihood_draws(p, xs)
    assert actual.shape == (3, 7)
    assert_allclose(actual, expected, atol=1e-10)
