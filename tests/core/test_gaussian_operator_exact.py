"""Independent cell integrals expose quadrature noise in shape derivatives."""

import numpy as np
import pytest
from scipy.integrate import quad

from multimodalsrm import Gaussian
from multimodalsrm.operators import observation_operator


@pytest.mark.parametrize("width,lag", [(0.31, 0.2), (0.9, 3.0), (2.5, 7.0)])
def test_gaussian_piecewise_linear_operator_matches_exact_cell_integrals(width, lag):
    grid = np.arange(-30, 30.1, 0.5)
    query = np.array([0.123, 1.25, 3.0])
    kernel = Gaussian(width, lag)
    op, valid = observation_operator(grid, query, kernel)
    assert valid.all()
    expected = np.zeros(op.shape)
    for row, t in enumerate(query):
        for j, (a, b) in enumerate(zip(grid[:-1], grid[1:])):
            lo, hi = max(t - b, kernel.support[0]), min(t - a, kernel.support[1])
            if hi <= lo:
                continue
            expected[row, j] += quad(
                lambda u: float(kernel.evaluate(u)) * (b - t + u) / (b - a),
                lo,
                hi,
                epsabs=1e-14,
                epsrel=1e-13,
            )[0]
            expected[row, j + 1] += quad(
                lambda u: float(kernel.evaluate(u)) * (t - a - u) / (b - a),
                lo,
                hi,
                epsabs=1e-14,
                epsrel=1e-13,
            )[0]
    np.testing.assert_allclose(op.toarray(), expected, atol=2e-13, rtol=2e-13)


def test_exact_gaussian_preserves_empty_and_unsupported_queries():
    grid = np.arange(50.0)
    kernel = Gaussian(1, 4)
    op, valid = observation_operator(grid, [-1, 10, 49], kernel, (-15, 22))
    assert not valid.any()
    assert op.nnz == 0
    empty, valid = observation_operator(grid, [], kernel)
    assert empty.shape == (0, 50)
    assert valid.size == 0
