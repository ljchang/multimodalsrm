"""Chunked quadrature operators equal the original row-by-row integration."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from multimodalsrm import DoubleGamma, Gamma, Identity, SampledKernel
from multimodalsrm.operators import _operator_inputs, observation_operator


def rowwise_reference(grid, times, kernel, support=None):
    """The original per-row three-point Gauss-Legendre integration."""
    grid, times, valid = _operator_inputs(grid, times, kernel, support)
    lo, hi = kernel.support
    rows, cols, values = [], [], []
    nodes, weights = np.polynomial.legendre.leggauss(3)
    if isinstance(kernel, Gamma):
        resolution = kernel.scale / 2
    elif isinstance(kernel, DoubleGamma):
        resolution = min(kernel.peak_scale, kernel.undershoot_scale) / 2
    else:
        resolution = None
    shape_knots = (
        np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / resolution)) + 1))
        if resolution
        else np.array([])
    )
    for row in np.flatnonzero(valid):
        t = times[row]
        if isinstance(kernel, Identity):
            positions = np.array([np.clip(t, grid[0], grid[-1])])
            mass = np.array([1.0])
        else:
            knots = t - grid
            interior = knots[(knots > lo) & (knots < hi)]
            splits = [np.array([lo, hi]), interior, shape_knots]
            if isinstance(kernel, SampledKernel):
                splits.append(kernel.lags)
            breaks = np.unique(np.concatenate(splits))
            half = np.diff(breaks) / 2
            mid = (breaks[:-1] + breaks[1:]) / 2
            lags = (mid[:, None] + half[:, None] * nodes).ravel()
            mass = (half[:, None] * weights).ravel() * kernel.evaluate(lags)
            positions = np.clip(t - lags, grid[0], grid[-1])
        left = np.searchsorted(grid, positions, side="right") - 1
        left = np.clip(left, 0, len(grid) - 2)
        fraction = (positions - grid[left]) / (grid[left + 1] - grid[left])
        rows.extend(np.repeat(row, len(left) * 2))
        cols.extend(np.column_stack([left, left + 1]).ravel())
        values.extend(np.column_stack([mass * (1 - fraction), mass * fraction]).ravel())
    op = csr_matrix((values, (rows, cols)), shape=(len(times), len(grid)))
    op.eliminate_zeros()
    return op, valid


KERNELS = {
    "identity": Identity(),
    "gamma": Gamma(6.0, 1.0, 0.5),
    "exponential_negative_lag": Gamma(1.0, 0.3, -1.2),
    "narrow_gamma": Gamma(2.5, 0.05),
    "double_gamma": DoubleGamma(lag=-0.7),
    "short_double_gamma": DoubleGamma(5.0, 0.4, 12.0, 0.5, 0.3, 1.0),
    "sampled": SampledKernel(
        np.array([-1.0, 0.0, 0.4, 2.0, 2.5]), np.array([0.0, 1.0, -0.5, 0.3, 0.0])
    ),
    "long_sampled": SampledKernel(np.linspace(0, 20, 41), np.random.default_rng(3).normal(size=41)),
}


def clocks(grid):
    return {
        "regular": np.arange(0, 80, 1.5),
        "irregular": np.sort(np.random.default_rng(1).uniform(-10, 90, 61)),
        "on_grid": grid[::3].copy(),
        "single": np.array([40.0]),
        "unsupported": np.array([-500.0, 900.0]),
    }


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("dt", [0.25, 1.0, 3.7])
def test_chunked_operator_equals_rowwise_integration(name, dt):
    kernel = KERNELS[name]
    grid = 5.0 + dt * np.arange(int(70 / dt) + 1)
    for times in clocks(grid).values():
        for support in (None, (kernel.support[0] - 1.0, kernel.support[1] + 2.0)):
            expected, expected_valid = rowwise_reference(grid, times, kernel, support)
            actual, valid = observation_operator(grid, times, kernel, support)
            np.testing.assert_array_equal(valid, expected_valid)
            np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-13, atol=1e-15)
            assert actual.nnz == expected.nnz


def test_chunk_boundaries_do_not_change_rows():
    kernel = SampledKernel(np.linspace(0, 30, 301), np.cos(np.linspace(0, 9, 301)))
    grid = 0.05 * np.arange(4001)
    times = np.sort(np.random.default_rng(5).uniform(0, 200, 700))
    expected, _ = rowwise_reference(grid, times, kernel)
    actual, _ = observation_operator(grid, times, kernel)
    # 600 latent knots per row forces many small chunks.
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-13, atol=1e-15)
