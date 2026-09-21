"""Sparse native-time convolution of piecewise-linear latent trajectories.

Gaussian filters use exact zeroth and first moments within each latent cell.
Other filters use three-point Gauss-Legendre integration within each cell
(and each custom-kernel segment) carries physical seconds explicitly. There is
no padding: unsupported rows are zero and their validity is False.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from .data import validate_times
from .kernels import (
    BachSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    Kernel,
    SampledKernel,
)


def _gaussian_operator(grid, times, kernel, valid, derivatives=False):
    """Integrate the two linear interpolation weights analytically.

    Removing parameter-dependent quadrature partitions avoids numerical noise
    in finite-difference shape gradients. Chunk queries to bound temporary arrays.
    """
    rows, cols, values = [], [], []
    gradient_values = {p: [] for p in ("width", "lag")}
    indices = np.flatnonzero(valid)
    width, lag = kernel.width, kernel.lag
    lo, hi = kernel.support
    energy = kernel._energy
    for start in range(0, len(indices), 128):
        selected = indices[start : start + 128]
        t = times[selected, None]
        lower = np.maximum(t - grid[None, 1:], lo)
        upper = np.minimum(t - grid[None, :-1], hi)
        r, c = np.nonzero(upper > lower)
        low = (lower[r, c] - lag) / width
        high = (upper[r, c] - lag) / width
        # Survival probabilities avoid subtraction near one in the upper tail.
        probability = np.where(low >= 0, ndtr(-low) - ndtr(-high), ndtr(high) - ndtr(low))
        mass = width * np.sqrt(2 * np.pi) * probability / energy
        centered_moment = width**2 * (np.exp(-(low**2) / 2) - np.exp(-(high**2) / 2)) / energy
        dt = grid[c + 1] - grid[c]
        left = ((grid[c + 1] - times[selected[r]] + lag) * mass + centered_moment) / dt
        right = mass - left
        rows.append(np.repeat(selected[r], 2))
        cols.append(np.column_stack([c, c + 1]).ravel())
        # Exact weights are nonnegative; remove tiny tail cancellation roundoff.
        values.append(np.maximum(np.column_stack([left, right]).ravel(), 0))
        if derivatives:
            exp_low, exp_high = np.exp(-(low**2) / 2), np.exp(-(high**2) / 2)
            # Standardized integration bounds at the moving six-sigma support
            # are constant. Other bounds move in standardized coordinates.
            lower_free = t[r, 0] - grid[c + 1] > lo
            upper_free = t[r, 0] - grid[c] < hi
            for parameter in gradient_values:
                is_width = parameter == "width"
                dlow = np.where(lower_free, -low / width if is_width else -1 / width, 0.0)
                dhigh = np.where(upper_free, -high / width if is_width else -1 / width, 0.0)
                # d log(energy)/d width = 1/(2 width), including truncation.
                dmass = width / energy * (exp_high * dhigh - exp_low * dlow)
                dmoment = width**2 / energy * (-low * exp_low * dlow + high * exp_high * dhigh)
                if is_width:
                    dmass += mass / (2 * width)
                    dmoment += 1.5 * centered_moment / width
                dleft = (
                    (grid[c + 1] - times[selected[r]] + lag) * dmass
                    + dmoment
                    + (0 if is_width else mass)
                ) / dt
                dright = dmass - dleft
                gradient_values[parameter].append(
                    np.column_stack(
                        [np.where(left > 0, dleft, 0), np.where(right > 0, dright, 0)]
                    ).ravel()
                )
    if not rows:
        empty = csr_matrix((len(times), len(grid)))
        if derivatives:
            return empty, {p: empty.copy() for p in gradient_values}, valid
        return empty, valid
    op = csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(times), len(grid)),
    )
    op.eliminate_zeros()
    if derivatives:
        gradients = {
            p: csr_matrix(
                (np.concatenate(v), (np.concatenate(rows), np.concatenate(cols))),
                shape=op.shape,
            )
            for p, v in gradient_values.items()
        }
        return op, gradients, valid
    return op, valid


def _operator_inputs(grid, times, kernel, support):
    grid = validate_times(grid)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("query times must be a finite vector")
    if len(grid) < 2 or not np.allclose(np.diff(grid), np.diff(grid)[0], rtol=1e-8, atol=1e-12):
        raise ValueError("latent grid must have at least two regularly spaced times")
    if not isinstance(kernel, Kernel):
        raise ValueError("unsupported kernel")
    envelope = np.asarray(kernel.support if support is None else support, dtype=float)
    if envelope.shape != (2,) or not np.isfinite(envelope).all() or envelope[0] > envelope[1]:
        raise ValueError("support must be a finite increasing pair")
    lo, hi = kernel.support
    if envelope[0] > lo + 1e-12 or envelope[1] < hi - 1e-12:
        raise ValueError("fixed support must contain kernel support")
    tol = 1e-10 * max(1.0, abs(grid[0]), abs(grid[-1]))
    valid = (times - envelope[1] >= grid[0] - tol) & (times - envelope[0] <= grid[-1] + tol)
    return grid, times, valid


def gaussian_operator_derivatives(grid, times, kernel, support=None):
    """Exact Gaussian cell integrals and derivatives in physical coordinates."""
    if type(kernel) is not Gaussian:
        raise ValueError("analytic response derivatives require Gaussian")
    grid, times, valid = _operator_inputs(grid, times, kernel, support)
    return _gaussian_operator(grid, times, kernel, valid, derivatives=True)


def observation_operator(grid, times, kernel, support=None):
    grid, times, valid = _operator_inputs(grid, times, kernel, support)
    if isinstance(kernel, Gaussian):
        return _gaussian_operator(grid, times, kernel, valid)
    lo, hi = kernel.support
    rows, cols, values = [], [], []
    nodes, weights = np.polynomial.legendre.leggauss(3)
    # Resolve the response itself even when much narrower than a latent cell.
    # This is integration accuracy, not an assertion of recoverable bandwidth.
    if isinstance(kernel, Gamma):
        resolution = kernel.scale / 2
    elif isinstance(kernel, DoubleGamma):
        resolution = min(kernel.peak_scale, kernel.undershoot_scale) / 2
    elif isinstance(kernel, BachSCR):
        resolution = kernel.sigma / 2
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
            # Integration variable is lag; every knot of z(t-lag) is a split.
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
