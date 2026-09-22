"""Batched rank-normalized split R-hat, ESS and MCSE over many quantities.

The definitions are those of Vehtari et al. (2021) as implemented by ArviZ:
split chains, rank normalization with Blom's offset, Geyer's initial
positive and monotone sequences for the autocorrelation sum, tail ESS as
the smaller of the 5% and 95% quantile-indicator ESS, and the mean and
standard-deviation Monte Carlo errors. Results agree with ArviZ's
per-quantity functions to floating-point rounding. The difference is that
every quantity is a leading array axis processed at once, with draws
contiguous in memory, so a posterior with tens of thousands of parameters
is diagnosed in seconds rather than through one Python call per quantity.

Internally arrays are (quantity, chain, draw); the public entry point takes
the package's (chain, draw, quantity) layout.
"""

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import ndtri

RANK_OFFSET = 3 / 8  # Blom (1958)
TAIL_PROBABILITIES = (0.05, 0.95)
MIN_DRAWS = 4
FIELDS = ("ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd")


def _split_chains(values):
    half = values.shape[2] // 2
    return np.concatenate([values[:, :, :half], values[:, :, -half:]], axis=1)


def _average_ranks(flat):
    """Average ranks along the last axis; tie order is irrelevant, so any sort will do."""
    count, size = flat.shape
    order = np.argsort(flat, axis=1)
    ordered = np.take_along_axis(flat, order, axis=1)
    positions = np.broadcast_to(np.arange(1, size + 1, dtype=float), flat.shape)
    ties = ordered[:, 1:] == ordered[:, :-1]
    if ties.any():
        start = np.ones(flat.shape, dtype=bool)
        start[:, 1:] = ~ties
        end = np.ones(flat.shape, dtype=bool)
        end[:, :-1] = ~ties
        first = np.maximum.accumulate(np.where(start, positions, 0.0), axis=1)
        last = np.minimum.accumulate(np.where(end, positions, np.inf)[:, ::-1], axis=1)[:, ::-1]
        positions = (first + last) / 2.0
    ranks = np.empty(flat.shape)
    np.put_along_axis(ranks, order, positions, axis=1)
    return ranks


def _z_scale(values):
    """Rank every quantity over all its (chain, draw) entries, then normal scores."""
    count, chains, draws = values.shape
    size = chains * draws
    ranks = _average_ranks(values.reshape(count, size))
    z = ndtri((ranks - RANK_OFFSET) / (size - 2 * RANK_OFFSET + 1))
    return z.reshape(count, chains, draws)


def _autocov(values):
    n = values.shape[2]
    m = next_fast_len(2 * n)
    centered = values - values.mean(axis=2, keepdims=True)
    spectrum = np.fft.rfft(centered, n=m, axis=2)
    spectrum *= np.conjugate(spectrum)
    return np.fft.irfft(spectrum, n=m, axis=2)[:, :, :n] / n


def _ess(values):
    """Geyer's ESS of split (count, chains, draws) values, one per leading entry.

    The sequential Geyer loops become array operations: the initial positive
    sequence keeps the leading run of positive autocorrelation pair sums, and
    the monotone sequence is a running minimum over those pair sums, which is
    all the final sum depends on.
    """
    count, chains, draws = values.shape
    size = chains * draws
    constant = (values.max(axis=(1, 2)) - values.min(axis=(1, 2))) < np.finfo(float).resolution
    with np.errstate(divide="ignore", invalid="ignore"):
        acov = _autocov(values)
        mean_var = acov[:, :, 0].mean(axis=1) * draws / (draws - 1.0)
        var_plus = mean_var * (draws - 1.0) / draws
        if chains > 1:
            var_plus = var_plus + values.mean(axis=2).var(axis=1, ddof=1)
        rho = 1.0 - (mean_var[:, None] - acov.mean(axis=1)) / var_plus[:, None]
    rho[:, 0] = 1.0
    if draws % 2:
        rho = np.concatenate([rho, np.zeros((count, 1))], axis=1)
    even, odd = rho[:, 0::2], rho[:, 1::2]
    pair_sum = even + odd
    pairs = np.arange(pair_sum.shape[1])[None, :]
    # Initial positive sequence: pair j is examined while every earlier pair sum
    # is positive and its second lag stays inside the window; m examined pairs.
    last = (draws - 3) // 2
    if last < 1:
        examined = np.zeros(count, dtype=int)
    else:
        positive = pair_sum[:, :last] > 0.0
        examined = np.where(positive.all(axis=1), last, positive.argmin(axis=1))
    stored = (pairs >= 1) & (pairs <= examined[:, None]) & (pair_sum >= 0.0)
    even_hat = np.where(stored, even, 0.0)
    odd_hat = np.where(stored, odd, 0.0)
    even_hat[:, 0], odd_hat[:, 0] = 1.0, rho[:, 1]
    # The last examined even lag is kept when positive even if its pair was not.
    rows = np.arange(count)
    tail_even = even[rows, examined]
    improve = (examined >= 1) & (stored[rows, examined] | (tail_even > 0.0))
    even_hat[rows, examined] = np.where(improve, tail_even, even_hat[rows, examined])
    # Initial monotone sequence over the pairs before the last examined one.
    active = pairs < examined[:, None]
    running = np.minimum.accumulate(np.where(active, even_hat + odd_hat, np.inf), axis=1)
    tau = -1.0 + 2.0 * np.where(active, running, 0.0).sum(axis=1) + even_hat[rows, examined]
    tau = np.maximum(tau, 1.0 / np.log10(size))
    ess = size / tau
    bad = ~(np.isfinite(even_hat).all(axis=1) & np.isfinite(odd_hat).all(axis=1))
    return np.where(constant, float(size), np.where(bad, np.nan, ess))


def _rhat(values):
    n = values.shape[2]
    between = n * values.mean(axis=2).var(axis=1, ddof=1)
    within = values.var(axis=2, ddof=1).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((between / within + n - 1.0) / n)


def _quantile_ess(values, prob):
    threshold = np.quantile(values.reshape(len(values), -1), prob, axis=1)
    indicator = (values <= threshold[:, None, None]).astype(float)
    return _ess(_split_chains(indicator))


def _diagnose(values):
    count, chains, _ = values.shape
    flat = values.reshape(count, -1)
    split = _split_chains(values)
    z = _z_scale(split)
    out = dict(ess_bulk=_ess(z))
    out["ess_tail"] = np.minimum(*(_quantile_ess(values, p) for p in TAIL_PROBABILITIES))
    if chains >= 2:
        median = np.median(split.reshape(count, -1), axis=1)
        folded = np.abs(split - median[:, None, None])
        out["r_hat"] = np.maximum(_rhat(z), _rhat(_z_scale(folded)))
    else:
        out["r_hat"] = np.full(count, np.nan)
    out["mcse_mean"] = flat.std(axis=1, ddof=1) / np.sqrt(_ess(split))
    centered = values - flat.mean(axis=1)[:, None, None]
    squared = centered**2
    variance = squared.reshape(count, -1).mean(axis=1)
    fourth = (squared**2).reshape(count, -1).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_error = (fourth - variance**2) / _ess(_split_chains(squared))
        out["mcse_sd"] = np.sqrt(variance_error / variance / 4.0)
    return out


def rank_diagnostics(values, *, chunk=1024):
    """Diagnostics of (chains, draws, count) values, one array of ``count`` per field.

    Quantities with a non-finite draw, and every quantity when there are
    fewer than four draws, are NaN; R-hat is NaN with a single chain. Work
    proceeds in blocks of ``chunk`` quantities to bound memory.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("values must have chain, draw and quantity axes")
    count = values.shape[-1]
    out = {field: np.full(count, np.nan) for field in FIELDS}
    finite = np.flatnonzero(np.isfinite(values).all(axis=(0, 1)))
    if values.shape[1] < MIN_DRAWS or not len(finite):
        return out
    for start in range(0, len(finite), chunk):
        take = finite[start : start + chunk]
        block = _diagnose(np.ascontiguousarray(values[..., take].transpose(2, 0, 1)))
        for field in FIELDS:
            out[field][take] = block[field]
    return out


def bfmi(energy):
    """Bayesian fraction of missing information per chain of (chains, draws) energies.

    The squared mean energy change over the sample variance of the energy,
    the estimator of Betancourt (2016) as reported by ArviZ.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 2:
        raise ValueError("energy must have chain and draw axes")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.square(np.diff(energy, axis=1)).mean(axis=1) / energy.var(axis=1, ddof=1)
