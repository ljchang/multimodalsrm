"""Residual diagnostics shared by both estimators.

A fitted model splits every observed feature into a shared reconstruction and
a remainder it treats as private noise. The summaries here ask whether that
remainder behaves like private noise: how much of it there is and how slowly it
varies, whether it is correlated across participants (shared signal the factors
missed), and whether it links modalities within a participant (a
participant-specific signal). They read a fit; they never change one.

Residual series are autocorrelated, so sample correlations are not judged by
independent-sample formulas. Every reference level comes from circular shifts,
which keep each series' own autocorrelation and destroy only its alignment
with the series it is compared to.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.utils import check_random_state

# A timescale is fitted only to a clear, decaying autocorrelation.
ACF_FLOOR = 0.05
ACF_LAG_ONE_FLOOR = 0.1
MINIMUM_PAIRS = 10


@dataclass
class ResidualSeries:
    """One participant, run and modality: time by feature, NaN where unobserved."""

    subject: object
    run: object
    modality: object
    times: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray


def _settings(acf_lags, max_lag, min_shift, n_shifts):
    acf_lags = np.asarray(acf_lags, dtype=float)
    if (
        acf_lags.ndim != 1
        or not len(acf_lags)
        or not np.isfinite(acf_lags).all()
        or np.any(acf_lags <= 0)
        or len(np.unique(acf_lags)) != len(acf_lags)
    ):
        raise ValueError("acf_lags must be distinct positive finite seconds")
    for name, value in (("max_lag", max_lag), ("min_shift", min_shift)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite seconds")
    if isinstance(n_shifts, bool) or not isinstance(n_shifts, (int, np.integer)) or n_shifts < 1:
        raise ValueError("n_shifts must be a positive integer")
    return acf_lags


def _interval(times):
    return float(np.median(np.diff(times))) if len(times) > 1 else np.nan


def _centered(values):
    """Remove each feature's mean over its observed samples; keep NaN."""
    with np.errstate(invalid="ignore"):
        count = np.isfinite(values).sum(0)
        mean = np.where(count > 0, np.nansum(values, 0) / np.maximum(count, 1), 0.0)
    return values - mean


def _standardized(values):
    """Unit-norm, zero-mean columns with zeros where unobserved; NaN columns if constant."""
    centered = _centered(values)
    filled = np.where(np.isfinite(centered), centered, 0.0)
    norm = np.sqrt((filled**2).sum(0))
    usable = norm > 0
    return filled / np.where(usable, norm, 1.0), usable


def _lagged_pairs(times, lag, tolerance):
    """Index pairs (i, j) whose timestamps differ by ``lag`` within ``tolerance``."""
    requested = times + lag
    right = np.searchsorted(times, requested)
    a, b = np.clip(right - 1, 0, len(times) - 1), np.clip(right, 0, len(times) - 1)
    closest = np.where(np.abs(times[a] - requested) <= np.abs(times[b] - requested), a, b)
    keep = (np.abs(times[closest] - requested) <= tolerance) & (closest > np.arange(len(times)))
    return np.flatnonzero(keep), closest[keep]


def autocorrelation(times, residual, lags):
    """Per-feature autocorrelation at ``lags`` seconds, from actual timestamp pairs.

    Pairs are timestamps separated by the lag within a quarter of the median
    sampling interval, so gaps and masked samples are never compressed. The
    estimate is the mean lagged product over the feature's variance; fewer
    than ten pairs give NaN.
    """
    lags = np.asarray(lags, dtype=float)
    centered = _centered(residual)
    observed = np.isfinite(centered)
    filled = np.where(observed, centered, 0.0)
    count = observed.sum(0)
    variance = (filled**2).sum(0) / np.maximum(count, 1)
    output = np.full((len(lags), residual.shape[1]), np.nan)
    step = _interval(times)
    if not np.isfinite(step):
        return output
    for k, lag in enumerate(lags):
        i, j = _lagged_pairs(times, lag, 0.25 * step)
        if not len(i):
            continue
        pairs = (observed[i] & observed[j]).sum(0)
        product = (filled[i] * filled[j]).sum(0) / np.maximum(pairs, 1)
        good = (pairs >= MINIMUM_PAIRS) & (variance > 0)
        output[k, good] = product[good] / variance[good]
    return output


def exponential_timescale(lags, acf):
    """Fit ``acf = slow_fraction * exp(-lag / timescale)`` to the leading lags.

    Uses the initial run of lags whose autocorrelation exceeds 0.05, and only
    when the first exceeds 0.1 and at least two lags qualify. Returns
    ``(timescale, slow_fraction, status)`` with None values and a reason when
    the fit is not well defined. ``slow_fraction`` is the fitted value at lag
    zero, capped at one: the share of residual variance in the exponentially
    correlated part, the rest being faster than one sample.
    """
    lags, acf = np.asarray(lags, dtype=float), np.asarray(acf, dtype=float)
    if not len(acf) or not np.isfinite(acf[0]):
        return None, None, "undefined: too few timestamp pairs at the sampling interval"
    if acf[0] < ACF_LAG_ONE_FLOOR:
        return None, None, f"undefined: lag-one autocorrelation below {ACF_LAG_ONE_FLOOR}"
    below = np.flatnonzero(~(acf > ACF_FLOOR))
    count = int(below[0]) if len(below) else len(acf)
    if count < 2:
        return None, None, f"undefined: fewer than two lags above {ACF_FLOOR}"
    slope, intercept = np.polyfit(lags[:count], np.log(acf[:count]), 1)
    if slope >= 0:
        return None, None, "undefined: autocorrelation does not decay over the fitted lags"
    return float(-1.0 / slope), float(min(1.0, np.exp(intercept))), "ok"


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def _feature_mean(rows):
    """Mean over participants per feature, when every participant has the same features."""
    if len({len(row) for row in rows}) != 1:
        return None
    stacked = np.vstack(rows)
    with np.errstate(invalid="ignore"):
        count = np.isfinite(stacked).sum(0)
        return np.where(count > 0, np.nansum(stacked, 0) / np.maximum(count, 1), np.nan)


def _variance_breakdown(series, acf_lags, detail):
    """Shared and residual variance shares and the residual's autocorrelation."""
    output = {}
    for modality in dict.fromkeys(s.modality for s in series):
        local = [s for s in series if s.modality == modality]
        step = float(np.nanmedian([_interval(s.times) for s in local]))
        curve_lags = (
            step * np.arange(1, min(60, int(np.ceil(acf_lags.max() / step))) + 1)
            if np.isfinite(step)
            else np.empty(0)
        )
        sums, rows, curves, requested = {}, {}, [], []
        for s in local:
            residual = s.observed - s.fitted
            observed = np.isfinite(residual)
            parts = [np.where(observed, v, np.nan) for v in (s.observed, s.fitted, residual)]
            # Sums over runs of count, sum and sum of squares give pooled variances.
            stats = np.array(
                [[observed.sum(0), np.nansum(v, 0), np.nansum(v**2, 0)] for v in parts]
            )
            sums[s.subject] = sums.get(s.subject, 0) + stats
            curve = autocorrelation(s.times, residual, curve_lags)
            curves.append(curve)
            requested.append(autocorrelation(s.times, residual, acf_lags))
            rows.setdefault(s.subject, []).append(curve[0] if len(curve) else None)
        shared, remainder, lag_one, people = [], [], [], {}
        for subject, stats in sums.items():
            count = np.maximum(stats[:, 0], 1)
            variance = stats[:, 2] / count - (stats[:, 1] / count) ** 2
            usable = (stats[0, 0] > 1) & (variance[0] > 0)
            scale = np.where(usable, variance[0], np.nan)
            shared.append(variance[1] / scale)
            remainder.append(variance[2] / scale)
            first = [r for r in rows[subject] if r is not None]
            lag_one.append(_feature_mean(first) if first else np.full(len(scale), np.nan))
            people[subject] = dict(
                shared=_mean(shared[-1]), residual=_mean(remainder[-1]), acf_lag1=_mean(lag_one[-1])
            )
        with np.errstate(invalid="ignore"):
            curve = (
                np.array(
                    [_mean(np.concatenate([c[k] for c in curves])) for k in range(len(curve_lags))],
                    dtype=float,
                )
                if len(curve_lags)
                else np.empty(0)
            )
        timescale, slow, status = exponential_timescale(curve_lags, curve)
        entry = dict(
            shared=_mean(np.concatenate(shared)),
            residual=_mean(np.concatenate(remainder)),
            sample_interval=step,
            acf_lag1=float(curve[0]) if len(curve) and np.isfinite(curve[0]) else None,
            acf={
                float(lag): _mean(np.concatenate([r[k] for r in requested]))
                for k, lag in enumerate(acf_lags)
            },
            timescale=timescale,
            slow_fraction=slow,
            timescale_status=status,
            n_participants=len(sums),
            per_feature=dict(
                shared=_feature_mean(shared),
                residual=_feature_mean(remainder),
                acf_lag1=_feature_mean(lag_one),
            ),
        )
        if detail:
            entry["participants"] = people
        output[modality] = entry
    return output


def _common_grid(group):
    """Place series with matching timestamps on one regular grid, or say why not."""
    steps = [_interval(s.times) for s in group]
    if not np.isfinite(steps).all():
        return None, None, "skipped: a series has a single timestamp"
    step, start = min(steps), min(s.times[0] for s in group)
    indices = [np.rint((s.times - start) / step).astype(int) for s in group]
    if any(
        np.abs(s.times - start - i * step).max() > 0.01 * step or len(np.unique(i)) != len(i)
        for s, i in zip(group, indices)
    ):
        return None, None, "skipped: timestamps do not lie on a common regular grid"
    return indices, step, None


def _loo_correlation(values):
    """Correlation of each participant's features with the others' mean, over time."""
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)
    donors = present.sum(0) - present
    others = np.where(donors > 0, (filled.sum(0) - filled) / np.maximum(donors, 1), np.nan)
    paired = present & np.isfinite(others)
    count = paired.sum(1)
    safe = np.maximum(count, 1)[:, None]
    a = np.where(paired, values, 0.0)
    b = np.where(paired, others, 0.0)
    a = np.where(paired, a - a.sum(1, keepdims=True) / safe, 0.0)
    b = np.where(paired, b - b.sum(1, keepdims=True) / safe, 0.0)
    scale = np.sqrt((a**2).sum(1) * (b**2).sum(1))
    good = (count >= MINIMUM_PAIRS) & (scale > 0)
    return np.where(good, (a * b).sum(1) / np.where(good, scale, 1.0), np.nan)


def _participant_correlation(series, min_shift, n_shifts, rng, detail):
    """Leave-one-out residual correlation across participants, with a shift null."""
    output = {}
    for modality in dict.fromkeys(s.modality for s in series):
        local = [s for s in series if s.modality == modality]
        observed, null, notes, people, used = [], np.zeros((n_shifts, 3)), [], {}, []
        features = {s.observed.shape[1] for s in local}
        for run in dict.fromkeys(s.run for s in local):
            group = [s for s in local if s.run == run]
            if len(group) < 2:
                notes.append(f"run {run}: skipped: fewer than two participants")
                continue
            if len({s.observed.shape[1] for s in group}) != 1:
                notes.append(f"run {run}: skipped: feature counts differ between participants")
                continue
            indices, step, reason = _common_grid(group)
            if reason:
                notes.append(f"run {run}: {reason}")
                continue
            length = max(i.max() for i in indices) + 1
            # Every participant moves at once, each within its own slot of the run,
            # so all pairs are displaced by at least ``shift`` samples and the
            # null keeps the dependence between participants' statistics.
            slot = length // len(group)
            shift = min(int(np.ceil(min_shift / step)), slot // 2)
            if shift < 1:
                notes.append(f"run {run}: skipped: too few samples to shift every participant")
                continue
            values = np.full((len(group), length, group[0].observed.shape[1]), np.nan)
            for k, (s, i) in enumerate(zip(group, indices)):
                values[k, i] = _centered(s.observed - s.fitted)
            r = _loo_correlation(values)
            if not np.isfinite(r).any():
                notes.append(f"run {run}: skipped: fewer than {MINIMUM_PAIRS} common timestamps")
                continue
            observed.append(r)
            used.append(shift * step)
            for s, row in zip(group, r):
                people.setdefault(s.subject, []).append(row)
            for n in range(n_shifts):
                offsets = rng.permutation(len(group)) * slot + rng.randint(
                    0, slot - shift + 1, size=len(group)
                )
                draw = _loo_correlation(
                    np.stack([np.roll(v, o, axis=0) for v, o in zip(values, offsets)])
                )
                null[n] += np.nansum(draw), np.nansum(draw**2), np.isfinite(draw).sum()
        if not observed:
            output[modality] = dict(
                correlation=None,
                null_mean=None,
                null_q95=None,
                p_shift=None,
                rms=None,
                rms_null_mean=None,
                rms_null_q95=None,
                rms_p_shift=None,
                min_shift_used=None,
                n_participants=len({s.subject for s in local}),
                per_feature=None,
                status="; ".join(notes) or "skipped",
            )
            continue
        pooled = np.concatenate([r.ravel() for r in observed])
        value, rms = _mean(pooled), float(np.sqrt(np.nanmean(pooled**2)))
        draws = null[:, 0] / np.maximum(null[:, 2], 1)
        rms_draws = np.sqrt(null[:, 1] / np.maximum(null[:, 2], 1))
        entry = dict(
            correlation=value,
            null_mean=float(draws.mean()),
            null_q95=float(np.quantile(draws, 0.95)),
            p_shift=float((1 + np.sum(draws >= value)) / (1 + n_shifts)),
            rms=rms,
            rms_null_mean=float(rms_draws.mean()),
            rms_null_q95=float(np.quantile(rms_draws, 0.95)),
            rms_p_shift=float((1 + np.sum(rms_draws >= rms)) / (1 + n_shifts)),
            min_shift_used=float(min(used)),
            n_participants=len(people),
            per_feature=_feature_mean([_feature_mean(r) for r in observed])
            if len(features) == 1
            else None,
            status="; ".join(["ok", *notes]),
        )
        if detail:
            entry["participants"] = {s: _mean(np.concatenate(rows)) for s, rows in people.items()}
        output[modality] = entry
    return output


def _resample(times, values, grid):
    """Linear interpolation onto ``grid``; exact where timestamps coincide.

    A grid point is NaN when it lies outside the observed samples or inside a
    gap longer than one and a half of the series' own sampling intervals, so
    masked stretches are not bridged. Returns the values and whether any point
    was interpolated.
    """
    output = np.full((len(grid), values.shape[1]), np.nan)
    interpolated = False
    step = _interval(times)
    for f in range(values.shape[1]):
        keep = np.isfinite(values[:, f])
        t, v = times[keep], values[keep, f]
        if len(t) < 2:
            continue
        right = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        left = right - 1
        exact = np.minimum(np.abs(grid - t[left]), np.abs(grid - t[right])) <= 1e-6 * step
        inside = (grid >= t[0] - 1e-6 * step) & (grid <= t[-1] + 1e-6 * step)
        good = inside & (exact | (t[right] - t[left] <= 1.5 * step))
        output[good, f] = np.interp(grid[good], t, v)
        interpolated |= bool(np.any(good & ~exact))
    return output, interpolated


def _squared_correlations(a, b, length):
    """Sum over feature pairs of squared cross-correlations at every circular lag.

    Columns are unit-norm with zeros where unobserved. Entry ``lag`` pairs
    ``a[t]`` with ``b[t + lag]``, indices modulo ``length``; zero padding to
    ``length`` beyond the series turns circular lags into ordinary ones.
    """
    fa = np.conj(np.fft.rfft(a, n=length, axis=0))
    fb = np.fft.rfft(b, n=length, axis=0)
    total = np.zeros(length)
    for column in fa.T:
        total += (np.fft.irfft(column[:, None] * fb, n=length, axis=0) ** 2).sum(1)
    return total


def _modality_cross_correlation(series, max_lag, min_shift, n_shifts, rng, detail):
    """Lagged residual cross-correlation between modalities within a participant."""
    output = {}
    modalities = list(dict.fromkeys(s.modality for s in series))
    lookup = {(s.subject, s.run, s.modality): s for s in series}
    units = list(dict.fromkeys((s.subject, s.run) for s in series))
    for first, second in combinations(modalities, 2):
        curves, tables, people, notes, steps, used, resampled = [], [], {}, [], [], [], False
        for subject, run in units:
            a, b = lookup.get((subject, run, first)), lookup.get((subject, run, second))
            if a is None or b is None:
                continue
            step = max(_interval(a.times), _interval(b.times))
            start, stop = max(a.times[0], b.times[0]), min(a.times[-1], b.times[-1])
            if not np.isfinite(step) or stop - start < 8 * step:
                notes.append(f"{subject}/{run}: skipped: too little overlap in time")
                continue
            if steps and abs(step - steps[0]) > 0.01 * steps[0]:
                notes.append(f"{subject}/{run}: skipped: sampling step differs from the others")
                continue
            grid = start + step * np.arange(int(np.floor((stop - start) / step + 1e-9)) + 1)
            reach = int(np.floor(max_lag / step + 1e-9))
            shift = int(np.ceil(min_shift / step))
            if reach < 1:
                notes.append("skipped: max_lag is shorter than the common sampling step")
                break
            if len(grid) < 4 * reach + 2:
                notes.append(f"{subject}/{run}: skipped: run shorter than four times max_lag")
                continue
            # A null offset must clear both the scanned lags and min_shift.
            shift = min(shift, (len(grid) - 2 * reach - 1) // 2)
            if shift < 1:
                notes.append(f"{subject}/{run}: skipped: too few samples to shift")
                continue
            xa, ia = _resample(a.times, a.observed - a.fitted, grid)
            xb, ib = _resample(b.times, b.observed - b.fitted, grid)
            (xa, ua), (xb, ub) = _standardized(xa), _standardized(xb)
            pairs = int(ua.sum() * ub.sum())
            if not pairs:
                notes.append(f"{subject}/{run}: skipped: no feature with residual variance")
                continue
            xa, xb = xa[:, ua], xb[:, ub]
            resampled |= ia or ib
            steps.append(step)
            used.append(shift * step)
            padded = _squared_correlations(xa, xb, len(grid) + reach) / pairs
            curve = np.sqrt(np.r_[padded[-reach:], padded[: reach + 1]])
            # A lag of k steps pairs len(grid) - k samples, the circular null all of
            # them. Rescale so unrelated series give the same level at every lag;
            # a true correlation at lag k is then understated by this same factor.
            overlap = 1 - np.abs(np.arange(-reach, reach + 1)) / len(grid)
            curve /= np.sqrt(overlap)
            circular = np.sqrt(_squared_correlations(xa, xb, len(grid)) / pairs)
            curves.append(curve)
            tables.append((circular, reach + shift))
            people.setdefault(subject, []).append(curve)
        if not curves:
            output[first, second] = dict(
                lags=None,
                correlation=None,
                null_mean=None,
                null_q95=None,
                peak_lag=None,
                peak=None,
                null_peak_q95=None,
                p_shift=None,
                min_shift_used=None,
                n_participants=0,
                step=None,
                resampled=None,
                status="; ".join(dict.fromkeys(notes)) or "skipped: no participant has both",
            )
            continue
        reach = (len(curves[0]) - 1) // 2
        window = np.arange(-reach, reach + 1)
        null = np.zeros((n_shifts, len(window)))
        for circular, clear in tables:
            offsets = rng.randint(clear, len(circular) - clear + 1, size=n_shifts)
            null += circular[(offsets[:, None] + window[None, :]) % len(circular)]
        null /= len(tables)
        curve = np.mean(curves, axis=0)
        peak = int(np.argmax(curve))
        entry = dict(
            lags=steps[0] * window,
            correlation=curve,
            null_mean=float(null.mean()),
            null_q95=float(np.quantile(null, 0.95)),
            peak_lag=float(steps[0] * window[peak]),
            peak=float(curve[peak]),
            null_peak_q95=float(np.quantile(null.max(1), 0.95)),
            p_shift=float((1 + np.sum(null.max(1) >= curve[peak])) / (1 + n_shifts)),
            min_shift_used=float(min(used)),
            n_participants=len(people),
            step=float(steps[0]),
            resampled=bool(resampled),
            status="; ".join(["ok", *dict.fromkeys(notes)]),
        )
        if detail:
            entry["participants"] = {s: np.mean(c, axis=0) for s, c in people.items()}
        output[first, second] = entry
    return output


def residual_report(
    series,
    *,
    acf_lags=(2.0, 5.0, 10.0, 30.0),
    max_lag=10.0,
    min_shift=30.0,
    n_shifts=100,
    random_state=0,
    detail=False,
    units="",
    reconstruction="",
):
    """Summarize ``ResidualSeries`` records; see the estimators' ``residual_report``."""
    acf_lags = _settings(acf_lags, max_lag, min_shift, n_shifts)
    if not series:
        raise ValueError("no residuals to summarize")
    rng = check_random_state(random_state)
    report = dict(
        variance=_variance_breakdown(series, acf_lags, detail),
        participant_correlation=_participant_correlation(series, min_shift, n_shifts, rng, detail),
        modality_cross_correlation=_modality_cross_correlation(
            series, max_lag, min_shift, n_shifts, rng, detail
        ),
        settings=dict(
            acf_lags=acf_lags.tolist(),
            max_lag=float(max_lag),
            min_shift=float(min_shift),
            n_shifts=int(n_shifts),
            random_state=random_state if isinstance(random_state, (int, np.integer)) else None,
            units=units,
            reconstruction=reconstruction,
            cross_correlation_statistic="root mean square of feature-pair correlations",
            null="circular shifts of at least min_shift seconds",
        ),
    )
    if detail:
        residuals = {}
        for s in series:
            residuals.setdefault(s.subject, {}).setdefault(s.run, {})[s.modality] = dict(
                times=s.times, residual=s.observed - s.fitted
            )
        report["residuals"] = residuals
    return report
