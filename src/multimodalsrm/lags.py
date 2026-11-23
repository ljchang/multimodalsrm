"""Coarse lag scans and reference-free response timing summaries.

A shared latent response makes only *relative* modality timing identifiable:
delaying every response by a constant is absorbed by shifting the latent in
time. The scan here estimates relative delays directly from the observations,
before any model is fitted, so that response lags start in the right basin and
their bounds can stay narrow. The timing report states fitted delays relative
to a named reference or to the across-modality mean, never as absolute times.

Nothing here changes a model objective.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar

from .data import normalize_data
from .kernels import Gamma, Gaussian, Identity, Kernel, Response, SampledKernel

# Mean excess squared canonical correlation at the best lag, per participant
# and component. Synthetic pairs that the scan resolved had values above 0.15;
# pairs it could not resolve stayed below 0.01.
MINIMUM_PROMINENCE = 0.03


def peak_delay(kernel):
    """Time of a response's largest positive value, in timestamp units.

    This is the timing quantity to compare across response families. For
    causal families the ``lag`` parameter is the onset, which trades off
    against the shape parameters; the peak is far better determined.
    Identity is an impulse at zero.
    """
    if not isinstance(kernel, Kernel):
        raise ValueError("peak_delay requires a response kernel")
    if isinstance(kernel, Identity):
        return 0.0
    if isinstance(kernel, Gaussian):
        return float(kernel.lag)
    if type(kernel) is Gamma:
        return float(kernel.lag + (kernel.shape - 1) * kernel.scale)
    if isinstance(kernel, SampledKernel):
        return float(kernel.lags[int(np.argmax(kernel.values))])
    grid = np.linspace(*kernel.support, 4097)
    values = kernel.evaluate(grid)
    i = int(np.argmax(values))
    if 0 < i < len(grid) - 1:
        return float(
            minimize_scalar(
                lambda t: -float(kernel.evaluate(t)),
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-10},
            ).x
        )
    return float(grid[i])


def _has_free_lag(response):
    return "lag" in response.free_parameters


@dataclass(frozen=True)
class LagScan:
    """Relative modality delays estimated from observations before fitting.

    ``delays`` are seconds (timestamp units) relative to the across-modality
    mean: only their differences carry information. ``pairwise[(a, b)]`` holds
    the directly scanned delay of ``b`` relative to ``a`` with its
    ``prominence`` (how far the best lag stands above the typical lag) and
    whether it sits ``at_limit`` of the scanned range. ``closure_error`` is the
    largest disagreement between a pairwise delay and the fitted delays; it is
    None with fewer than three modalities. ``notes`` explains every reason
    ``reliable`` is False.
    """

    delays: dict
    pairwise: dict
    closure_error: float | None
    max_lag: float
    step: float
    components: int
    notes: tuple = field(default_factory=tuple)

    @property
    def reliable(self):
        return not self.notes

    def relative_to(self, reference):
        """Delays with ``reference`` at zero."""
        if reference not in self.delays:
            raise ValueError(f"unknown reference modality {reference!r}")
        return {m: d - self.delays[reference] for m, d in self.delays.items()}

    def seed_responses(self, responses, *, margin=2.0):
        """Return responses whose free lags start at the scanned delays.

        Each free lag is moved so the response *peaks* at its scanned delay,
        and its bounds become that lag plus or minus ``margin``. Narrow bounds
        matter twice: they keep the optimizer in the scanned basin, and they
        keep the fixed support envelope small, so fewer edge observations are
        excluded. The clock is taken from modalities whose lag is not
        estimated (fixed lags and Identity responses); with every lag free,
        the mean peak delay of the supplied responses is preserved.
        Modalities the scan could not place are returned unchanged.
        """
        if not isinstance(responses, Mapping) or not all(
            isinstance(r, Response) for r in responses.values()
        ):
            raise ValueError("responses must map modality names to Response objects")
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be positive finite")
        placed = [m for m in responses if np.isfinite(self.delays.get(m, np.nan))]
        peaks = {m: peak_delay(responses[m].initial_kernel()) for m in placed}
        anchors = [m for m in placed if not _has_free_lag(responses[m])] or placed
        if not anchors:
            return dict(responses)
        origin = float(np.mean([peaks[m] - self.delays[m] for m in anchors]))
        seeded = dict(responses)
        for m in placed:
            response = responses[m]
            if not _has_free_lag(response):
                continue
            kernel = response.initial_kernel()
            lag = float(kernel.lag + self.delays[m] + origin - peaks[m])
            seeded[m] = replace(
                response,
                kernel=response.kernel.with_parameters(lag=lag),
                bounds={**response.bounds, "lag": (lag - margin, lag + margin)},
            )
        return seeded


def _resampled(series, step):
    """Per-feature linear interpolation of observed values onto a regular grid."""
    grid = np.arange(series.times[0], series.times[-1] + 1e-9 * step, step)
    columns = []
    for j in range(series.values.shape[1]):
        observed = series.mask[:, j]
        if observed.sum() >= 2:
            columns.append(np.interp(grid, series.times[observed], series.values[observed, j]))
    return grid, (np.column_stack(columns) if columns else np.empty((len(grid), 0)))


def _components(runs, modality, step, count):
    """Leading principal time courses of one participant's modality, by run."""
    resampled = {
        r: _resampled(mods[modality], step) for r, mods in runs.items() if modality in mods
    }
    widths = {x.shape[1] for _, x in resampled.values()}
    if not resampled or len(widths) != 1 or 0 in widths:
        return {}
    stacked = np.vstack([x for _, x in resampled.values()])
    center, scale = stacked.mean(0), stacked.std(0)
    keep = scale > np.finfo(float).eps * max(1.0, float(np.abs(center).max(initial=0.0)))
    if not keep.any() or len(stacked) < 3:
        return {}
    _, _, basis = np.linalg.svd(
        (stacked[:, keep] - center[keep]) / scale[keep], full_matrices=False
    )
    basis = basis[: min(count, len(basis))]
    return {
        r: (grid, ((x[:, keep] - center[keep]) / scale[keep]) @ basis.T)
        for r, (grid, x) in resampled.items()
    }


def _canonical_score(parts):
    """Sum of squared canonical correlations over pooled paired samples."""
    x = np.vstack([p[0] for p in parts])
    y = np.vstack([p[1] for p in parts])
    qx, _ = np.linalg.qr(x - x.mean(0))
    qy, _ = np.linalg.qr(y - y.mean(0))
    return float(np.sum(np.linalg.svd(qx.T @ qy, compute_uv=False) ** 2))


def _pair_curve(first, second, lags, minimum_overlap):
    """Score of ``second`` observed ``lag`` after ``first``, for one participant."""
    curve = np.full(len(lags), np.nan)
    shared = [r for r in first if r in second]
    for i, lag in enumerate(lags):
        parts = []
        for r in shared:
            grid, x = first[r]
            other_grid, y = second[r]
            query = grid + lag
            keep = (query >= other_grid[0]) & (query <= other_grid[-1])
            if keep.sum() < max(minimum_overlap * min(len(grid), len(other_grid)), 4 * x.shape[1]):
                continue
            paired = np.column_stack(
                [np.interp(query[keep], other_grid, y[:, j]) for j in range(y.shape[1])]
            )
            parts.append((x[keep], paired))
        if parts:
            curve[i] = _canonical_score(parts)
    return curve


def scan_lags(data, *, max_lag=15.0, step=0.25, components=2, modalities=None):
    """Estimate relative modality delays by lagged canonical correlation.

    For every pair of modalities and every participant who has both, the
    leading ``components`` principal time courses of one stream are compared
    with those of the other at each lag from ``-max_lag`` to ``max_lag``.
    Pairwise delays are then reconciled by prominence-weighted least squares
    into one delay per modality. No model is fitted; the cost is a small
    fraction of one fit.

    The scan locates a basin for response lags; it is not a timing estimate.
    It returns approximately the delay between response peaks. It can fail at
    very low signal-to-noise and can alias by one period on a near-periodic
    signal; both cases are reported in ``notes`` (low prominence, pairwise
    delays that do not add up) and make ``reliable`` False. Lags are in
    timestamp units, and at least half of each run must overlap at every
    scanned lag, so ``max_lag`` should stay well below half a run.
    """
    data = normalize_data(data)
    if not np.isfinite([max_lag, step]).all() or step <= 0 or max_lag < step:
        raise ValueError("max_lag and step must be positive finite with max_lag >= step")
    if (
        isinstance(components, bool)
        or not isinstance(components, (int, np.integer))
        or components < 1
    ):
        raise ValueError("components must be a positive integer")
    observed = list(
        dict.fromkeys(m for runs in data.values() for mods in runs.values() for m in mods)
    )
    names = observed if modalities is None else list(modalities)
    if len(set(names)) != len(names) or not set(names) <= set(observed):
        raise ValueError("modalities must be distinct observed modality names")
    if len(names) < 2:
        raise ValueError("a lag scan needs at least two modalities")
    steps = int(round(max_lag / step))
    lags = step * np.arange(-steps, steps + 1)
    parts = {
        (s, m): _components(runs, m, step, components) for s, runs in data.items() for m in names
    }
    notes, pairwise, rows, targets, weights = [], {}, [], [], []
    for a, b in combinations(names, 2):
        curves, ranks = [], []
        for s in data:
            if not (parts[s, a] and parts[s, b]):
                continue
            curve = _pair_curve(parts[s, a], parts[s, b], lags, 0.5)
            if np.isfinite(curve).any():
                curves.append(curve)
                ranks.append(
                    min(
                        next(iter(parts[s, a].values()))[1].shape[1],
                        next(iter(parts[s, b].values()))[1].shape[1],
                    )
                )
        if not curves:
            notes.append(f"{a} and {b} are never observed together for long enough to compare")
            continue
        total = np.nansum(curves, axis=0)
        total[np.all(np.isnan(curves), axis=0)] = np.nan
        best = int(np.nanargmax(total))
        delay = float(lags[best])
        if 0 < best < len(lags) - 1 and np.isfinite(total[best - 1 : best + 2]).all():
            left, middle, right = total[best - 1 : best + 2]
            curvature = left - 2 * middle + right
            if curvature < 0:
                delay += float(step * 0.5 * (left - right) / curvature)
        prominence = float((total[best] - np.nanmedian(total)) / np.sum(ranks))
        scanned = np.flatnonzero(np.isfinite(total))
        at_limit = bool(best in (scanned[0], scanned[-1]))
        pairwise[a, b] = dict(
            delay=delay, prominence=prominence, at_limit=at_limit, participants=len(curves)
        )
        if prominence < MINIMUM_PROMINENCE:
            notes.append(
                f"{a}/{b}: no lag stands out (prominence {prominence:.3f} < {MINIMUM_PROMINENCE})"
            )
        if at_limit:
            notes.append(f"{a}/{b}: best lag {delay:g} is at the edge of the scanned range")
        row = np.zeros(len(names))
        row[names.index(b)], row[names.index(a)] = 1.0, -1.0
        rows.append(row)
        targets.append(delay)
        weights.append(max(prominence, 1e-6))
    delays = {m: float("nan") for m in names}
    closure = None
    if rows:
        design, target, weight = np.array(rows), np.array(targets), np.sqrt(weights)
        # Solve within each connected set of modalities, centred on its mean.
        remaining = set(range(len(names)))
        groups = []
        while remaining:
            group, frontier = set(), [remaining.pop()]
            while frontier:
                i = frontier.pop()
                group.add(i)
                linked = {int(j) for row in design[design[:, i] != 0] for j in np.flatnonzero(row)}
                frontier.extend(linked - group)
                remaining -= linked
            groups.append(sorted(group))
        largest = max(groups, key=len)
        if len(largest) < len(names):
            missing = [names[i] for g in groups if g is not largest for i in g]
            notes.append(f"no usable comparison links {missing} to the other modalities")
        used = np.all(design[:, [i for i in range(len(names)) if i not in largest]] == 0, axis=1)
        local = design[used][:, largest] * weight[used, None]
        solution = np.linalg.lstsq(
            np.vstack([local, np.ones(len(largest))]),
            np.r_[target[used] * weight[used], 0.0],
            rcond=None,
        )[0]
        for i, value in zip(largest, solution):
            delays[names[i]] = float(value)
        if len(largest) >= 3:
            closure = float(np.max(np.abs(design[used][:, largest] @ solution - target[used])))
            if closure > max(1.0, 4 * step):
                notes.append(
                    f"pairwise delays disagree by {closure:.2f}: the signal may be near-periodic"
                )
    return LagScan(
        delays, pairwise, closure, float(max_lag), float(step), int(components), tuple(notes)
    )


def resolve_reference(reference, estimated):
    """Pick what defines zero delay: a modality, the model clock, or the mean.

    An explicit name wins. Otherwise a single modality whose lag is not
    estimated (a fixed lag or an Identity response) is the reference. Several
    such modalities already fix the model's clock between them, so delays are
    reported on that clock unchanged. With every lag estimated the common
    offset is arbitrary and delays are centred on the across-modality mean.
    """
    if reference is not None:
        if reference not in estimated:
            raise ValueError(f"reference {reference!r} has no fitted response")
        return reference, "requested_reference"
    fixed = [m for m, free in estimated.items() if not free]
    if len(fixed) == 1:
        return fixed[0], "only_modality_without_an_estimated_lag"
    if fixed:
        return None, "model_clock_set_by_modalities_without_estimated_lags"
    return None, "across_modality_mean"


def timing_summary(peaks, estimated, reference=None):
    """Reference-free timing table from peak delays (scalars or draw arrays).

    Only differences between modalities are identified, so every delay is
    reported relative to ``reference`` or to the across-modality mean.
    ``clock_origin`` is what was subtracted: a latent event at model time
    ``t`` occurs at ``t + clock_origin`` on the reported clock.
    """
    if not peaks:
        raise ValueError("timing needs at least one response")
    reference, convention = resolve_reference(reference, estimated)
    arrays = {m: np.asarray(v, dtype=float) for m, v in peaks.items()}
    if reference is not None:
        origin = arrays[reference]
    elif convention == "across_modality_mean":
        origin = np.mean(np.stack(np.broadcast_arrays(*arrays.values())), axis=0)
    else:
        origin = np.zeros_like(next(iter(arrays.values())))

    def describe(values):
        values = np.asarray(values, dtype=float)
        return (
            float(values)
            if values.ndim == 0
            else dict(mean=float(values.mean()), sd=float(values.std()), draws=values.copy())
        )

    return dict(
        quantity="peak_delay",
        units="timestamp units (normally seconds)",
        convention=convention,
        reference=reference,
        clock_origin=describe(origin),
        relative_peak_delay={m: describe(v - origin) for m, v in arrays.items()},
        pairwise_difference={
            f"{b} - {a}": describe(arrays[b] - arrays[a]) for a, b in combinations(arrays, 2)
        },
        lag_estimated=dict(estimated),
        interpretation=(
            "Differences between modalities are estimated; the common offset is a "
            "reporting convention and does not establish physiological timing."
        ),
    )


def regularized_timing_report(model, reference=None):
    """Timing summary of a fitted ``MultimodalSRM``."""
    responses = model.responses_
    estimated, peaks, participants = {}, {}, {}
    for m, response in responses.items():
        fitted = {s: mods[m] for s, mods in model.subject_kernels_.items() if m in mods}
        if not fitted:
            continue
        estimated[m] = "lag" in response.free_parameters
        individual = {s: peak_delay(k) for s, k in fitted.items()}
        group = model.group_kernels_.get(m)
        peaks[m] = (
            peak_delay(group) if group is not None else float(np.mean(list(individual.values())))
        )
        if response.pooling != "shared" and response.free_parameters:
            participants[m] = individual
    report = timing_summary(peaks, estimated, reference)
    origin = report["clock_origin"]
    report["peak_delay_model_clock"] = dict(peaks)
    if participants:
        report["participant_relative_peak_delay"] = {
            m: {s: value - origin for s, value in values.items()}
            for m, values in participants.items()
        }
    rows = model.optimization_diagnostics_["kernel_stationarity"].get("parameters", [])
    report["lags_at_bounds"] = [
        dict(
            modality=r["modality"],
            subjects=list(r["subjects"]),
            bound="lower" if r["at_lower_bound"] else "upper",
        )
        for r in rows
        if r["parameter"] == "lag" and (r["at_lower_bound"] or r["at_upper_bound"])
    ]
    scan = getattr(model, "lag_scan_", None)
    if scan is not None:
        shared = [m for m in peaks if np.isfinite(scan.delays.get(m, np.nan))]
        if len(shared) >= 2:
            shift = np.mean([peaks[m] - scan.delays[m] for m in shared])
            report["scan_disagreement"] = {
                m: float(peaks[m] - scan.delays[m] - shift) for m in shared
            }
            report["scan_reliable"] = scan.reliable
    return report
