"""Posterior mixing diagnostics and descriptive residual/donor checks."""

import numpy as np

from ..data import validate_times


def diagnostic_summary(values, *, name="parameter"):
    """Keep chain/draw axes and the declared 5%/95% tail-ESS definition.

    An explicit Dataset works with both ArviZ 0.x and 1.x. The latter's
    default tail probabilities depend on its credible-interval setting;
    upgrading diagnostics must not silently change our qualification gate.
    """
    import arviz as az
    import xarray as xr

    values = np.asarray(values)
    if values.ndim not in (2, 3):
        raise ValueError("diagnostic values must have chain/draw[/quantity] shape")
    dims = ("chain", "draw") + ((f"{name}_dim_0",) if values.ndim == 3 else ())
    dataset = xr.Dataset({name: (dims, values)})
    result = az.summary(dataset, kind="diagnostics", round_to="none")
    tails = az.ess(dataset, method="tail", prob=(0.05, 0.95))
    result["ess_tail"] = np.asarray(tails[name]).reshape(-1)
    return result


def _arrays(*arrays):
    values = [np.asarray(a) for a in arrays]
    if not values or values[0].ndim != 2 or values[0].shape[1] == 0:
        raise ValueError("observations must be time by feature")
    if any(a.shape != values[0].shape for a in values):
        raise ValueError("all observation arrays must have identical shapes")
    return values


def _correlation(a, b):
    if len(a) < 3:
        return None
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / scale, -1.0, 1.0)) if scale > 0 else None


def residual_summary(times, observed, mean, variance, mask, *, lags, tolerance=1e-6):
    """Per-feature prediction residuals and lagged-pair Pearson correlation.

    Call separately for independent runs. Pairs use actual timestamps separated
    by each positive lag within ``tolerance`` seconds; no interpolation, time
    binning or compression across masked rows. At least three pairs and nonzero
    paired variances are needed for a correlation. This is a descriptive ACF,
    not an iid-noise test: target prediction errors can remain correlated under
    a valid GP. Standardized RMS uses marginal predictive observation variance.
    """
    times = validate_times(times)
    y, mu, v, mask = _arrays(observed, mean, variance, mask)
    if len(times) != len(y) or mask.dtype.kind != "b":
        raise ValueError("times and Boolean mask must match observations")
    lags = np.asarray(lags, dtype=float)
    if (
        lags.ndim != 1
        or not len(lags)
        or not np.isfinite(lags).all()
        or np.any(lags <= 0)
        or len(np.unique(lags)) != len(lags)
    ):
        raise ValueError("lags must be distinct positive finite seconds")
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or 2 * tolerance >= lags.min()
        or (len(times) > 1 and 2 * tolerance >= np.diff(times).min())
    ):
        raise ValueError("tolerance must be smaller than half the lag and timestamp spacing")
    finite = mask & np.isfinite(y) & np.isfinite(mu) & np.isfinite(v)
    if np.any(v[finite] <= 0):
        raise ValueError("predictive observation variance must be positive")
    pairs = []
    for lag in lags:
        requested = times + lag
        right = np.searchsorted(times, requested)
        a, b = np.clip(right - 1, 0, len(times) - 1), np.clip(right, 0, len(times) - 1)
        closest = np.where(abs(times[a] - requested) <= abs(times[b] - requested), a, b)
        valid = (abs(times[closest] - requested) <= tolerance) & (closest > np.arange(len(times)))
        pairs.append((np.flatnonzero(valid), closest[valid]))
    output = []
    for feature in range(y.shape[1]):
        good = finite[:, feature]
        residual = y[:, feature] - mu[:, feature]
        e = residual[good]
        acf = []
        for lag, (i, j) in zip(lags, pairs):
            keep = good[i] & good[j]
            acf.append(
                dict(
                    lag_seconds=float(lag),
                    n_pairs=int(keep.sum()),
                    correlation=_correlation(residual[i[keep]], residual[j[keep]]),
                )
            )
        output.append(
            dict(
                feature=feature,
                n_values=int(good.sum()),
                n_excluded=int((~good).sum()),
                residual_mean=float(e.mean()) if len(e) else None,
                rmse=float(np.sqrt(np.mean(e**2))) if len(e) else None,
                standardized_rms=float(np.sqrt(np.mean(e**2 / v[good, feature])))
                if len(e)
                else None,
                acf=acf,
            )
        )
    return dict(
        features=output,
        lags_seconds=lags.tolist(),
        pair_tolerance_seconds=tolerance,
        correlation_definition="Pearson on valid timestamp pairs; at least three pairs",
        calibration_established=False,
    )


def compare_predictions(observed, full, removed, mask):
    """Compare on identical finite target rows; negative delta MSE favors removal."""
    y, full, removed, mask = _arrays(observed, full, removed, mask)
    if mask.dtype.kind != "b":
        raise ValueError("mask must be Boolean")
    good = mask & np.isfinite(y) & np.isfinite(full) & np.isfinite(removed)
    output = []
    for f in range(y.shape[1]):
        take = good[:, f]
        a, b = y[take, f] - full[take, f], y[take, f] - removed[take, f]
        output.append(
            dict(
                feature=f,
                n_values=int(take.sum()),
                mse_full=float(np.mean(a**2)) if len(a) else None,
                mse_removed=float(np.mean(b**2)) if len(a) else None,
                delta_mse=float(np.mean(b**2) - np.mean(a**2)) if len(a) else None,
                prediction_change_rms=float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else None,
            )
        )
    return output


def fixed_prediction(problem, parameters, run, times, *, target, drop_modalities=()):
    """Predict a target group after deleting donor modalities, without refitting.

    ``problem`` must already exclude every feature of the participant-modality
    target from this run. Delete only prepared, support-eligible donor rows;
    retain the original domain/support policy externally for all comparisons.
    Return noisy-observation marginal moments in the problem's supplied units.
    Empty donors give the GP prior at the same parameters. This intervention
    measures model dependence, not causal influence or a refitted ablation.
    Dense/grouped problems are supported; spectral-basis changes are not implied.
    """
    from .._observation_preparation import ObservationSystem
    from .prediction import project
    from .problem import BayesianProblem

    times = validate_times(times)
    x = np.asarray(parameters, dtype=float)
    if x.shape != (len(problem.names),) or not np.isfinite(x).all():
        raise ValueError("parameters must be a finite physical vector matching the problem")
    if any(not lo <= value <= hi for value, (lo, hi) in zip(x, problem.bounds)):
        raise ValueError("physical parameters must lie within the declared bounds")
    if any(
        value <= 0
        for name, value in zip(problem.names, x)
        if name[0] == "noise" or (name[0] == "filter" and name[-1] == "width")
    ):
        raise ValueError("noise and width parameters must be positive")
    target = tuple(target)
    if target not in problem.groups:
        raise ValueError("target mapping is unavailable")
    system = problem.systems[run]
    if any(k[:2] == target for k in system.keys):
        raise ValueError("target observations must be excluded before diagnostic prediction")
    drop = set(drop_modalities)
    if not drop <= set(problem.modalities):
        raise ValueError("unknown donor modality")
    if problem.linear_algebra not in ("dense", "grouped"):
        raise ValueError("fixed diagnostics support dense/grouped problems only")
    keep = np.array([k[1] not in drop for k in system.keys])
    keys = [k for k in problem.keys if k[:2] == target]
    if keep.any():
        reduced = problem
        if not keep.all():
            reduced = BayesianProblem(
                problem.adapter,
                problem.priors,
                anchor=problem.anchor,
                reference_modality=problem.reference_convention["reference_modality"],
                systems={
                    run: ObservationSystem(
                        system.times[keep],
                        [k for k, take in zip(system.keys, keep) if take],
                        system.values[keep],
                    )
                },
                linear_algebra=problem.linear_algebra,
                run_baseline_sd=problem.run_baseline_sd,
                noise_timescales=problem.noise_timescales,
                response_quadrature_order=problem.response_quadrature_order,
                state_space_gaussian=problem.state_space_gaussian,
                length_scale_prior=problem.length_scale_prior,
            )
            if reduced.names != problem.names:
                raise ValueError("donor removal changed parameter coordinates")
        parts = [project(reduced, x[None], run, times, key=k, include_noise=True) for k in keys]
        mean = np.column_stack([p[0][0] for p in parts])
        variance = np.column_stack([p[1][0] for p in parts])
    else:
        weights, offsets, noise, widths, lags = map(np.asarray, problem.arrays(x))
        mi = problem.modalities.index(target[1])
        prior = float(
            problem.temporal_covariance(
                x, np.zeros(1), np.array([mi]), np.zeros(1), np.array([mi])
            )[0, 0]
        )
        indices = [problem.keys.index(k) for k in keys]
        squared_norm = (
            weights[indices] ** 2
            if problem.features == 1
            else np.sum(weights[indices] ** 2, axis=1)
        )
        mean = np.tile(offsets[indices], (len(times), 1))
        variance = np.tile(
            prior * squared_norm
            + noise[problem.groups.index(target)]
            + problem.run_baseline_sd.get(target[1], 0.0) ** 2,
            (len(times), 1),
        )
    return dict(
        mean=mean,
        variance=variance,
        remaining_observations=int(keep.sum()),
        removed_observations=int((~keep).sum()),
        target=list(target),
        dropped_modalities=sorted(drop),
        parameter_refitting=False,
    )
