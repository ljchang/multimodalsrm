"""Residuals of the conditioning observations from the existing prediction code."""

import numpy as np

from ..residuals import ResidualSeries
from .prediction import result, selected_draws

# Largest condition number of the loadings used to recover a modality's
# filtered factors from a few queried features.
MAXIMUM_CONDITION = 1e6


def _loadings(problem, draws):
    """Draw by key by factor loadings and draw by key offsets."""
    weights, offsets = [], []
    for x in draws:
        w, mu = problem.arrays(x)[:2]
        weights.append(np.asarray(w).reshape(len(problem.keys), problem.features))
        offsets.append(np.asarray(mu))
    return np.array(weights), np.array(offsets)


def _basis(weights):
    """Keys whose loadings span the factors best, or None if they do not."""
    features = weights.shape[-1]
    if weights.shape[1] < features:
        return None
    chosen, remainder = [], weights[0].copy()
    for _ in range(features):
        k = int(np.argmax((remainder**2).sum(1)))
        norm = np.linalg.norm(remainder[k])
        if norm == 0:
            return None
        chosen.append(k)
        direction = remainder[k] / norm
        remainder = remainder - np.outer(remainder @ direction, direction)
    if max(np.linalg.cond(w[chosen]) for w in weights) > MAXIMUM_CONDITION:
        return None
    return chosen


def filtered_signal(model, run, modality, times, keys, draws):
    """Posterior-mean filtered signal of ``keys`` at ``times``, time by key.

    The filtered-signal mean of a feature is its offset plus its loadings
    times the conditional mean of the response-filtered factors, which all
    features of a modality share. ``prediction.result`` is therefore queried
    for as many features as there are factors, chosen for well-conditioned
    loadings, and the others follow by linear algebra. That keeps the cost
    independent of the number of features. When the loadings cannot be
    inverted, or run baselines add a term of their own to each feature, every
    feature is queried directly. The state-space smoother already returns the
    filtered factors, so it is run once and projected as its own ``project``
    does.
    """
    problem = model.problem_
    index = np.array([problem.keys.index(k) for k in keys])
    if problem.linear_algebra == "state_space" and not problem.run_baseline_sd:
        from ._backend import runtime
        from .state_space import smoother

        jax, jnp, _, _ = runtime()
        smooth = jax.jit(
            smoother(problem, run, np.asarray(times), problem.modalities.index(modality))
        )
        fitted = np.zeros((len(times), len(keys)))
        for x, w, mu in zip(draws, *_loadings(problem, draws)):
            fitted += np.asarray(smooth(jnp.asarray(x))[0]) @ w[index].T + mu[index]
        return fitted / len(draws)
    direct = modality in problem.run_baseline_sd
    if not direct:
        weights, offsets = _loadings(problem, draws)
        chosen = _basis(weights[:, index])
        direct = chosen is None
    if direct:
        if problem.linear_algebra == "grouped" and not problem.run_baseline_sd:
            return np.array(result(model, run, times, draws, keys, include_noise=False).values)
        return np.column_stack(
            [result(model, run, times, draws, [k], include_noise=False).values for k in keys]
        )
    basis = index[chosen]
    queried = result(
        model, run, times, draws, [problem.keys[k] for k in basis], include_noise=False
    ).component_means
    fitted = np.zeros((len(times), len(keys)))
    for mean, w, mu in zip(queried, weights, offsets):
        factors = np.linalg.solve(w[basis], (mean - mu[basis]).T).T
        fitted += factors @ w[index].T + mu[index]
    return fitted / len(draws)


def residual_series(model, max_draws=200):
    """Observed and fitted values of every conditioning observation, by series."""
    problem = model.problem_
    draws = selected_draws(model, max_draws)
    series = []
    for run, system in problem.systems.items():
        rows = {}
        for i, key in enumerate(system.keys):
            rows.setdefault(key, []).append(i)
        for modality in problem.modalities:
            keys = [k for k in rows if k[1] == modality]
            if not keys:
                continue
            times = np.unique(np.concatenate([system.times[rows[k]] for k in keys]))
            fitted = filtered_signal(model, run, modality, times, keys, draws)
            if not np.isfinite(fitted).all():
                raise ValueError(
                    f"no reconstruction is available for {modality!r} in run {run!r} at its "
                    "observed times; residual_report supports dense, grouped, state_space and "
                    "spectral fits queried inside their support"
                )
            for subject in dict.fromkeys(k[0] for k in keys):
                local = [k for k in keys if k[0] == subject]
                width = max(k[2] for k in local) + 1
                own = np.unique(np.concatenate([system.times[rows[k]] for k in local]))
                observed = np.full((len(own), width), np.nan)
                mean = np.full((len(own), width), np.nan)
                for k in local:
                    at = system.times[rows[k]]
                    observed[np.searchsorted(own, at), k[2]] = system.values[rows[k]]
                    mean[np.searchsorted(own, at), k[2]] = fitted[
                        np.searchsorted(times, at), keys.index(k)
                    ]
                series.append(ResidualSeries(subject, run, modality, own, observed, mean))
    return series
