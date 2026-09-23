"""Common rows and width/peak priors for a research response-family comparison.

MAP density is defined in physical FWHM and peak-time coordinates for every
candidate. The loading, offset, variance, and EDA coordinates are unchanged.
This module does not change the production inference backend.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from gp_response_candidates import candidate_model
from scipy.optimize import brentq

from multimodalsrm import Gamma, Gaussian, TimeSeries
from multimodalsrm.bayesian import BayesianPriors, Prior
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.problem import BayesianProblem, _group_priors
from multimodalsrm.bayesian.workflow import TrainingStandardizer

CANDIDATES = ("gamma2", "gamma3", "gamma6", "gaussian")
BLOCKS = {"brain": (220, 240), "face": (260, 280), "rating": (300, 320), "eda": (340, 360)}
BLOCK_FOLDS = {
    "original": BLOCKS,
    "rotated": {"brain": (260, 280), "face": (300, 320), "rating": (340, 360), "eda": (220, 240)},
    "late": {"brain": (300, 320), "face": (340, 360), "rating": (380, 400), "eda": (260, 280)},
}
WIDTH_BOUNDS = {"face": (0.2, 7.0), "rating": (1.0, 16.0)}
PEAK_BOUNDS = {"face": (-4.0, 10.0), "rating": (-4.0, 12.0)}


def width_factor(candidate):
    if candidate == "gaussian":
        return np.sqrt(8 * np.log(2))
    mode = int(candidate.removeprefix("gamma")) - 1

    def half(t):
        return mode * np.log(t / mode) - (t - mode) + np.log(2)

    return brentq(half, mode, mode + 30) - brentq(half, 1e-12, mode)


def make_model(data, candidate, algebra="state_space", order=192, *, features=1):
    gaussian = candidate == "gaussian"
    shape = 3 if gaussian else int(candidate.removeprefix("gamma"))
    model = candidate_model(
        data,
        features=features,
        family="gaussian" if gaussian else "gamma",
        face_shape=shape,
        rating_shape=shape,
        algebra=algebra,
    )
    factor = width_factor(candidate)
    responses = dict(model.responses)
    priors = {m: dict(p) for m, p in model.priors.filters.items()}
    for m in ("face", "rating"):
        lo, hi = np.asarray(WIDTH_BOUNDS[m]) / factor
        plo, phi = PEAK_BOUNDS[m]
        bounds = {"width" if gaussian else "scale": (lo, hi), "lag": (plo, phi)}
        if not gaussian:
            bounds["lag"] = (plo - (shape - 1) * hi, phi - (shape - 1) * lo)
        width = 3.4 / factor
        kernel = (
            Gaussian(width=width, lag=2)
            if gaussian
            else Gamma(shape=shape, scale=width, lag=2 - (shape - 1) * width)
        )
        responses[m] = replace(responses[m], kernel=kernel, bounds=bounds)
        # These native-coordinate priors are placeholders used to prepare the
        # production likelihood. CanonicalProblem supplies the actual density.
        priors[m] = {p: Prior.uniform(*interval) for p, interval in bounds.items()}
    model.set_params(
        responses=responses,
        priors=BayesianPriors(
            loading_sd=1.5, offset_sd=1, noise=model.priors.noise, filters=priors
        ),
        response_quadrature_order=None if algebra == "state_space" else order,
    )
    return model


def with_masks(data, masks, values=None):
    result = {}
    for s, runs in data.items():
        for r, mods in runs.items():
            for m, ts in mods.items():
                key = (s, r, m)
                result.setdefault(s, {}).setdefault(r, {})[m] = TimeSeries(
                    ts.values if values is None else values[key], ts.times, masks[key]
                )
    return result


def split_data(data, *, fold="original", excluded_folds=None):
    """Predeclare blocks; standardize using only common eligible training rows."""
    blocks = BLOCK_FOLDS[fold]
    excluded_folds = (fold,) if excluded_folds is None else tuple(excluded_folds)
    if fold not in excluded_folds or any(f not in BLOCK_FOLDS for f in excluded_folds):
        raise ValueError("excluded_folds must contain the scoring fold and known fold names")
    models = [make_model(data, candidate) for candidate in CANDIDATES]
    support = {
        m: (
            min(model.responses[m].support_envelope()[0] for model in models),
            max(model.responses[m].support_envelope()[1] for model in models),
        )
        for m in models[0].responses
    }
    domains = {}
    for runs in data.values():
        for r, mods in runs.items():
            times = np.concatenate([ts.times[ts.mask.any(axis=1)] for ts in mods.values()])
            a, b = domains.get(r, (np.inf, -np.inf))
            domains[r] = (min(a, times.min()), max(b, times.max()))
    training, common, testing = {}, {}, {}
    for s, runs in data.items():
        for r, mods in runs.items():
            a, b = domains[r]
            for m, ts in mods.items():
                key = (s, r, m)
                lo, hi = support[m]
                eligible = (ts.times - hi >= a - 1e-8) & (ts.times - lo <= b + 1e-8)
                start, end = blocks[m]
                hidden = (ts.times >= start) & (ts.times < end)
                reserved = np.logical_or.reduce(
                    [
                        (ts.times >= BLOCK_FOLDS[f][m][0]) & (ts.times < BLOCK_FOLDS[f][m][1])
                        for f in excluded_folds
                    ]
                )
                training[key] = ts.mask & ~reserved[:, None]
                common[key] = training[key] & eligible[:, None]
                testing[key] = ts.mask & hidden[:, None] & eligible[:, None]
                assert not np.any(common[key] & testing[key])
                assert testing[key].sum() > 0
    scaler = TrainingStandardizer.fit(with_masks(data, common))
    standardized = scaler.transform(data)
    # Full-clip z-scoring in the existing cache is a per-feature affine map.
    # A second training-only standardization cancels that map. Numerically
    # verify affine invariance and insensitivity to held-out payload changes.
    affine, poisoned = {}, {}
    for s, runs in data.items():
        for r, mods in runs.items():
            for m, ts in mods.items():
                key = (s, r, m)
                affine[key] = ts.values * 2.3 + 7.0
                poisoned[key] = np.where(ts.mask & ~training[key], ts.values + 10000, ts.values)
    affine_data = with_masks(data, common, affine)
    affine_scale = TrainingStandardizer.fit(affine_data)
    affine_z = affine_scale.transform(affine_data)
    poison_scale = TrainingStandardizer.fit(with_masks(data, common, poisoned))
    for (s, m), stats in scaler.statistics.items():
        for field in ("mean", "scale"):
            np.testing.assert_array_equal(stats[field], poison_scale.statistics[s, m][field])
        for r, mods in standardized[s].items():
            np.testing.assert_allclose(
                affine_z[s][r][m].values, mods[m].values, atol=1e-11, rtol=1e-11
            )
    return dict(
        training=with_masks(standardized, training),
        common=with_masks(standardized, common),
        testing=with_masks(standardized, testing),
        scaler=scaler,
        support=support,
        domains=domains,
        blocks=blocks,
    )


def prepare(
    split,
    candidate,
    algebra="state_space",
    order=192,
    *,
    features=1,
    research_covariance_tolerance=None,
):
    model = make_model(split["training"], candidate, algebra, order, features=features)
    if research_covariance_tolerance is not None:
        # Explicit approximation study only; production qualification is unchanged.
        model.set_params(covariance_tolerance=research_covariance_tolerance)
    adapter, full = _prepare(model, split["training"])
    systems, _ = adapter._systems(split["common"], adapter.domains_)
    base = BayesianProblem(
        adapter,
        model.priors,
        systems=systems,
        anchor=full.anchor,
        reference_modality="brain",
        linear_algebra=algebra,
        response_quadrature_order=None if algebra == "state_space" else order,
    )
    return model, CanonicalProblem(base, candidate)


class CanonicalProblem:
    """MAP in common physical FWHM/peak coordinates, with matched proper priors."""

    def __init__(self, base, candidate):
        self.base, self.candidate = base, candidate
        self.factor = width_factor(candidate)
        self.shape = None if candidate == "gaussian" else int(candidate.removeprefix("gamma"))
        self.names = list(base.names)
        self.parameter_priors = list(base.parameter_priors)
        self.response_indices = {}
        for m in ("face", "rating"):
            width = base.indices[("filter", m, "width" if self.shape is None else "scale")]
            lag = base.indices[("filter", m, "lag")]
            self.names[width], self.names[lag] = ("filter", m, "fwhm"), ("filter", m, "peak")
            self.parameter_priors[width] = Prior.lognormal(np.log(3.4), 0.7).bounded(
                *WIDTH_BOUNDS[m]
            )
            self.parameter_priors[lag] = Prior.normal(2.0, 3.0).bounded(*PEAK_BOUNDS[m])
            self.response_indices[m] = (width, lag)
        self.indices = {n: i for i, n in enumerate(self.names)}
        self.bounds = [(p.lower, p.upper) for p in self.parameter_priors]
        self.initial = np.array([p.ppf(0.5) for p in self.parameter_priors])
        self.prior_groups = [
            (i, p, p.distribution()) for i, p in _group_priors(self.parameter_priors)
        ]
        self._vg = jax.jit(jax.value_and_grad(self.objective))

    def __getattr__(self, key):
        return getattr(self.base, key)

    def native(self, x):
        result = jnp.asarray(x)
        for width, peak in self.response_indices.values():
            scale = x[width] / self.factor
            lag = x[peak] if self.shape is None else x[peak] - (self.shape - 1) * scale
            result = result.at[width].set(scale).at[peak].set(lag)
        return result

    def objective(self, x):
        density = 0.0
        for indices, _, distribution in self.prior_groups:
            density += jnp.sum(distribution.log_prob(x[indices]))
        return self.base.nll(self.native(x)) - density

    def value_gradient(self, x):
        value, gradient = self._vg(jnp.asarray(x))
        return float(value), np.asarray(gradient)
