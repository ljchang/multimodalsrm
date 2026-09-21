"""Estimator-independent validation, fixed support and native data preparation.

These internal helpers contain no fitting, covariance or inference backend.
The reference estimators and Bayesian adapter share the same preparation rules;
configuration metadata is retained for compatibility with existing consumers.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator

from .data import normalize_data
from .kernels import DoubleGamma, Gaussian, Identity, Response


@dataclass
class NativeBlock:
    subject: object
    run: object
    modality: object
    times: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    valid: np.ndarray


@dataclass
class ObservationSystem:
    times: np.ndarray
    keys: list
    values: np.ndarray

    def rows(self, loadings):
        return np.array([loadings[s][m][f] for s, m, f in self.keys])


class ObservationPreparation(BaseEstimator):
    """Common options, response validation and support-eligible preprocessing."""

    def __init__(
        self,
        features=3,
        *,
        latent_dt,
        responses=None,
        length_scale=3.0,
        learn_length_scale=False,
        length_scale_bounds=(0.2, 30.0),
        noise_variance=0.1,
        learn_noise=True,
        noise_bounds=(1e-5, 10.0),
        loading_ridge=1e-3,
        max_iter=100,
        tol=1e-6,
        n_init=3,
        random_state=None,
        standardize=True,
        max_latent_size=1500,
    ):
        self.features = features
        self.latent_dt = latent_dt
        self.responses = responses
        self.length_scale = length_scale
        self.learn_length_scale = learn_length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_variance = noise_variance
        self.learn_noise = learn_noise
        self.noise_bounds = noise_bounds
        self.loading_ridge = loading_ridge
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.standardize = standardize
        self.max_latent_size = max_latent_size

    def _validate(self, data):
        for name in ("features", "max_iter", "n_init", "max_latent_size"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("latent_dt", "length_scale", "noise_variance", "tol"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        if not np.isfinite(self.loading_ridge) or self.loading_ridge < 0:
            raise ValueError("loading_ridge must be nonnegative finite")
        for name in ("learn_noise", "learn_length_scale", "standardize"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be Boolean")
        for name, initial, active in (
            ("noise_bounds", self.noise_variance, self.learn_noise),
            ("length_scale_bounds", self.length_scale, self.learn_length_scale),
        ):
            bounds = np.asarray(getattr(self, name), float)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not 0 < bounds[0] < bounds[1]
            ):
                raise ValueError(f"{name} must be two increasing positive finite bounds")
            if active and not bounds[0] <= initial <= bounds[1]:
                raise ValueError(f"initial value lies outside {name}")
        self.subjects_ = list(data)
        self.modalities_ = list(
            dict.fromkeys(m for runs in data.values() for mods in runs.values() for m in mods)
        )
        if self.responses is not None and not isinstance(self.responses, Mapping):
            raise ValueError("responses must map modality names to Response objects")
        supplied = self.responses or {}
        if not set(supplied) <= set(self.modalities_):
            raise ValueError("responses contain unobserved modalities")
        self.responses_ = {
            m: copy.deepcopy(
                supplied.get(m, Response(Identity(), pooling="shared", estimate=False))
            )
            for m in self.modalities_
        }
        for response in self.responses_.values():
            if not isinstance(response, Response) or response.pooling != "shared":
                raise ValueError("ProbabilisticMultimodalSRM requires Response(pooling='shared')")
            kernel = response.initial_kernel()
            if isinstance(kernel, DoubleGamma):
                bounds = response.parameter_bounds()
                extremes = {
                    p: bounds[p] if p in response.free_parameters else (v, v)
                    for p, v in kernel.parameters.items()
                }
                peak_max = extremes["peak_shape"][1] * extremes["peak_scale"][1]
                undershoot_min = extremes["undershoot_shape"][0] * extremes["undershoot_scale"][0]
                if peak_max >= undershoot_min:
                    raise ValueError(
                        "DoubleGamma bounds must preserve peak/undershoot mean ordering throughout the parameter box; narrow bounds or use Response.lag_only(..., pooling='shared')"
                    )

    def _support(self, times, modality, domain):
        lo, hi = self.responses_[modality].support_envelope()
        a, b = domain
        tolerance = 1e-10 * max(1.0, abs(a), abs(b))
        return (
            (times >= a - tolerance)
            & (times <= b + tolerance)
            & (times - hi >= a - tolerance)
            & (times - lo <= b + tolerance)
        )

    def _prepare(self, data):
        data = normalize_data(data)
        self._validate(data)
        self.grids_, self.domains_ = self._grids(data)
        # Learn preprocessing only from entries eligible for the fixed envelope.
        self.preprocessing_ = {}
        for s, runs in data.items():
            self.preprocessing_[s] = {}
            for m in dict.fromkeys(m for mods in runs.values() for m in mods):
                series = [(r, mods[m]) for r, mods in runs.items() if m in mods]
                values = np.concatenate([ts.values for _, ts in series])
                mask = np.concatenate(
                    [
                        ts.mask & self._support(ts.times, m, self.domains_[r])[:, None]
                        for r, ts in series
                    ]
                )
                count = mask.sum(axis=0)
                if np.any(count == 0):
                    raise ValueError(f"no support-eligible observations for a feature of {s}/{m}")
                mean = np.where(mask, values, 0).sum(axis=0) / count
                scale = np.sqrt(np.where(mask, (values - mean) ** 2, 0).sum(axis=0) / count)
                constant = scale <= np.finfo(float).eps
                scale[constant] = 1.0
                if not self.standardize:
                    mean, scale = np.zeros_like(mean), np.ones_like(scale)
                self.preprocessing_[s][m] = dict(mean=mean, scale=scale, constant_features=constant)
        self._training_data = data
        self.configuration_ = dict(
            likelihood="independent_gaussian_observations",
            noise_units="preprocessed_variance",
            latent_prior="unit_variance_independent_matern32_factors",
            approximation="GP_on_grid_with_piecewise_linear_convolution",
            uncertainty="conditional_on_fitted_parameters",
            latent_pooling="shared",
            kernel_pooling="shared",
            objective="negative_marginal_log_likelihood_plus_parameter_penalties",
            prior_jitter=1e-8,
            latent_dt=self.latent_dt,
            mean="jointly_fitted_unpenalized_offsets_in_preprocessed_units",
            restart_selection="minimum_finite_training_objective",
            response_metadata={m: r.metadata for m, r in self.responses_.items()},
            timing="relative unless independently anchored; fixed priors do not establish identification",
        )
        return data


class NativeObservationPreparation(ObservationPreparation):
    """Prepare observation-space systems without choosing an inference backend."""

    def __init__(
        self,
        features=3,
        *,
        latent_dt,
        responses=None,
        length_scale=3.0,
        learn_length_scale=False,
        length_scale_bounds=(0.2, 30.0),
        noise_variance=0.1,
        learn_noise=True,
        noise_bounds=(1e-5, 10.0),
        loading_ridge=1e-3,
        max_iter=100,
        tol=1e-6,
        n_init=3,
        random_state=None,
        standardize=True,
        max_latent_size=1500,
        covariance_tolerance=1e-7,
        max_observations=800,
        maxcor=100,
        gradient_tolerance=1e-5,
    ):
        super().__init__(
            features,
            latent_dt=latent_dt,
            responses=responses,
            length_scale=length_scale,
            learn_length_scale=learn_length_scale,
            length_scale_bounds=length_scale_bounds,
            noise_variance=noise_variance,
            learn_noise=learn_noise,
            noise_bounds=noise_bounds,
            loading_ridge=loading_ridge,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
            standardize=standardize,
            max_latent_size=max_latent_size,
        )
        self.covariance_tolerance = covariance_tolerance
        self.max_observations = max_observations
        self.maxcor = maxcor
        self.gradient_tolerance = gradient_tolerance

    def _validate(self, data):
        super()._validate(data)
        for m, kernel in getattr(self, "_known_kernels", {}).items():
            if m in self.responses_ and m not in (self.responses or {}):
                self.responses_[m] = Response(kernel, pooling="shared", estimate=False)
        if self.learn_length_scale:
            raise ValueError("learn_length_scale is unsupported by the continuous reference")
        for name in ("max_observations", "maxcor"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("covariance_tolerance", "gradient_tolerance"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        for response in self.responses_.values():
            if type(response.initial_kernel()) not in getattr(
                self, "_response_types", (Identity, Gaussian)
            ):
                raise ValueError(
                    "continuous reference supports only Identity and Gaussian responses"
                )

    def _grids(self, data):
        # Native unions are safe bookkeeping, not GP discretization.
        times = {}
        for runs in data.values():
            for r, mods in runs.items():
                for ts in mods.values():
                    times.setdefault(r, []).append(ts.times[ts.mask.any(axis=1)])
        grids = {r: np.unique(np.concatenate(t)) for r, t in times.items()}
        domains = {r: (t[0], t[-1]) for r, t in grids.items()}
        if any(b <= a for a, b in domains.values()):
            raise ValueError("each run requires positive observed duration")
        return grids, domains

    def _prepare(self, data):
        result = super()._prepare(data)
        self.configuration_.update(
            approximation="continuous_response_covariance",
            prior_jitter=0.0,
            covariance_tolerance=self.covariance_tolerance,
            max_observations=self.max_observations,
            uncertainty="conditional_on_parameters",
        )
        self._systems(result, self.domains_)  # Guard before initialization or covariance.
        return result

    def _systems(self, data, domains):
        blocks = []
        parts = {r: [] for r in domains}
        counts = {r: 0 for r in domains}
        for s, runs in data.items():
            for r, mods in runs.items():
                for m, ts in mods.items():
                    if m not in self.preprocessing_.get(s, {}):
                        raise ValueError(f"conditioning mapping {s}/{m} unavailable")
                    stats = self.preprocessing_[s][m]
                    if ts.values.shape[1] != len(stats["mean"]):
                        raise ValueError(f"feature count differs from fitted mapping for {s}/{m}")
                    valid = self._support(ts.times, m, domains[r]) & ts.mask.any(axis=1)
                    values = (ts.values - stats["mean"]) / stats["scale"]
                    block = NativeBlock(s, r, m, ts.times, values, ts.mask, valid)
                    blocks.append(block)
                    for f in range(values.shape[1]):
                        rows = valid & ts.mask[:, f]
                        count = int(rows.sum())
                        counts[r] += count
                        if counts[r] > self.max_observations:
                            raise ValueError(
                                f"run {r} exceeds max_observations={self.max_observations}"
                            )
                        if count:
                            parts[r].append((ts.times[rows], [(s, m, f)] * count, values[rows, f]))
        systems = {}
        for r, chunks in parts.items():
            if not chunks:
                raise ValueError(f"run {r} has no support-eligible conditioning observations")
            systems[r] = ObservationSystem(
                np.concatenate([c[0] for c in chunks]),
                [key for c in chunks for key in c[1]],
                np.concatenate([c[2] for c in chunks]),
            )
        return systems, blocks
