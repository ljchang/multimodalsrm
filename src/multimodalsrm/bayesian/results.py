"""Marginal Gaussian mixtures and joint conditional trajectory samples."""

import copy
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.special import logsumexp, ndtr

from ..data import readonly_array, validate_times


@dataclass(frozen=True, eq=False)
class TrajectorySamples:
    """Joint paths with axes parameter draw / conditional draw / time / factor.

    Conditional draws sharing a parameter draw are not additional independent
    posterior parameter samples. Invalid query rows contain NaN. Corresponding
    draws across runs from one sampling call share their parameter draw.
    """

    samples: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    metadata: dict

    def __post_init__(self):
        samples = np.array(self.samples, float, copy=True)
        times = validate_times(self.times)
        valid = np.asarray(self.valid)
        if valid.dtype != bool or valid.shape != times.shape:
            raise ValueError("valid must be a Boolean vector matching times")
        if samples.ndim != 4 or 0 in samples.shape or samples.shape[2] != len(times):
            raise ValueError("samples must have nonempty parameter/conditional/time/factor axes")
        if not np.isfinite(samples[:, :, valid]).all():
            raise ValueError("valid trajectory samples must be finite")
        samples[:, :, ~valid] = np.nan
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "valid", readonly_array(valid, bool))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


@dataclass(frozen=True, eq=False)
class GaussianMixtureSeries:
    """Equal-weight marginal mixtures; component shape is draw/time/feature.

    These are marginal distributions, not independent joint trajectory draws.
    A MAP result is represented by one component. Interval endpoints are actual
    mixture quantiles, not Gaussian approximations based on the mixture SD.
    """

    component_means: np.ndarray
    component_variances: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    metadata: dict

    def __post_init__(self):
        means = np.array(self.component_means, float, copy=True)
        variances = np.array(self.component_variances, float, copy=True)
        times = validate_times(self.times)
        valid = np.asarray(self.valid)
        if valid.dtype != bool or valid.shape != times.shape:
            raise ValueError("valid must be a Boolean vector matching times")
        if (
            means.ndim != 3
            or 0 in means.shape
            or means.shape[1] != len(times)
            or variances.shape != means.shape
        ):
            raise ValueError("components must have matching nonempty draw/time/feature shapes")
        if (
            not np.isfinite(means[:, valid]).all()
            or not np.isfinite(variances[:, valid]).all()
            or np.any(variances[:, valid] < 0)
        ):
            raise ValueError("valid components must be finite with nonnegative variance")
        means[:, ~valid] = np.nan
        variances[:, ~valid] = np.nan
        object.__setattr__(self, "component_means", readonly_array(means))
        object.__setattr__(self, "component_variances", readonly_array(variances))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "valid", readonly_array(valid, bool))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @cached_property
    def values(self):
        return readonly_array(self.component_means.mean(axis=0))

    @cached_property
    def within_parameter_variance(self):
        return readonly_array(self.component_variances.mean(axis=0))

    @cached_property
    def between_parameter_variance(self):
        return readonly_array(self.component_means.var(axis=0))

    @cached_property
    def variance(self):
        return readonly_array(self.within_parameter_variance + self.between_parameter_variance)

    @cached_property
    def std(self):
        return readonly_array(np.sqrt(self.variance))

    def interval(self, probability=0.95):
        if not np.isscalar(probability) or not np.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("interval probability must lie strictly between zero and one")
        means, sd = self.component_means, np.sqrt(self.component_variances)

        def quantile(p):
            lower = np.min(means - 40 * sd, axis=0)
            upper = np.max(means + 40 * sd, axis=0)
            for _ in range(80):
                mid = 0.5 * (lower + upper)
                cdf = np.where(
                    sd > 0,
                    ndtr((mid - means) / np.where(sd > 0, sd, 1.0)),
                    mid >= means,
                ).mean(axis=0)
                lower = np.where(cdf < p, mid, lower)
                upper = np.where(cdf < p, upper, mid)
            return readonly_array(0.5 * (lower + upper))

        return quantile((1 - probability) / 2), quantile((1 + probability) / 2)

    def log_density(self, values):
        """Pointwise log density, excluding invalid query rows from scoring.

        Zero-variance components are atoms, for which ordinary continuous
        density scoring is undefined; such results must not be scored here.
        """
        y = np.asarray(values, float)
        if y.shape != self.values.shape or not np.isfinite(y[self.valid]).all():
            raise ValueError("values must match the time/feature shape and be finite on valid rows")
        if np.any(self.component_variances[:, self.valid] == 0):
            raise ValueError("continuous log density requires positive component variances")
        terms = -0.5 * (
            np.log(2 * np.pi * self.component_variances)
            + (y - self.component_means) ** 2 / self.component_variances
        )
        return readonly_array(logsumexp(terms, axis=0) - np.log(len(terms)))
