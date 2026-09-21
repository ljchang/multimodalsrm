"""Proper scalar priors in physical units, shared by MAP and sampling."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import truncnorm


@dataclass(frozen=True)
class Prior:
    """Normal or lognormal prior, optionally truncated, or a finite uniform.

    For lognormal priors, ``loc`` and ``scale`` describe log(parameter).
    The density itself is with respect to the physical parameter. ``bounded``
    truncates and renormalizes; it does not add a second penalty.
    """

    family: str
    loc: float = 0.0
    scale: float = 1.0
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self):
        if self.family not in ("normal", "lognormal", "uniform"):
            raise ValueError("prior family must be normal, lognormal or uniform")
        if not np.isfinite([self.loc, self.scale]).all() or self.scale <= 0:
            raise ValueError("prior location and positive scale must be finite")
        if np.isnan([self.lower, self.upper]).any() or self.lower >= self.upper:
            raise ValueError("prior bounds must be increasing")
        if self.family == "lognormal" and self.lower < 0:
            raise ValueError("lognormal support must be nonnegative")
        if self.family == "uniform" and not np.isfinite([self.lower, self.upper]).all():
            raise ValueError("uniform bounds must be finite")

    @classmethod
    def normal(cls, mean=0.0, sd=1.0):
        return cls("normal", mean, sd)

    @classmethod
    def lognormal(cls, log_mean, log_sd):
        return cls("lognormal", log_mean, log_sd, 0.0)

    @classmethod
    def uniform(cls, lower, upper):
        return cls("uniform", lower=lower, upper=upper)

    def bounded(self, lower, upper):
        return replace(self, lower=max(self.lower, lower), upper=min(self.upper, upper))

    def _normal_bounds(self):
        if self.family == "lognormal":
            lo = np.log(self.lower) if self.lower > 0 else -np.inf
            return lo, np.log(self.upper)
        return self.lower, self.upper

    def ppf(self, probability):
        """Prior quantiles for independent, reproducible initial designs."""
        p = np.asarray(probability, float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must lie between zero and one")
        if self.family == "uniform":
            return self.lower + (self.upper - self.lower) * p
        lo, hi = self._normal_bounds()
        value = truncnorm.ppf(
            p,
            (lo - self.loc) / self.scale,
            (hi - self.loc) / self.scale,
            loc=self.loc,
            scale=self.scale,
        )
        return np.exp(value) if self.family == "lognormal" else value

    def distribution(self):
        """NumPyro distribution; also defines the MAP log density."""
        from ._backend import runtime

        _, _, _, dist = runtime()
        if self.family == "uniform":
            return dist.Uniform(self.lower, self.upper)
        lo, hi = self._normal_bounds()
        # NumPyro 0.19's two-sided normalizer subtracts log CDFs even in the
        # right tail, where both CDFs round to one. Reflect that interval into
        # the left tail before constructing the distribution.
        reflected = lo >= self.loc
        loc = -self.loc if reflected else self.loc
        if reflected:
            lo, hi = -hi, -lo
        base = (
            dist.Normal(loc, self.scale)
            if np.isneginf(lo) and np.isposinf(hi)
            else dist.TruncatedNormal(
                loc,
                self.scale,
                low=None if np.isneginf(lo) else lo,
                high=None if np.isposinf(hi) else hi,
            )
        )
        if reflected:
            base = dist.TransformedDistribution(base, dist.transforms.AffineTransform(0.0, -1.0))
        if self.family == "lognormal":
            base = dist.TransformedDistribution(base, dist.transforms.ExpTransform())
        if not np.isfinite(float(base.log_prob(self.ppf(0.5)))):
            raise ValueError("prior normalization is outside the supported float64 regime")
        return base


@dataclass(frozen=True)
class BayesianPriors:
    """Priors in supplied data units; filter priors map modality then parameter.

    ``noise`` is a prior on variance, not on standard deviation or log variance.
    Loading and offset scales are illustrative defaults; applications should
    set all scales from their measurement units and prior predictive checks.
    """

    noise: Prior
    loading_sd: float = 1.0
    offset_sd: float = 1.0
    filters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.noise, Prior) or self.noise.lower < 0:
            raise ValueError("noise prior must have nonnegative physical variance support")
        if (
            not np.isfinite([self.loading_sd, self.offset_sd]).all()
            or min(self.loading_sd, self.offset_sd) <= 0
        ):
            raise ValueError("loading_sd and offset_sd must be positive finite")
        copied = {m: dict(parameters) for m, parameters in self.filters.items()}
        if any(
            not isinstance(p, Prior) for parameters in copied.values() for p in parameters.values()
        ):
            raise ValueError("filter priors must be Prior objects")
        object.__setattr__(self, "filters", copied)
