"""Stationary response families with explicit finite supports and L2 scale.

Lag is an additional reference-shape shift in seconds. Gaussian truncates at
six standard deviations; gamma families truncate each tail at survival 1e-8.
Bach's canonical response truncates at 90 seconds, matching PsPM's duration.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import cached_property
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import gamma as gamma_distribution

from .data import readonly_array, validate_times


class Kernel:
    positive_parameters = ()
    reference = None
    normalization = "continuous_l2"

    @property
    def parameters(self):
        return {
            f.name: float(getattr(self, f.name))
            for f in fields(self)
            if f.name not in ("version", "lags", "values")
        }

    def _validate(self):
        if any(not np.isfinite(v) for v in self.parameters.values()):
            raise ValueError("kernel parameters must be finite")
        if any(self.parameters[k] <= 0 for k in self.positive_parameters):
            raise ValueError("positive kernel parameters must exceed zero")

    def with_parameters(self, **updates):
        unknown = set(updates) - self.parameters.keys()
        if unknown:
            raise ValueError(f"unknown kernel parameters: {unknown}")
        return replace(self, **updates)

    def to_coordinates(self):
        return {
            k: float(np.log(v)) if k in self.positive_parameters else v
            for k, v in self.parameters.items()
        }

    def from_coordinates(self, coordinates):
        return self.with_parameters(
            **{
                k: float(np.exp(v)) if k in self.positive_parameters else float(v)
                for k, v in coordinates.items()
            }
        )

    @cached_property
    def _energy(self):
        result = quad(
            lambda t: float(self._raw(np.asarray(t))) ** 2,
            *self.support,
            epsabs=1e-10,
            limit=200,
        )[0]
        if not np.isfinite(result) or result <= 0:
            raise ValueError("kernel must have positive finite continuous energy")
        return np.sqrt(result)

    def evaluate(self, lags):
        x = np.asarray(lags, dtype=float)
        lo, hi = self.support
        return np.where((x >= lo) & (x <= hi), self._raw(x) / self._energy, 0.0)

    __call__ = evaluate

    @property
    def metadata(self):
        return {
            "family": type(self).__name__,
            "parameters": self.parameters,
            "support": self.support,
            "normalization": self.normalization,
            "reference": self.reference,
            "coordinates": {
                k: "log" if k in self.positive_parameters else "linear" for k in self.parameters
            },
            "truncation": self.truncation,
        }


@dataclass(frozen=True)
class Identity(Kernel):
    normalization = "unit_mass_impulse"
    truncation = "analytic impulse"
    support = (0.0, 0.0)

    def evaluate(self, lags):
        raise ValueError(
            "Identity is an analytic impulse; use observation_operator, not sampled evaluation"
        )

    __call__ = evaluate


@dataclass(frozen=True)
class Gaussian(Kernel):
    width: float = 1.0
    lag: float = 0.0
    positive_parameters = ("width",)
    truncation = "six standard deviations either side of lag"

    def __post_init__(self):
        self._validate()

    @property
    def support(self):
        return (self.lag - 6 * self.width, self.lag + 6 * self.width)

    def _raw(self, x):
        return np.exp(-0.5 * ((x - self.lag) / self.width) ** 2)


@dataclass(frozen=True)
class Gamma(Kernel):
    shape: float = 3.0
    scale: float = 1.0
    lag: float = 0.0
    positive_parameters = ("shape", "scale")
    truncation = "gamma survival probability 1e-8"

    def __post_init__(self):
        self._validate()
        if self.shape < 1:
            raise ValueError("gamma shape must be at least one")

    @cached_property
    def support(self):
        return (
            self.lag,
            self.lag + float(gamma_distribution.ppf(1 - 1e-8, self.shape, scale=self.scale)),
        )

    def _raw(self, x):
        return gamma_distribution.pdf(x - self.lag, self.shape, scale=self.scale)


@dataclass(frozen=True)
class DoubleGamma(Kernel):
    peak_shape: float = 6.0
    peak_scale: float = 1.0
    undershoot_shape: float = 16.0
    undershoot_scale: float = 1.0
    undershoot_ratio: float = 1 / 6
    lag: float = 0.0
    positive_parameters = (
        "peak_shape",
        "peak_scale",
        "undershoot_shape",
        "undershoot_scale",
        "undershoot_ratio",
    )
    truncation = "each gamma survival probability 1e-8"

    def __post_init__(self):
        self._validate()
        if self.peak_shape < 1 or self.undershoot_shape < 1:
            raise ValueError("gamma shapes must be at least one")
        if self.undershoot_shape * self.undershoot_scale <= self.peak_shape * self.peak_scale:
            raise ValueError("undershoot mean must follow peak mean")
        if self.undershoot_ratio >= 1:
            raise ValueError("undershoot_ratio must lie strictly between zero and one")

    @cached_property
    def support(self):
        return (
            self.lag,
            self.lag
            + max(
                float(gamma_distribution.ppf(1 - 1e-8, s, scale=c))
                for s, c in [
                    (self.peak_shape, self.peak_scale),
                    (self.undershoot_shape, self.undershoot_scale),
                ]
            ),
        )

    def _raw(self, x):
        return gamma_distribution.pdf(
            x - self.lag, self.peak_shape, scale=self.peak_scale
        ) - self.undershoot_ratio * gamma_distribution.pdf(
            x - self.lag, self.undershoot_shape, scale=self.undershoot_scale
        )


@dataclass(frozen=True)
class BachSCR(Kernel):
    """Bach et al. (2010), canonical evoked SCR, without derivative bases.

    g(u)=exp(-(u-t0)^2/(2 sigma^2)) for u>=0; d(u)=exp(-lambda1*u)
    +exp(-lambda2*u) for u>=0; raw(t)=integral_0^t g(u)d(t-u)du.
    Closed-form integration gives a grid-independent continuous counterpart of
    PsPM pspm_bf_scrf_f.m's sampled convolution. Our L2 normalization deliberately
    replaces PsPM peak normalization. Reference: Bach DR et al., Int J
    Psychophysiol 75:349-356, doi:10.1016/j.ijpsycho.2010.01.005.
    Verified against https://raw.githubusercontent.com/bachlab/PsPM/develop/src/pspm_bf_scrf_f.m
    """

    version: str = "2010"
    t0: float = 3.0745
    sigma: float = 0.7013
    lambda1: float = 0.3176
    lambda2: float = 0.0708
    lag: float = 0.0
    positive_parameters = ("t0", "sigma", "lambda1", "lambda2")
    reference = "Bach et al. (2010), Int J Psychophysiol 75:349-356; PsPM pspm_bf_scrf_f.m"
    truncation = (
        "fixed 90-second response duration after additional lag; no tail-probability guarantee"
    )

    def __post_init__(self):
        self._validate()
        if self.version != "2010":
            raise ValueError("only BachSCR version 2010 is supported")

    @property
    def support(self):
        return (self.lag, self.lag + 90.0)

    def _raw(self, x):
        t = np.clip(x - self.lag, 0, 90)
        answer = np.zeros_like(t)
        for rate in (self.lambda1, self.lambda2):
            center = self.t0 + rate * self.sigma**2
            answer += (
                np.exp(-rate * t + rate * self.t0 + 0.5 * (rate * self.sigma) ** 2)
                * self.sigma
                * np.sqrt(2 * np.pi)
                * (ndtr((t - center) / self.sigma) - ndtr(-center / self.sigma))
            )
        return np.where(x >= self.lag, answer, 0.0)


@dataclass(frozen=True, eq=False)
class SampledKernel(Kernel):
    lags: np.ndarray
    values: np.ndarray
    truncation = "zero outside supplied piecewise-linear curve support"

    def __post_init__(self):
        lags = validate_times(self.lags)
        values = readonly_array(self.values)
        if len(lags) < 2 or values.shape != lags.shape or not np.isfinite(values).all():
            raise ValueError("sampled kernel needs at least two finite lag/value pairs")
        if not np.any(values):
            raise ValueError("sampled kernel must have nonzero energy")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "values", values)

    @property
    def support(self):
        return (float(self.lags[0]), float(self.lags[-1]))

    @cached_property
    def _energy(self):
        a, b = self.values[:-1], self.values[1:]
        return np.sqrt(np.sum(np.diff(self.lags) * (a * a + a * b + b * b) / 3))

    def _raw(self, x):
        return np.interp(x, self.lags, self.values, left=0.0, right=0.0)


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("Normal requires finite mean and positive finite sd")


@dataclass(frozen=True)
class KernelPrior:
    reference: object
    strength: float

    def __post_init__(self):
        if not np.isfinite(self.strength) or self.strength < 0:
            raise ValueError("prior strength must be finite and nonnegative")
        if not isinstance(self.reference, (Kernel, Mapping)):
            raise ValueError("prior reference must be a kernel or named parameter mapping")
        if isinstance(self.reference, Mapping):
            if not all(np.isfinite(v) for v in self.reference.values()):
                raise ValueError("prior parameter values must be finite")
            object.__setattr__(self, "reference", dict(self.reference))


@dataclass(frozen=True)
class Response:
    """Configure response sharing and which kernel parameters are learned.

    ``estimate=True`` learns parameters not listed in ``fixed``. For example,
    a Gaussian with ``fixed={"width": 0.4}`` learns only its lag, while omitting
    ``fixed`` learns both width and lag. ``estimate=False`` fixes the full
    response. ``pooling="shared"`` shares a modality's response across people.
    """

    kernel: Kernel
    pooling: str = "partial"
    estimate: bool = True
    fixed: dict | None = None
    bounds: dict | None = None
    pooling_strength: float = 1.0
    lag_prior: Normal | None = None
    prior: KernelPrior | None = None

    @classmethod
    def lag_only(
        cls,
        kernel,
        *,
        pooling="partial",
        bounds=None,
        pooling_strength=1.0,
        lag_prior=None,
    ):
        """Compatibility convenience that fixes every non-lag parameter.

        Prefer ``Response(kernel, estimate=True, fixed={...})`` in new examples
        to make the selected parameters explicit.
        """
        if not isinstance(kernel, Kernel) or "lag" not in kernel.parameters:
            raise ValueError("lag-only response requires a kernel with a lag parameter")
        fixed = {name: value for name, value in kernel.parameters.items() if name != "lag"}
        return cls(
            kernel,
            pooling=pooling,
            estimate=True,
            fixed=fixed,
            bounds=bounds,
            pooling_strength=pooling_strength,
            lag_prior=lag_prior,
        )

    def __post_init__(self):
        if not isinstance(self.kernel, Kernel):
            raise ValueError("kernel must be a supported response family")
        if self.pooling not in ("shared", "partial", "none"):
            raise ValueError("invalid kernel pooling")
        if not isinstance(self.estimate, (bool, np.bool_)):
            raise ValueError("estimate must be Boolean")
        if not np.isfinite(self.pooling_strength) or self.pooling_strength < 0:
            raise ValueError("pooling_strength must be finite nonnegative")
        object.__setattr__(self, "fixed", dict(self.fixed or {}))
        object.__setattr__(self, "bounds", dict(self.bounds or {}))
        kernel = self.initial_kernel()
        bounds = self.parameter_bounds()
        for name, (lo, hi) in bounds.items():
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("bounds must be finite and increasing")
            if name in kernel.positive_parameters and lo <= 0:
                raise ValueError("positive parameter bounds must be positive")
            if "shape" in name and lo < 1:
                raise ValueError("gamma shape bounds must be at least one")
            if name == "undershoot_ratio" and hi >= 1:
                raise ValueError("undershoot_ratio upper bound must be below one")
            if name in self.free_parameters and not lo <= kernel.parameters[name] <= hi:
                raise ValueError(f"initial {name} lies outside bounds")
        if self.lag_prior is not None and (
            not isinstance(self.lag_prior, Normal) or "lag" not in kernel.parameters
        ):
            raise ValueError("lag_prior requires Normal and a kernel lag parameter")
        if self.prior is not None:
            if not isinstance(self.prior, KernelPrior):
                raise ValueError("prior must be KernelPrior")
            if isinstance(self.prior.reference, Kernel) and type(self.prior.reference) is not type(
                kernel
            ):
                raise ValueError("prior kernel family must match")
            ref = (
                self.prior.reference.parameters
                if isinstance(self.prior.reference, Kernel)
                else self.prior.reference
            )
            kernel.with_parameters(**ref)
        if isinstance(kernel, BachSCR) and {"t0", "lag"}.issubset(self.free_parameters):
            warnings.warn(
                "BachSCR t0 and lag are both free and may be timing-confounded",
                UserWarning,
                stacklevel=2,
            )

    def initial_kernel(self):
        return self.kernel.with_parameters(**self.fixed)

    @property
    def free_parameters(self):
        return tuple(k for k in self.kernel.parameters if self.estimate and k not in self.fixed)

    def parameter_bounds(self):
        unknown = set(self.bounds) - self.kernel.parameters.keys()
        if unknown:
            raise ValueError(f"unknown bounded parameters: {unknown}")
        result = {}
        for k, v in self.initial_kernel().parameters.items():
            if k == "lag":
                bound = (v - 2.0, v + 2.0)
            elif "shape" in k:
                bound = (max(1.0, v * 0.75), v * 1.25)
            elif k == "undershoot_ratio":
                bound = (
                    max(np.nextafter(0.0, 1.0), v * 0.5),
                    min(np.nextafter(1.0, 0.0), v * 1.5, v + (1.0 - v) * 0.5),
                )
            else:
                bound = (v * 0.5, v * 1.5)
            result[k] = tuple(self.bounds.get(k, bound))
        return result

    def support_envelope(self):
        """Fixed union over bounded family extremes, independent of fit iterates."""
        kernel = self.initial_kernel()
        bounds = self.parameter_bounds()
        # Supports are monotone in each positive parameter for these families;
        # double gamma is handled componentwise before validating coupled shapes.
        if isinstance(kernel, Identity):
            return kernel.support
        if isinstance(kernel, SampledKernel):
            return kernel.support
        free = self.free_parameters
        if isinstance(kernel, DoubleGamma):
            # A rejected corner can bound a valid interior point. Retain the
            # conservative rectangle of component supports even if its maximal
            # combination violates the peak/undershoot ordering constraint.
            extremes = {
                name: bounds[name] if name in free else (value, value)
                for name, value in kernel.parameters.items()
            }
            longest_tail = max(
                float(
                    gamma_distribution.ppf(
                        1 - 1e-8,
                        extremes[prefix + "_shape"][1],
                        scale=extremes[prefix + "_scale"][1],
                    )
                )
                for prefix in ("peak", "undershoot")
            )
            return extremes["lag"][0], extremes["lag"][1] + longest_tail
        candidates = [kernel.support]
        for vals in product(*[bounds[k] for k in free]):
            updates = dict(zip(free, vals))
            try:
                candidate = kernel.with_parameters(**updates)
            except ValueError:
                continue
            candidates.append(candidate.support)
        return min(s[0] for s in candidates), max(s[1] for s in candidates)

    @property
    def metadata(self):
        timing_confounds = []
        if isinstance(self.initial_kernel(), BachSCR) and {"t0", "lag"}.issubset(
            self.free_parameters
        ):
            timing_confounds.append(
                "BachSCR t0 and lag are both free and may encode redundant timing shifts."
            )
        return {
            **self.initial_kernel().metadata,
            "pooling": self.pooling,
            "estimate": self.estimate,
            "fixed": dict(self.fixed),
            "free_parameters": self.free_parameters,
            "bounds": self.parameter_bounds(),
            "support_envelope": self.support_envelope(),
            "prior_scaling": "0.5 * strength * squared coordinate distance",
            "lag_prior": self.lag_prior,
            "prior": self.prior,
            "timing_confounds": timing_confounds,
        }
