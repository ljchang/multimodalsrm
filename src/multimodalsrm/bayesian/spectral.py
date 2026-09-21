"""Explicit finite spectral approximation; exact Gaussian coefficient integration.

The Dirichlet sine expansion follows Solin and Särkkä (2020),
https://doi.org/10.1007/s11222-019-09886-w . Gaussian responses act on the
sine functions extended to the real line. Padding limits boundary effects;
finite rank omits spectral mass. Neither error is certified by this module.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real

import numpy as np

from ._backend import runtime


@dataclass(frozen=True)
class SpectralConfig:
    """Explicit basis rank and padding (seconds on each side of each run).

    Rank/padding require numerical and posterior validation for each use.
    Increasing padding without increasing rank lowers the frequency cutoff.
    """

    rank: int
    padding: float

    def __post_init__(self):
        if (
            isinstance(self.rank, (bool, np.bool_))
            or not isinstance(self.rank, (int, np.integer))
            or self.rank < 1
        ):
            raise ValueError("spectral rank must be a positive integer")
        if (
            isinstance(self.padding, (bool, np.bool_))
            or not isinstance(self.padding, Real)
            or not np.isfinite(self.padding)
            or self.padding <= 0
        ):
            raise ValueError("spectral padding must be positive finite seconds")


def validate_config(linear_algebra, spectral):
    if linear_algebra not in ("dense", "grouped", "spectral", "state_space"):
        raise ValueError("linear_algebra must be dense, grouped, spectral or state_space")
    if linear_algebra == "spectral":
        if not isinstance(spectral, SpectralConfig):
            raise ValueError("spectral algebra requires an explicit SpectralConfig")
    elif spectral is not None:
        raise ValueError("spectral configuration requires linear_algebra='spectral'")


@dataclass(frozen=True)
class SpectralBasis:
    left: float
    right: float
    length_scale: float
    config: SpectralConfig

    @classmethod
    def prepare(cls, times, length_scale, config):
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError("spectral basis needs nonempty finite one-dimensional times")
        if not np.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("spectral length_scale must be positive and finite")
        if not isinstance(config, SpectralConfig):
            raise ValueError("spectral basis requires SpectralConfig")
        return cls(
            float(times.min() - config.padding),
            float(times.max() + config.padding),
            float(length_scale),
            config,
        )

    def features(self, times, widths, lags):
        _, jnp, _, _ = runtime()
        omega = jnp.arange(1, self.config.rank + 1) * (np.pi / (self.right - self.left))
        rate = np.sqrt(3) / self.length_scale
        amplitude = jnp.sqrt(8 * rate**3 / ((self.right - self.left) * (rate**2 + omega**2) ** 2))
        widths, lags = jnp.asarray(widths), jnp.asarray(lags)
        # Avoid an infinite derivative of sqrt(width) on the Identity branch.
        identity = widths == 0
        safe_width = jnp.where(identity, 1.0, widths)
        mass = jnp.where(
            identity,
            1.0,
            np.sqrt(2) * np.pi**0.25 * jnp.sqrt(safe_width) / np.sqrt(math.erf(6)),
        )
        envelope = mass[..., None] * jnp.exp(-0.5 * widths[..., None] ** 2 * omega**2)
        phase = (jnp.asarray(times) - lags - self.left)[..., None] * omega
        return amplitude * envelope * jnp.sin(phase)

    def metadata(self):
        return dict(
            method="dirichlet_spectral_matern32",
            rank=int(self.config.rank),
            padding_seconds=float(self.config.padding),
            domain_seconds=[self.left, self.right],
            domain_source="support_eligible_conditioning_observations",
            basis_extended_to_real_line=True,
            extrapolation_validated=False,
            frequency_cutoff_radians_per_second=float(
                self.config.rank * np.pi / (self.right - self.left)
            ),
            nominal_latent_variance=1.0,
            renormalized=False,
            error_bound=None,
            exact_response_error_bound_excludes_spectral_error=True,
            error_sources=["finite_rank", "padded_domain_boundary"],
            calibration_established=False,
        )


def factor_arrays(features, weights, residual, variance, nodes):
    """Return nll, coefficient mean and lower precision Cholesky.

    Coefficients have independent N(0,1) priors. Grouping is exact within
    this finite basis; zero loadings and singular feature covariance are valid.
    """
    jax, jnp, jsp, _ = runtime()
    precision = jax.ops.segment_sum(weights**2 / variance, nodes, num_segments=len(features))
    score = jax.ops.segment_sum(weights * residual / variance, nodes, num_segments=len(features))
    A = jnp.eye(features.shape[1]) + features.T @ (precision[:, None] * features)
    L = jnp.linalg.cholesky(A)
    mean = jsp.linalg.cho_solve((L, True), features.T @ score)
    error = residual - weights * (features @ mean)[nodes]
    quadratic = jnp.sum(error**2 / variance) + mean @ mean
    nll = 0.5 * (
        quadratic
        + jnp.log(variance).sum()
        + 2 * jnp.log(jnp.diag(L)).sum()
        + len(residual) * np.log(2 * np.pi)
    )
    return nll, mean, L


def operands(problem, x, run):
    _, jnp, _, _ = runtime()
    nodes = problem.grouped_systems[run]
    ki, gi, _ = problem._packed[run]
    weights, offsets, noise, widths, lags = problem.arrays(x)
    features = problem.spectral_bases[run].features(
        nodes.times, widths[nodes.modalities], lags[nodes.modalities]
    )
    residual = jnp.asarray(problem.systems[run].values) - offsets[ki]
    return features, weights[ki], residual, noise[gi], nodes.observation_nodes


def factor(problem, x, run):
    return factor_arrays(*operands(problem, x, run))


@lru_cache(maxsize=1)
def gaussian_score():
    """Analytic Gaussian differential on the finite feature model.

    With P=(I+F' D F)^-1 and coefficient mean m, alpha=N^-1(y-WFm).
    dL/dF = D F P - segment_sum(w*alpha) m'. The observation scores
    use the posterior functional variance diag(F P F'). This is the same
    finite likelihood; it adds no approximation to the physical gradient.
    """
    jax, jnp, jsp, _ = runtime()

    @jax.custom_jvp
    def nll(features, weights, residual, variance, nodes):
        return factor_arrays(features, weights, residual, variance, nodes)[0]

    @nll.defjvp
    def differential(primals, tangents):
        F, w, residual, variance, nodes = primals
        dF, dw, dr, dv, _ = tangents
        value, m, L = factor_arrays(*primals)
        mean = F @ m
        error = residual - w * mean[nodes]
        alpha = error / variance
        FP = jsp.linalg.cho_solve((L, True), F.T).T
        cdiag = jnp.sum(F * FP, axis=1)[nodes]
        precision = jax.ops.segment_sum(w**2 / variance, nodes, num_segments=len(F))
        score = jax.ops.segment_sum(w * alpha, nodes, num_segments=len(F))
        gF = precision[:, None] * FP - score[:, None] * m[None, :]
        gw = (w * cdiag - error * mean[nodes]) / variance
        gv = 0.5 * (1 / variance - (error**2 + w**2 * cdiag) / variance**2)
        tangent = jnp.sum(gF * dF) + gw @ dw + alpha @ dr + gv @ dv
        return value, tangent

    return nll


def nll(problem, x, run):
    return gaussian_score()(*operands(problem, x, run))
