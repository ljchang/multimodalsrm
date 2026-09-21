"""Fixed-shape causal Bach response templates with explicit cutoff error.

The rational template represents the raw causal Gaussian. Its convolution
with two exponential filters restores the tail beyond 90 seconds; that error
is included, and callers must enforce their covariance tolerance. The existing
finite-support continuous L2 normalizer is preserved. Bounds are numerically
qualified, not interval certificates, and exclude floating-point/posterior error.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigvals, lstsq
from scipy.special import erfcx, ndtr, roots_legendre

from ..kernels import BachSCR
from .state_space_gaussian_rational import _basis, _paired_indices

BACH_ORDERS = (20, 24)


def _immutable(values):
    array = np.asarray(values, dtype=complex)
    return np.frombuffer(array.tobytes(), dtype=complex).reshape(array.shape)


@lru_cache(maxsize=32)
def _unit_template(ratio, order):
    if not 0 < ratio <= 12:
        raise ValueError("compact Bach fit requires 0 < t0/sigma ratio <= 12; use dense/grouped")
    frequencies = np.unique(np.r_[np.linspace(0, 14, 1001), np.geomspace(14, 1e4, 140)])
    frequencies = np.r_[-frequencies[:0:-1], frequencies]
    s = 1j * frequencies
    target = np.sqrt(np.pi / 2) * np.exp(-(ratio**2) / 2) * erfcx((s - ratio) / np.sqrt(2))
    imaginary = np.linspace(0.3, 10, order // 2)
    poles = np.r_[-2 + 1j * imaginary, -2 - 1j * imaginary]
    for _ in range(30):
        basis = _basis(s, poles)
        solution = lstsq(np.column_stack((basis, -target[:, None] * basis)), target, cond=1e-13)[0]
        relocated = eigvals(np.diag(poles) - np.ones((order, 1)) * solution[order:][None, :])
        poles = -np.abs(relocated.real) + 1j * relocated.imag
    pairs = _paired_indices(poles)
    for pair in pairs:
        if len(pair) == 1:
            poles[pair[0]] = poles[pair[0]].real
        else:
            i, j = pair
            pole = (poles[i] + poles[j].conjugate()) / 2
            poles[i], poles[j] = pole, pole.conjugate()
    residues = lstsq(_basis(s, poles), target, cond=1e-13)[0]
    for pair in pairs:
        if len(pair) == 1:
            residues[pair[0]] = residues[pair[0]].real
        else:
            i, j = pair
            residue = (residues[i] + residues[j].conjugate()) / 2
            residues[i], residues[j] = residue, residue.conjugate()
    if not np.isfinite(poles).all() or not np.isfinite(residues).all() or poles.real.max() >= -1e-8:
        raise ValueError("compact Bach Gaussian fit is not stable; use dense/grouped")
    endpoint = 128.0
    edges = np.arange(0, endpoint + 0.125, 0.125)
    dt = np.diff(edges)
    bounds = []
    for n in (32, 64):
        nodes, weights = roots_legendre(n)
        u = (edges[:-1, None] + edges[1:, None]) / 2 + dt[:, None] * nodes / 2
        actual = (np.exp(u.ravel()[:, None] * poles) @ residues).reshape(u.shape)
        residual = abs(actual - np.exp(-0.5 * (u - ratio) ** 2))
        bounds.append(float(np.sum(np.sqrt(dt * dt / 2 * (residual**2 @ weights)))))
    tail = float(np.sum(abs(residues) * np.exp(poles.real * endpoint) / -poles.real))
    tail += float(np.sqrt(2 * np.pi) * ndtr(ratio - endpoint))
    refinement = abs(bounds[0] - bounds[1])
    bound = max(bounds) + tail + 1e-12 + 10 * refinement
    if not np.isfinite(bound) or refinement > max(1e-12, max(bounds) * 1e-4) or bound > 1e-3:
        raise ValueError(
            "compact Bach Gaussian fit failed residual qualification; use dense/grouped"
        )
    return _immutable(poles), _immutable(residues), bound, refinement


@dataclass(frozen=True)
class BachTemplate:
    """Physical raw-Gaussian poles/residues and normalized response bounds."""

    poles: np.ndarray
    residues: np.ndarray
    finite_normalizer: float
    mass_bound: float
    l1_error_bound: float
    order: int
    _details: dict

    def metadata(self):
        return dict(
            method="stable_causal_gaussian_vector_fit_plus_two_exponentials",
            order=self.order,
            normalization="existing_finite_support_continuous_l2",
            physical_poles="unit_poles_divided_by_sigma",
            physical_residues="raw_causal_gaussian_residues",
            supported_learning="additional_lag_only",
            response_support_seconds=90.0,
            restored_tail=True,
            numerical_l1_error_bound=self.l1_error_bound,
            absolute_mass_bound=self.mass_bound,
            finite_normalizer=self.finite_normalizer,
            interval_arithmetic_certificate=False,
            bound_excludes="floating_point_error_and_posterior_error",
            **deepcopy(self._details),
        )


def bach_template(kernel, order=20):
    """Prepare a fixed Bach shape; tolerance admission belongs to the caller.

    The bound is Young's convolution bound for the causal-Gaussian fit plus
    the positive restored response tail. Identical decay rates are allowed.
    """
    if type(kernel) is not BachSCR:
        raise ValueError("bach_template requires a BachSCR kernel")
    if order not in BACH_ORDERS:
        raise ValueError(f"compact Bach order must be one of {BACH_ORDERS}")
    ratio = kernel.t0 / kernel.sigma
    try:
        poles, residues, unit_error, refinement = _unit_template(ratio, order)
        normalizer = float(kernel._energy)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("compact Bach fit failed; use dense/grouped") from exc
    rates = (kernel.lambda1, kernel.lambda2)
    raw_mass = kernel.sigma * np.sqrt(2 * np.pi) * ndtr(ratio) * sum(1 / r for r in rates)
    gaussian_error = kernel.sigma * unit_error
    fit_error = gaussian_error * sum(1 / r for r in rates) / normalizer
    # Swap positive integrals: for u<90 the tail weight is exp(-r(90-u))/r;
    # for u>=90 it is 1/r. Integrate in standardized Gaussian coordinates.
    limit = min(90 / kernel.sigma, ratio + 12)
    raw_tail, quadrature_error = 0.0, 0.0
    for rate in rates:
        value, error = quad(
            lambda x: np.exp(-0.5 * (x - ratio) ** 2 - rate * (90 - kernel.sigma * x)),
            0,
            limit,
            points=[min(ratio, limit)],
            epsabs=1e-13,
            epsrel=1e-11,
        )
        raw_tail += kernel.sigma * (value + np.sqrt(2 * np.pi) * ndtr(ratio - limit)) / rate
        quadrature_error += kernel.sigma * error / rate
    # The Gaussian tail after the integration limit is an upper bound when
    # that limit precedes90. Otherwise it is the exact second swapped term.
    tail_bound = (raw_tail + 10 * quadrature_error) / normalizer + 1e-13
    error_bound = fit_error + tail_bound
    mass_bound = raw_mass / normalizer + fit_error
    if not np.isfinite([normalizer, error_bound, mass_bound]).all() or normalizer <= 0:
        raise ValueError("compact Bach normalization or error bound failed; use dense/grouped")
    return BachTemplate(
        _immutable(poles / kernel.sigma),
        _immutable(residues),
        normalizer,
        float(mass_bound),
        float(error_bound),
        order,
        dict(
            causal_gaussian_center_in_widths=ratio,
            causal_gaussian_l1_error_bound=float(gaussian_error),
            convolved_gaussian_l1_error_bound=float(fit_error),
            restored_tail_mass_bound=float(tail_bound),
            gaussian_quadrature_refinement=float(refinement),
            residual_method="refined_panel_cauchy_plus_analytic_tail_and_young_convolution",
        ),
    )
