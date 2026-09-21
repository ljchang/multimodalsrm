"""Numerically qualified Laguerre templates for the Gaussian *response*.

The latent covariance is unchanged.  For width ``w``, use bank poles
``rate / w``, the returned coefficients without rescaling, and the internal
delay ``lag - shift * w``.  The response L1 error and mass bounds both scale
by ``sqrt(w)``.  Thus one fixed order covers an entire declared width range.

The continuous Cauchy--Schwarz and tail inequalities below are analytic.
Their finite-interval squared residuals are evaluated numerically with rule
refinement, rather than interval arithmetic.  The resulting bound is a
conservative numerical qualification; it excludes floating-point error and
posterior approximation error.  Small residuals are evaluated directly, never
by subtracting almost equal total and projected energies.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import erf, erfc, gammaincc, gammaln, logsumexp, roots_legendre

GAUSSIAN_ORDERS = (40, 48, 56, 64)
_RATES = {40: 5.0, 48: 5.0, 56: 6.0, 64: 6.0}
_SHIFT = 6.0
_NORMALIZER = np.sqrt(np.sqrt(np.pi) * erf(6.0))
_TRUE_MASS = np.sqrt(2 * np.pi) * erf(6 / np.sqrt(2)) / _NORMALIZER
_RESTORED_TAIL = np.sqrt(np.pi / 2) * erfc(6 / np.sqrt(2)) / _NORMALIZER
_MAX_PANEL_WIDTH = 0.25
_NUMERICAL_MARGIN = 1e-12


def _laguerre_impulses(u, rate, order):
    """Evaluate exp(-rate*u) L_j(2*rate*u) without large polynomials."""
    u = np.asarray(u)
    values = np.empty((order, u.size))
    x = 2 * rate * u.ravel()
    values[0] = np.sqrt(2 * rate) * np.exp(-x / 2)
    values[1] = (1 - x) * values[0]
    for j in range(1, order - 1):
        values[j + 1] = ((2 * j + 1 - x) * values[j] - j * values[j - 1]) / (j + 1)
    return values


def _project(order, rate, rule_order):
    nodes, weights = roots_legendre(rule_order)
    u = 9 * (nodes + 1)
    target = np.exp(-0.5 * (u - _SHIFT) ** 2) / _NORMALIZER
    return _laguerre_impulses(u, rate, order) @ (9 * weights * target)


def _residual_cauchy_bound(coefficients, rate, endpoint, rule_order):
    nodes, weights = roots_legendre(rule_order)
    edges = np.linspace(0, endpoint, int(np.ceil(endpoint / _MAX_PANEL_WIDTH)) + 1)
    lengths = np.diff(edges)
    u = (edges[:-1, None] + edges[1:, None]) / 2 + lengths[:, None] * nodes / 2
    target = np.exp(-0.5 * (u - _SHIFT) ** 2) / _NORMALIZER
    approximate = coefficients @ _laguerre_impulses(u, rate, len(coefficients))
    residual = approximate.reshape(u.shape) - target
    # On each panel I, integral_I |r| <= sqrt(|I| * integral_I r**2).
    energy = lengths / 2 * (residual**2 @ weights)
    return float(np.sum(np.sqrt(lengths * energy)))


def _laguerre_tail_bound(coefficients, rate, endpoint):
    """Bound the absolute polynomial term integrals beyond ``endpoint``.

    T_j <= sqrt(2/rate) sum_k binom(j,k) 2**k Q(k+1,rate*endpoint),
    where Q is the regularized upper incomplete gamma function.  Summing
    |coefficient_j| T_j bounds the complete response tail even with signed
    coefficients.  Logarithms avoid large intermediate binomial terms.
    """
    tails = []
    for j, coefficient in enumerate(coefficients):
        k = np.arange(j + 1)
        probability = gammaincc(k + 1, rate * endpoint)
        log_probability = np.full(j + 1, -np.inf)
        positive = probability > 0
        log_probability[positive] = np.log(probability[positive])
        log_terms = (
            gammaln(j + 1) - gammaln(k + 1) - gammaln(j - k + 1) + k * np.log(2) + log_probability
        )
        tails.append(abs(coefficient) * np.sqrt(2 / rate) * np.exp(logsumexp(log_terms)))
    return float(np.sum(tails))


@dataclass(frozen=True)
class GaussianTemplate:
    order: int
    rate: float
    shift: float
    coefficients: np.ndarray
    l1_error_bound: float
    mass_bound: float
    _diagnostics: tuple

    def metadata(self):
        """Return fresh, serializable numerical-qualification metadata."""
        return dict(
            method="shifted_causal_laguerre_projection",
            order=self.order,
            dimensionless_rate=self.rate,
            dimensionless_shift=self.shift,
            unit_width_l1_error_bound=self.l1_error_bound,
            unit_width_absolute_mass_bound=self.mass_bound,
            unit_width_true_absolute_mass=float(_TRUE_MASS),
            width_scaling="sqrt_width",
            normalization="existing_finite_support_continuous_l2",
            residual_method="panelwise_direct_squared_residual_cauchy_schwarz",
            quadrature_qualification="numerical_rule_refinement",
            interval_arithmetic_certificate=False,
            bound_excludes="floating_point_error_and_posterior_error",
            **dict(self._diagnostics),
        )


@lru_cache(maxsize=len(GAUSSIAN_ORDERS))
def gaussian_template(order):
    """Return an immutable unit-width template at a qualified fixed order.

    The exact target is exp(-(u-6)**2/2) / (pi**.25 sqrt(erf(6))) on
    [0,12].  Projection uses its smooth causal extension to u>=0.  Direct
    residual verification includes the actual returned coefficients, so any
    projection quadrature discrepancy is visible in that residual.
    """
    if not isinstance(order, (int, np.integer)) or order not in GAUSSIAN_ORDERS:
        raise ValueError(f"Gaussian state_space order must be one of {GAUSSIAN_ORDERS}")
    order = int(order)
    rate = _RATES[order]
    coarse_coefficients = _project(order, rate, 256)
    coefficients = _project(order, rate, 512)
    projection_discrepancy = float(np.linalg.norm(coefficients - coarse_coefficients))
    endpoint = (4 * order + 60) / rate
    coarse = _residual_cauchy_bound(coefficients, rate, endpoint, 32)
    fine = _residual_cauchy_bound(coefficients, rate, endpoint, 64)
    refinement = abs(fine - coarse)
    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(fine)
        or projection_discrepancy > 1e-10
        or refinement > max(1e-12, fine * 1e-5)
    ):
        raise ValueError("Gaussian state_space response failed numerical quadrature qualification")
    laguerre_tail = _laguerre_tail_bound(coefficients, rate, endpoint)
    gaussian_tail = float(np.sqrt(np.pi / 2) * erfc((endpoint - _SHIFT) / np.sqrt(2)) / _NORMALIZER)
    margin = _NUMERICAL_MARGIN + 10 * refinement
    bound = float(max(coarse, fine) + laguerre_tail + gaussian_tail + _RESTORED_TAIL + margin)
    diagnostics = dict(
        projection_interval=(0.0, 18.0),
        projection_rule_orders=(256, 512),
        projection_coefficient_refinement_discrepancy=projection_discrepancy,
        residual_rule_orders=(32, 64),
        maximum_panel_width=_MAX_PANEL_WIDTH,
        residual_integration_endpoint=float(endpoint),
        residual_cauchy_bound=max(coarse, fine),
        residual_refinement_discrepancy=refinement,
        restored_gaussian_tail_bound=float(_RESTORED_TAIL),
        laguerre_tail_bound=laguerre_tail,
        residual_gaussian_tail_bound=gaussian_tail,
        numerical_margin=float(margin),
    )
    # Immutable backing bytes prevent a caller from re-enabling writes on a
    # cached coefficient array and thereby corrupting other prepared models.
    frozen_coefficients = np.frombuffer(coefficients.tobytes(), dtype=coefficients.dtype)
    return GaussianTemplate(
        order=order,
        rate=rate,
        shift=_SHIFT,
        coefficients=frozen_coefficients,
        l1_error_bound=bound,
        mass_bound=float(_TRUE_MASS + bound),
        _diagnostics=tuple(diagnostics.items()),
    )
