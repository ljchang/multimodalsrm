"""Qualified compact Gaussian response templates with immutable unit poles.

Pole/residue scaling preserves the finite L2 convention and permits physical
width and lag derivatives. Numerical L1 bounds exclude floating-point error.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import eigvals, lstsq
from scipy.special import erf, roots_legendre

RATIONAL_GAUSSIAN_ORDERS = (20, 24)
_SHIFT = 6.0
_NORMALIZER = np.sqrt(np.sqrt(np.pi) * erf(6.0))
_FULL_MASS = np.sqrt(2 * np.pi) / _NORMALIZER
_TRUE_MASS = _FULL_MASS * erf(6 / np.sqrt(2))


def _basis(s, poles):
    return 1 / (s[:, None] - poles[None, :])


def _paired_indices(poles):
    remaining = set(range(len(poles)))
    pairs = []
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        if abs(poles[i].imag) < 1e-7:
            pairs.append((i,))
            continue
        if not remaining:
            raise FloatingPointError("unpaired complex response pole")
        j = min(remaining, key=lambda j: abs(poles[j] - poles[i].conjugate()))
        if abs(poles[j] - poles[i].conjugate()) > 1e-4:
            raise FloatingPointError("response pole conjugacy failed")
        remaining.remove(j)
        pairs.append((i, j))
    return pairs


def _fit(order):
    frequencies = np.unique(np.r_[np.linspace(0, 14, 1001), np.geomspace(14, 1e4, 140)])
    frequencies = np.r_[-frequencies[:0:-1], frequencies]
    s = 1j * frequencies
    target = _FULL_MASS * np.exp(-0.5 * frequencies**2 - 1j * _SHIFT * frequencies)
    imaginary = np.linspace(0.3, 10, order // 2)
    poles = np.r_[-2 + 1j * imaginary, -2 - 1j * imaginary]
    for _ in range(30):
        basis = _basis(s, poles)
        design = np.column_stack((basis, -target[:, None] * basis))
        solution = lstsq(design, target, cond=1e-13)[0]
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
        raise FloatingPointError("invalid stable rational Gaussian response fit")
    return poles, residues


def qualify_impulse(poles, residues):
    """Numerical panel Cauchy bounds and an analytic infinite exponential tail."""
    endpoint = 128.0
    # The exact target is zero beyond 12; split at that discontinuity explicitly.
    edges = np.unique(np.r_[np.arange(0, endpoint + 0.125, 0.125), 12.0])
    answers = []
    for order in (32, 64):
        nodes, weights = roots_legendre(order)
        u = (edges[1:] + edges[:-1])[:, None] / 2
        u = u + np.diff(edges)[:, None] * nodes / 2
        actual = (np.exp(u.ravel()[:, None] * poles) @ residues).reshape(u.shape)
        target = np.where(u <= 12, np.exp(-0.5 * (u - _SHIFT) ** 2) / _NORMALIZER, 0)
        residual = abs(actual - target)
        energy = np.diff(edges) / 2 * (residual**2 @ weights)
        answers.append(
            dict(
                rule_order=order,
                integrated_absolute_residual=float(
                    np.sum(np.diff(edges) / 2 * (residual @ weights))
                ),
                cauchy_bound=float(np.sum(np.sqrt(np.diff(edges) * energy))),
                max_imaginary_impulse=float(abs(actual.imag).max()),
            )
        )
    tail = float(np.sum(abs(residues) * np.exp(poles.real * endpoint) / -poles.real))
    refinement = abs(answers[0]["cauchy_bound"] - answers[1]["cauchy_bound"])
    maximum = max(answer["cauchy_bound"] for answer in answers)
    if not np.isfinite(maximum) or refinement > max(1e-12, maximum * 1e-4):
        raise FloatingPointError("rational Gaussian impulse quadrature failed refinement")
    margin = 1e-12 + 10 * refinement
    return dict(
        rules=answers,
        endpoint=endpoint,
        absolute_pole_tail_bound=tail,
        numerical_l1_bound=maximum + tail + margin,
        refinement_discrepancy=refinement,
        numerical_margin=margin,
        unit_true_response_mass=float(_TRUE_MASS),
        maximum_residue=float(abs(residues).max()),
    )


@dataclass(frozen=True)
class RationalGaussianTemplate:
    order: int
    poles: np.ndarray
    residues: np.ndarray
    l1_error_bound: float
    mass_bound: float
    _details: dict
    shift: float = _SHIFT

    def metadata(self):
        return dict(
            method="stable_conjugate_pole_vector_fit",
            order=self.order,
            dimensionless_shift=self.shift,
            width_scaling="sqrt_width",
            physical_poles="unit_poles_divided_by_width",
            physical_residues="unit_residues_divided_by_sqrt_width",
            internal_delay="lag_minus_six_widths",
            unit_width_l1_error_bound=self.l1_error_bound,
            unit_width_absolute_mass_bound=self.mass_bound,
            normalization="existing_finite_support_continuous_l2",
            residual_method="panelwise_direct_squared_residual_cauchy_schwarz",
            interval_arithmetic_certificate=False,
            bound_excludes="floating_point_error_and_posterior_error",
            **deepcopy(self._details),
        )

    def impulse(self, u):
        u = np.asarray(u)
        values = (np.exp(np.maximum(u.ravel(), 0)[:, None] * self.poles) @ self.residues).real
        return np.where(u.ravel() >= 0, values, 0).reshape(u.shape)

    def transfer(self, omega, width=1.0, lag=0.0):
        s = 1j * np.asarray(omega) * width
        response = _basis(s.ravel(), self.poles) @ self.residues
        response = response.reshape(s.shape)
        return (
            np.sqrt(width) * response * np.exp(-1j * np.asarray(omega) * (lag - self.shift * width))
        )

    def transfer_derivatives(self, omega, width=1.0, lag=0.0):
        """Analytic template derivatives; no estimator parameter-learning claim."""
        omega = np.asarray(omega)
        s = 1j * omega * width
        basis = _basis(s.ravel(), self.poles)
        response = (basis @ self.residues).reshape(s.shape)
        derivative = (-(basis**2) @ self.residues).reshape(s.shape)
        phase = np.exp(-1j * omega * (lag - self.shift * width))
        return dict(
            width=phase
            * (
                response / (2 * np.sqrt(width))
                + np.sqrt(width) * 1j * omega * (derivative + self.shift * response)
            ),
            lag=-1j * omega * self.transfer(omega, width, lag),
        )


@lru_cache(maxsize=2)
def rational_template(order):
    """Cached immutable unit template, qualified using its actual fitted arrays."""
    if not isinstance(order, (int, np.integer)) or order not in RATIONAL_GAUSSIAN_ORDERS:
        raise ValueError(f"rational Gaussian order must be one of {RATIONAL_GAUSSIAN_ORDERS}")
    poles, residues = _fit(int(order))
    details = qualify_impulse(poles, residues)
    error = details["numerical_l1_bound"]
    return RationalGaussianTemplate(
        int(order),
        np.frombuffer(poles.tobytes(), dtype=poles.dtype),
        np.frombuffer(residues.tobytes(), dtype=residues.dtype),
        error,
        _TRUE_MASS + error,
        details,
    )
