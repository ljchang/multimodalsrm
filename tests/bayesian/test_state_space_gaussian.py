"""Independent impulse and transfer checks for a Gaussian response template."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import eval_laguerre, roots_legendre

from multimodalsrm import Gaussian


def template(order):
    from multimodalsrm.bayesian.state_space_gaussian import (
        gaussian_template,
    )

    return gaussian_template(order)


def independent_impulse(result, u):
    """SciPy polynomial evaluation, independent of the production recurrence."""
    u = np.asarray(u)
    values = eval_laguerre(np.arange(result.order)[:, None], 2 * result.rate * u.ravel()[None, :])
    return (
        np.sqrt(2 * result.rate) * np.exp(-result.rate * u.ravel()) * (result.coefficients @ values)
    ).reshape(u.shape)


@pytest.mark.parametrize("order, ceiling", [(40, 2e-6), (48, 4e-8), (56, 3e-9), (64, 2e-9)])
def test_direct_finite_support_impulse_residual_is_bounded(order, ceiling):
    result = template(order)
    target = Gaussian(width=1.0, lag=result.shift)
    # Split exactly at the finite-support endpoint to avoid integrating across
    # its jump. This evaluates |actual residual|, not an energy difference.
    nodes, weights = roots_legendre(48)
    end = result.metadata()["residual_integration_endpoint"]
    edges = np.unique(np.r_[np.linspace(0, end, 1501), 12.0])
    u = (edges[:-1, None] + edges[1:, None]) / 2
    u = u + np.diff(edges)[:, None] * nodes / 2
    residual = np.abs(independent_impulse(result, u) - target(u))
    error = np.sum(np.diff(edges) / 2 * (residual @ weights))
    assert error < result.l1_error_bound < ceiling
    assert result.mass_bound >= quad(lambda x: float(target(x)), 0, 12)[0] + error


@pytest.mark.parametrize("order", [40, 48, 56, 64])
@pytest.mark.parametrize("width", [0.03, 1.0, 9.0])
def test_physical_width_and_center_preserve_the_transfer_error_bound(order, width):
    result = template(order)
    lag = -0.7
    kernel = Gaussian(width=width, lag=lag)
    rate = result.rate / width
    frequencies = np.array([0, 0.03, 0.3, 1.0, 5.0, 12.0]) / width
    s = 1j * frequencies
    z = (s - rate) / (s + rate)
    actual = (
        np.polynomial.polynomial.polyval(z, result.coefficients)
        * np.sqrt(2 * rate)
        / (s + rate)
        * np.exp(-s * (lag - result.shift * width))
    )
    expected = np.array(
        [
            quad(
                lambda t: float(kernel(t)) * np.cos(w * t),
                *kernel.support,
                epsabs=1e-12,
            )[0]
            - 1j
            * quad(
                lambda t: float(kernel(t)) * np.sin(w * t),
                *kernel.support,
                epsabs=1e-12,
            )[0]
            for w in frequencies
        ]
    )
    assert np.max(abs(actual - expected)) < np.sqrt(width) * result.l1_error_bound


def test_squared_residual_refinement_and_declared_numerical_qualification():
    result = template(48)
    details = result.metadata()
    end = details["residual_integration_endpoint"]
    edges = np.linspace(0, end, int(np.ceil(end / details["maximum_panel_width"])) + 1)
    nodes, weights = roots_legendre(128)
    u = (edges[:-1, None] + edges[1:, None]) / 2
    u = u + np.diff(edges)[:, None] * nodes / 2
    # Compare to the smooth causal Gaussian; the omitted upper response tail
    # has its own analytic allowance in the template.
    g = np.exp(-0.5 * (u - 6) ** 2) / np.pi**0.25
    residual = independent_impulse(result, u) - g
    panel_energy = np.diff(edges) / 2 * (residual**2 @ weights)
    independently_refined = np.sum(np.sqrt(np.diff(edges) * panel_energy))
    assert_allclose(details["residual_cauchy_bound"], independently_refined, atol=2e-13, rtol=0)
    assert details["restored_gaussian_tail_bound"] == pytest.approx(1.85753984585352e-9)
    assert details["laguerre_tail_bound"] < 1e-40
    assert details["residual_refinement_discrepancy"] < 1e-12
    assert details["interval_arithmetic_certificate"] is False
    assert "floating_point" in details["bound_excludes"]
    assert details["numerical_margin"] > 0
    assert details["width_scaling"] == "sqrt_width"
    json.dumps(details, allow_nan=False)


def test_cached_coefficients_and_metadata_cannot_be_accidentally_changed():
    result = template(48)
    assert result is template(48)
    with pytest.raises(ValueError):
        result.coefficients[0] = 0
    details = result.metadata()
    details["numerical_margin"] = 42
    assert result.metadata()["numerical_margin"] != 42


@pytest.mark.parametrize("order", [0, 47, 65, 128, 48.5, "48"])
def test_unqualified_orders_are_rejected(order):
    with pytest.raises(ValueError, match="order"):
        template(order)
