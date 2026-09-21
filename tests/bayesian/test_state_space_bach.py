"""Independent qualification of the compact, causal Bach response template."""

import importlib

import numpy as np
import pytest
from scipy.integrate import quad

from multimodalsrm.kernels import BachSCR


def _template(kernel, order=20):
    name = "multimodalsrm.bayesian.state_space_bach"
    assert importlib.util.find_spec(name) is not None, "Bach template module is missing"
    return importlib.import_module(name).bach_template(kernel, order=order)


def _convolved(template, kernel, t):
    p, r = template.poles, template.residues
    return (
        sum(
            np.sum(r * (np.exp(p * t) - np.exp(-rate * t)) / (p + rate)).real
            for rate in (kernel.lambda1, kernel.lambda2)
        )
        / template.finite_normalizer
    )


@pytest.mark.parametrize(
    "kernel", [BachSCR(), BachSCR(lambda2=0.2), BachSCR(lambda1=0.2, lambda2=0.2)]
)
def test_bach_preserves_independent_finite_convolution(kernel):
    template = _template(kernel)
    assert template.finite_normalizer == kernel._energy
    for t in [0.1, 1.0, 3.0, 5.0, 12.0, 40.0, 89.0]:
        exact = (
            quad(
                lambda u: (
                    np.exp(-0.5 * ((u - kernel.t0) / kernel.sigma) ** 2)
                    * (np.exp(-kernel.lambda1 * (t - u)) + np.exp(-kernel.lambda2 * (t - u)))
                ),
                0,
                t,
                epsabs=1e-12,
                points=[min(t, kernel.t0)],
            )[0]
            / kernel._energy
        )
        assert _convolved(template, kernel, t) == pytest.approx(exact, abs=2e-8)


def test_bach_reports_cutoff_error_and_default_tolerance_obstruction():
    k = BachSCR()
    template = _template(k)
    actual_tail = quad(lambda t: _convolved(template, k, t), 90, 600, epsabs=1e-11)[0]
    assert template.l1_error_bound >= actual_tail
    assert template.l1_error_bound < 0.008531
    bound = 2 * template.mass_bound * template.l1_error_bound
    assert 0.0836 < bound < 0.084
    assert bound > 1e-6
    assert template.metadata()["normalization"] == "existing_finite_support_continuous_l2"
    assert template.metadata()["interval_arithmetic_certificate"] is False


def test_bach_fast_decay_can_meet_default_tolerance():
    template = _template(BachSCR(lambda2=0.2))
    assert 2 * template.mass_bound * template.l1_error_bound < 1e-6


def test_bach_uses_causal_onset_and_immutable_physical_gaussian_arrays():
    k = BachSCR()
    template = _template(k)
    for t in [0.0, 0.1, 1.0, 3.0, 8.0]:
        actual = np.sum(template.residues * np.exp(template.poles * t)).real
        assert actual == pytest.approx(np.exp(-0.5 * ((t - k.t0) / k.sigma) ** 2), abs=2e-7)
    with pytest.raises(ValueError):
        template.poles[0] = 0
    with pytest.raises(ValueError):
        template.residues.setflags(write=True)
    shifted = _template(BachSCR(lag=2))
    np.testing.assert_array_equal(template.poles, shifted.poles)
    np.testing.assert_array_equal(template.residues, shifted.residues)


def test_bach_rejects_unqualified_compact_fit_domain():
    with pytest.raises(ValueError, match="order"):
        _template(BachSCR(), order=3)
    with pytest.raises(ValueError, match="ratio|fit"):
        _template(BachSCR(t0=30, sigma=0.1))
