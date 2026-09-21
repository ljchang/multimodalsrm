"""Bach state-space admission preserves its finite support and physical delay."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from multimodalsrm import BachSCR, Identity, Response


def prepare(kernel=None, tolerance=1e-6, learned_lag=False):
    pytest.importorskip("jax").config.update("jax_enable_x64", True)
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    kernel = kernel or BachSCR(lambda2=0.2, lag=0.3)
    response = (
        Response.lag_only(kernel, bounds={"lag": (-0.5, 0.8)}, pooling="shared")
        if learned_lag
        else Response(kernel, estimate=False)
    )
    return ResponseStateSpace.prepare(
        {"ref": Response(Identity(), estimate=False), "scr": response}, 1.5, tolerance
    )


def test_bach_fast_decay_is_admitted_with_bound_and_exact_latent():
    model = prepare()
    assert model.dimension == 24
    assert model.tail_bound <= 1e-6
    assert model.metadata()["responses"]["scr"]["family"] == "BachSCR"
    assert model.metadata()["responses"]["scr"]["bach_template"]["response_support_seconds"] == 90
    assert_allclose(
        model.generator + model.generator.T + np.outer(model.driving, model.driving),
        0,
        atol=1e-12,
    )
    assert_allclose(model.outputs[-1, 2:], 0, atol=1e-12)


def test_canonical_bach_rejects_unattainable_tolerance_and_requires_explicit_admission():
    with pytest.raises(ValueError, match="90|BachSCR|bound"):
        prepare(BachSCR(), tolerance=1e-6)
    model = prepare(BachSCR(), tolerance=0.085)
    assert 0.083 < model.tail_bound < 0.085
    assert model.metadata()["responses"]["scr"]["bach_template"]["restored_tail"]


def test_bach_transfer_matches_finite_kernel_and_physical_lag_derivative():
    from multimodalsrm.bayesian._backend import runtime

    jax, jnp, _, _ = runtime()
    kernel = BachSCR(lambda2=0.2, lag=0.3)
    model = prepare(kernel, learned_lag=True)
    index = {("filter", "scr", "lag"): 0}
    frequency = 0.7

    def value(lag):
        F, b, C, lags = model.realize(jnp.array([lag]), index)
        return (
            C[1]
            @ jnp.linalg.solve(1j * frequency * jnp.eye(model.dimension) - F, b)
            * jnp.exp(-1j * frequency * lags[1])
        )

    actual = jax.jit(value)(0.3)
    transfer = (
        quad(
            lambda t: float(kernel(t)) * np.cos(frequency * t),
            *kernel.support,
            epsabs=1e-12,
        )[0]
        - 1j
        * quad(
            lambda t: float(kernel(t)) * np.sin(frequency * t),
            *kernel.support,
            epsabs=1e-12,
        )[0]
    )
    rate = np.sqrt(3) / 1.5
    expected = transfer * np.sqrt(4 * rate**3) / (rate + 1j * frequency) ** 2
    assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    derivative = jax.jacfwd(value)(0.3)
    assert_allclose(derivative, -1j * frequency * expected, atol=1e-7, rtol=1e-7)
    assert_allclose(model.lags, [0, 0.3, 0], atol=1e-12)


def test_bach_shape_learning_is_explicitly_rejected():
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    with pytest.raises(ValueError, match="BachSCR.*fixed|fixed.*BachSCR"):
        ResponseStateSpace.prepare(
            {"scr": Response(BachSCR(), pooling="shared", fixed={"t0": 3.0745})},
            1.5,
            0.1,
        )
