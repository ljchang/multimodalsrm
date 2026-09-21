"""Independent transition values and elapsed-time derivatives for learned delays."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import DoubleGamma, Gamma, Identity, Response


@pytest.mark.parametrize("scales", [(0.7, 0.700001), (0.02, 20.0)])
def test_dynamic_transitions_and_derivatives_cover_zero_small_and_large_intervals(
    scales,
):
    from multimodalsrm.bayesian._backend import runtime
    from multimodalsrm.bayesian.state_space_delays import (
        transition_function,
    )
    from multimodalsrm.bayesian.state_space_responses import (
        ResponseStateSpace,
    )

    jax, jnp, _, _ = runtime()
    model = ResponseStateSpace.prepare(
        {
            m: Response(k, estimate=False, pooling="shared")
            for m, k in {
                "ref": Identity(),
                "a": Gamma(3, scales[0]),
                "b": DoubleGamma(3, scales[0], 5, scales[1], 0.2),
            }.items()
        },
        3.0,
        1e-4,
    )
    transition = transition_function(model)
    deltas = jnp.array([0.0, 1e-12, 1e-8, 0.01, 0.5, 3.0, 40.0, 3000.0])
    values, derivatives = jax.jit(
        lambda ds: jax.jvp(jax.vmap(transition), (ds,), (jnp.ones_like(ds),))
    )(deltas)
    expected = model.transitions(np.asarray(deltas), 1)
    assert_allclose(values, expected, atol=3e-13, rtol=2e-10)
    for i, A in enumerate(np.asarray(values[0])):
        assert_allclose(derivatives[0][i], model.generator @ A, atol=2e-12)
        ab = A @ model.driving
        assert_allclose(derivatives[1][i], np.outer(ab, ab), atol=2e-12)
    derivative = jax.jit(jax.grad(lambda t: jnp.sum(transition(t)[0]) + jnp.sum(transition(t)[1])))
    for delta in (0.1, 1.0, 10.0):
        plus = model._transition(delta + 1e-5)
        minus = model._transition(delta - 1e-5)
        finite = sum(np.sum(a - b) for a, b in zip(plus, minus)) / 2e-5
        assert_allclose(derivative(delta), finite, atol=1e-7, rtol=1e-7)
