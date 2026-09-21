"""CPU algebra keeps exact Gaussian derivatives without cubic block products."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm.bayesian._backend import runtime

from .test_bayesian_multifactor import fixture


def equations(graph):
    """Visit nested traced functions as well as the outer density."""
    if hasattr(graph, "jaxpr"):
        graph = graph.jaxpr
    if not hasattr(graph, "eqns"):
        return
    for equation in graph.eqns:
        yield equation
        for parameter in equation.params.values():
            for child in parameter if isinstance(parameter, (tuple, list)) else (parameter,):
                yield from equations(child)


@pytest.mark.parametrize("gradient", [False, True])
def test_grouped_density_does_not_multiply_full_latent_square_matrices(gradient):
    # Reintroducing a dense K @ blockdiag(D) makes likelihood setup cubic.
    # The factorization itself remains necessary and is not forbidden here.
    jax, _, _, _ = runtime()
    problem, x, _ = fixture(features=3, algebra="grouped")
    size = len(problem.grouped_systems["train"].times) * problem.features
    function = jax.value_and_grad(problem.nll) if gradient else problem.nll
    graph = jax.make_jaxpr(function)(x)
    products = [
        equation
        for equation in equations(graph)
        if equation.primitive.name == "dot_general"
        and all(getattr(v.aval, "shape", None) == (size, size) for v in equation.invars[:2])
    ]
    assert not products, "Likelihood setup must exploit block-diagonal precision"


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize(
    "case", ["ordinary", "zero_factor", "all_zero", "tiny_loadings", "high_signal"]
)
def test_three_factor_value_gradient_and_curvature_match_dense_gaussian(case, gaussian):
    jax, jnp, _, _ = runtime()
    # Identity responses duplicate temporal rows across modalities: K is
    # singular. The Gaussian cases also exercise free response parameters.
    dense, x, _ = fixture(features=3, gaussian=gaussian)
    grouped, _, _ = fixture(features=3, algebra="grouped", gaussian=gaussian)
    for i, name in enumerate(dense.names):
        if case == "zero_factor" and name[0] == "loading" and name[-1] == 2:
            x[i] = 0.0
        elif case == "all_zero" and name[0] == "loading":
            x[i] = 0.0
        elif case == "tiny_loadings" and name[0] == "loading":
            x[i] *= 1e-150
        elif case == "high_signal" and name[0] == "loading":
            x[i] *= 5
        elif case == "high_signal" and name[0] == "noise":
            x[i] *= 0.01
    actual, gradient = grouped.value_gradient(x)
    expected, expected_gradient = dense.value_gradient(x)
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)
    assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-5)
    # Curvature is used by physical-coordinate qualification/polishing.
    direction = np.random.default_rng(621).normal(size=len(x)) * np.minimum(1, np.abs(x) + 1e-4)

    def hvp(problem):
        return jax.jit(lambda a, v: jax.jvp(jax.grad(problem.objective), (a,), (v,))[1])(
            jnp.asarray(x), jnp.asarray(direction)
        )

    assert_allclose(hvp(grouped), hvp(dense), rtol=2e-6, atol=1e-3)
