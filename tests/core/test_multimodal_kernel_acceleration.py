"""Kernel acceleration checked against the original observation-space loss."""

import copy

import numpy as np
import pytest

from multimodalsrm import Gamma, Gaussian, Identity, Response, TimeSeries
from multimodalsrm.kernel_optimization import (
    KernelObjective,
    kernel_penalty,
    optimize_kernels,
)
from multimodalsrm.kernels import KernelPrior, Normal
from multimodalsrm.objective import (
    PreparedBlocks,
    build_grids,
    fit_preprocessing,
    make_blocks,
)
from multimodalsrm.operators import (
    gaussian_operator_derivatives,
    observation_operator,
)


def problem(pooling="shared", exact=False, gaussian=True):
    rng = np.random.RandomState(38)
    data = {}
    for s in ("a", "b"):
        data[s] = {}
        for r in ("one", "two"):
            t = np.r_[np.arange(0.0, 32.0, 0.7), np.arange(35.0, 71.0, 0.7)]
            x = rng.normal(size=(len(t), 7))
            mask = rng.uniform(size=x.shape) > 0.13
            data[s][r] = {"m": TimeSeries(x, t, mask=mask)}
            if s == "a":
                data[s][r]["fixed"] = TimeSeries(x[::3, :2], t[::3])
    responses = {
        "m": Response(
            Gaussian(1.1, 0.2) if gaussian else Gamma(2.0, 0.3, 0.2),
            pooling=pooling,
            fixed={} if gaussian else {"shape": 2.0},
            bounds={"width": (0.4, 2.0), "lag": (-2.0, 2.0)}
            if gaussian
            else {"scale": (0.2, 0.4), "lag": (-2.0, 2.0)},
            pooling_strength=0.3,
            lag_prior=Normal(0.4, 1.2),
            prior=KernelPrior({"width": 0.9} if gaussian else {"scale": 0.25}, strength=0.7),
        ),
        "fixed": Response(Identity(), estimate=False),
    }
    kernels = {
        s: {m: v.initial_kernel() for m, v in responses.items() if m == "m" or s == "a"}
        for s in data
    }
    weights = {"m": 0.6, "fixed": 0.4}
    grids, domains = build_grids(data, 1.3)
    prep = fit_preprocessing(data, weights)

    def build(k):
        return make_blocks(data, grids, domains, prep, k, responses, weights)

    blocks = build(kernels)
    z = {s: {r: rng.normal(size=(len(g), 3)) for r, g in grids.items()} for s in data}
    w = {s: {m: rng.normal(size=(len(prep[s][m]["mean"]), 3)) for m in kernels[s]} for s in data}
    if exact:
        for b in blocks:
            b.values[:] = (b.H @ z[b.subject][b.run]) @ w[b.subject][b.modality].T

    def full(k):
        # Use original operators, all observed features, and a fixed arbitrary
        # latent/loading penalty to check the constant retained by the shortcut.
        total = 1.7 + kernel_penalty(k, responses)
        for b in blocks:
            h, _ = observation_operator(
                grids[b.run],
                b.times,
                k[b.subject][b.modality],
                responses[b.modality].support_envelope(),
            )
            residual = b.values - (h @ z[b.subject][b.run]) @ w[b.subject][b.modality].T
            total += np.sum(b.coefficients * residual**2)
        return total

    prepared = PreparedBlocks(blocks, grids, responses)
    return kernels, responses, prepared, z, w, full, build


def test_prepared_blocks_preserve_masks_normalization_and_values():
    kernels, _, prepared, _, _, _, build = problem()
    candidate = copy.deepcopy(kernels)
    for mods in candidate.values():
        mods["m"] = Gaussian(0.61, -1.4)
    for actual, expected, original in zip(prepared(candidate), build(candidate), prepared.blocks):
        for field in ("values", "mask", "valid", "coefficients", "times"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            assert getattr(actual, field) is getattr(original, field)
        np.testing.assert_array_equal(actual.H.toarray(), expected.H.toarray())
    del candidate["a"]["m"]
    with pytest.raises(ValueError, match="fitted pairs"):
        prepared(candidate)


@pytest.mark.parametrize("width,lag", [(1.1, 0.23), (0.035, -0.31), (0.5, 0.0), (3.0, 1.7)])
def test_gaussian_operator_derivatives_against_original_finite_difference(width, lag):
    grid = np.arange(-40.0, 41.0, 0.5)
    times = np.r_[-39.0, -20.0, -0.003, 0.0, 0.25, 2.713, 20.0, 39.0]
    kernel = Gaussian(width, lag)
    envelope = (-25.0, 25.0)
    h, gradients, valid = gaussian_operator_derivatives(grid, times, kernel, envelope)
    expected, expected_valid = observation_operator(grid, times, kernel, envelope)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(h.toarray(), expected.toarray())
    for name in ("width", "lag"):
        step = 2e-6 * width
        plus, _ = observation_operator(
            grid,
            times,
            kernel.with_parameters(**{name: kernel.parameters[name] + step}),
            envelope,
        )
        minus, _ = observation_operator(
            grid,
            times,
            kernel.with_parameters(**{name: kernel.parameters[name] - step}),
            envelope,
        )
        np.testing.assert_allclose(
            gradients[name].toarray(),
            (plus - minus).toarray() / (2 * step),
            atol=2e-8,
            rtol=3e-6,
        )


@pytest.mark.parametrize("pooling", ["shared", "partial", "none"])
@pytest.mark.parametrize("exact", [False, True])
def test_kernel_loss_and_gradient_against_full_feature_reconstruction(pooling, exact):
    kernels, responses, prepared, z, w, full, _ = problem(pooling, exact)
    fast = KernelObjective(kernels, responses, prepared, z, w, full)
    candidate = copy.deepcopy(kernels)
    for i, mods in enumerate(candidate.values()):
        mods["m"] = Gaussian(0.7 + 0.15 * i, -0.8 + 0.3 * i)
    for point in (kernels, candidate):
        value, gradients = fast.value_gradient(point)
        assert value == pytest.approx(full(point), rel=1e-13, abs=2e-13)
        assert fast(point) == pytest.approx(value, abs=2e-13)
        for s in point:
            for p in ("width", "lag"):
                coords = point[s]["m"].to_coordinates()
                step = 1e-5

                def evaluate(sign):
                    k = copy.deepcopy(point)
                    k[s]["m"] = point[s]["m"].from_coordinates(
                        {**coords, p: coords[p] + sign * step}
                    )
                    return full(k)

                expected = (evaluate(1) - evaluate(-1)) / (2 * step)
                assert gradients[s, "m", p] == pytest.approx(expected, abs=2e-8, rel=2e-6)


def test_analytic_optimizer_preserves_full_objective_acceptance_and_bounds():
    kernels, responses, prepared, z, w, full, _ = problem()
    fast = KernelObjective(kernels, responses, prepared, z, w, full)
    result, diagnostic = optimize_kernels(kernels, responses, fast, 40)
    assert diagnostic["gradient_method"] == "analytic"
    assert full(result) <= full(kernels)
    assert diagnostic["objective_after"] == full(result)
    np.testing.assert_array_equal(
        list(result["a"]["m"].parameters.values()),
        list(result["b"]["m"].parameters.values()),
    )
    for p, value in result["a"]["m"].parameters.items():
        lo, hi = responses["m"].parameter_bounds()[p]
        assert lo <= value <= hi


def test_invalid_gaussian_derivative_support_is_rejected():
    with pytest.raises(ValueError, match="contain"):
        gaussian_operator_derivatives(
            np.arange(100.0), np.arange(100.0), Gaussian(2.0), (-1.0, 1.0)
        )


def test_non_gaussian_candidate_loss_and_optimizer_use_numerical_fallback():
    kernels, responses, prepared, z, w, full, _ = problem(gaussian=False)
    fast = KernelObjective(kernels, responses, prepared, z, w, full)
    assert not fast.analytic_gradient
    candidate = copy.deepcopy(kernels)
    for mods in candidate.values():
        mods["m"] = Gamma(2.0, 0.34, -0.3)
    assert fast(candidate) == pytest.approx(full(candidate), rel=1e-13)
    result, diagnostic = optimize_kernels(kernels, responses, fast, 3)
    assert diagnostic["gradient_method"] == "finite_difference"
    assert full(result) <= full(kernels)
    assert all(mods["m"].shape == 2.0 for mods in result.values())


@pytest.mark.parametrize("analytic", [False, True])
def test_full_objective_rejects_misleading_candidate_loss(analytic):
    class MisleadingObjective:
        analytic_gradient = analytic

        def __call__(self, kernels):
            return (kernels["s"]["m"].lag - 1.0) ** 2

        def value_gradient(self, kernels):
            return self(kernels), {("s", "m", "lag"): 2 * (kernels["s"]["m"].lag - 1.0)}

        def full_objective(self, kernels):
            return kernels["s"]["m"].lag ** 2

    kernels = {"s": {"m": Gaussian()}}
    response = Response(Gaussian(), fixed={"width": 1.0}, pooling="shared")
    result, diagnostic = optimize_kernels(kernels, {"m": response}, MisleadingObjective(), 30)
    assert result is kernels
    assert not diagnostic["accepted"]
    assert diagnostic["objective_after"] > diagnostic["objective_before"]


def test_prepared_cache_stays_bounded_and_does_not_mutate_saved_blocks():
    kernels, _, prepared, _, _, _, build = problem()
    original = [b.H.copy() for b in prepared.blocks]
    for width in np.linspace(0.5, 1.5, 70):
        candidate = copy.deepcopy(kernels)
        for mods in candidate.values():
            mods["m"] = Gaussian(width, -0.7)
        prepared(candidate)
    assert len(prepared.cache) == 256
    for b, h, expected in zip(prepared.blocks, original, build(kernels)):
        np.testing.assert_array_equal(b.H.toarray(), h.toarray())
        np.testing.assert_array_equal(b.coefficients, expected.coefficients)
