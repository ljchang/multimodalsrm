"""Mathematical checks: finite differences and independent scalar minimization."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from multimodalsrm import Identity
from multimodalsrm.objective import (
    ObservationBlock,
    complete_objective,
    solve_latents,
    solve_loadings,
)
from multimodalsrm.operators import observation_operator


def fixture(pooling):
    rng = np.random.default_rng(28)
    grids = {"short": np.arange(4.0) * 0.7, "long": np.arange(6.0) * 0.7}
    z = {s: {r: rng.normal(size=(len(g), 2)) for r, g in grids.items()} for s in ["a", "b"]}
    if pooling == "shared":
        z["b"] = deepcopy(z["a"])
    w = {s: {m: rng.normal(size=(f, 2)) * 8 for m, f in [("x", 2), ("y", 3)]} for s in z}
    blocks = []
    for s in z:
        for r, g in grids.items():
            for m in w[s]:
                times = np.linspace(0.1, g[-1] - 0.1, 3 if m == "x" else 5)
                H, valid = observation_operator(g, times, Identity())
                values = rng.normal(size=(len(times), len(w[s][m])))
                mask = rng.random(values.shape) > 0.25
                coefficients = mask * rng.uniform(0.01, 0.1, size=values.shape)
                blocks.append(
                    ObservationBlock(s, r, m, values, mask, times, H, coefficients, valid)
                )
    return dict(
        blocks=blocks,
        grids=grids,
        latents=z,
        loadings=w,
        pooling=pooling,
        strength=0.31,
        ridge=0.23,
        temporal=0.17,
        loading_ridge=0.19,
        affinity={"a": {"b": 0.8}, "b": {"a": 0.8}},
        kernel_penalty=0.13,
    )


@pytest.mark.parametrize("pooling", ["shared", "population", "neighborhood"])
def test_components_and_scale_match_original_objective_and_scalar_optimum(pooling):
    from multimodalsrm.optimization import balance_global_scale, objective_components

    args = fixture(pooling)
    original = deepcopy(args)
    comp = objective_components(**args)
    assert comp["total"] == pytest.approx(complete_objective(**args))
    assert sum(comp["reconstruction_by_modality"].values()) == pytest.approx(comp["reconstruction"])
    for term, replacements in [
        ("loading_penalty", dict(loading_ridge=0)),
        ("latent_penalty", dict(ridge=0, temporal=0, strength=0)),
        ("kernel_penalty", dict(kernel_penalty=0)),
    ]:
        assert comp[term] == pytest.approx(
            complete_objective(**args) - complete_objective(**(args | replacements))
        )

    def scaled(logc):
        c = np.exp(logc)
        z = {s: {r: v * c for r, v in runs.items()} for s, runs in args["latents"].items()}
        w = {s: {m: v / c for m, v in mods.items()} for s, mods in args["loadings"].items()}
        return complete_objective(**(args | dict(latents=z, loadings=w)))

    optimum = minimize_scalar(scaled, bounds=(-8, 8), method="bounded", options={"xatol": 1e-12})
    z, w, d = balance_global_scale(**args)
    assert d["accepted"]
    assert d["scale"] == pytest.approx(np.exp(optimum.x), rel=2e-6)
    assert d["after"] <= d["before"]
    assert d["after"] == pytest.approx(optimum.fun)
    assert d["relative_scale_imbalance"] < 1e-12
    for b in args["blocks"]:
        np.testing.assert_allclose(
            (b.H @ z[b.subject][b.run]) @ w[b.subject][b.modality].T,
            (b.H @ args["latents"][b.subject][b.run]) @ args["loadings"][b.subject][b.modality].T,
            atol=1e-13,
        )
    for name in ["latents", "loadings"]:
        for s in args[name]:
            for k in args[name][s]:
                np.testing.assert_array_equal(args[name][s][k], original[name][s][k])
    if pooling == "shared":
        for r in args["grids"]:
            np.testing.assert_array_equal(z["a"][r], z["b"][r])


@pytest.mark.parametrize("pooling", ["shared", "population", "neighborhood"])
def test_gradients_are_derivatives_of_complete_objective(pooling):
    from multimodalsrm.optimization import stationarity_diagnostics, training_gradients

    args = fixture(pooling)
    gw, gz = training_gradients(**args)
    for kind, gradients in [("loadings", gw), ("latents", gz)]:
        for s, entries in gradients.items():
            for key, grad in entries.items():
                for index in np.ndindex(grad.shape):
                    plus, minus = deepcopy(args), deepcopy(args)
                    subjects = (
                        list(args["latents"]) if kind == "latents" and pooling == "shared" else [s]
                    )
                    for ss in subjects:
                        plus[kind][ss][key][index] += 1e-6
                        minus[kind][ss][key][index] -= 1e-6
                    numerical = (complete_objective(**plus) - complete_objective(**minus)) / 2e-6
                    assert grad[index] == pytest.approx(numerical, abs=2e-7, rel=2e-6)
    d = stationarity_diagnostics(**args)
    assert d["loading_max_abs_gradient"] > 0
    assert d["latent_max_abs_gradient"] > 0
    args["latents"], _ = solve_latents(
        args["blocks"],
        args["grids"],
        list(args["latents"]),
        args["loadings"],
        2,
        pooling,
        args["strength"],
        args["ridge"],
        args["temporal"],
        args["affinity"],
    )
    assert stationarity_diagnostics(**args)["latent_relative_residual"] < 1e-10
    args["loadings"] = solve_loadings(args["blocks"], args["latents"], args["loading_ridge"])
    assert stationarity_diagnostics(**args)["loading_relative_residual"] < 1e-10


@pytest.mark.parametrize(
    "replacement",
    [
        dict(loading_ridge=0),
        dict(ridge=0, temporal=0, strength=0),
        dict(kernel_penalty=float("inf")),
    ],
)
def test_degenerate_scale_is_nonmutating_noop(replacement):
    from multimodalsrm.optimization import balance_global_scale

    args = fixture("population") | replacement
    z, w, d = balance_global_scale(**args)
    assert not d["accepted"]
    assert d["scale"] == 1
    for name, result in [("latents", z), ("loadings", w)]:
        for s, entries in result.items():
            for k, v in entries.items():
                np.testing.assert_array_equal(v, args[name][s][k])


def test_shared_variable_count_and_zero_gradient_diagnostics():
    from multimodalsrm.optimization import stationarity_diagnostics

    args = fixture("shared")
    d = stationarity_diagnostics(**args)
    assert d["latent_variable_count"] == 20  # (4+6) time nodes, rank two
    assert d["loading_variable_count"] == 20  # two subjects, five features, rank two
    for entries in args["latents"].values():
        for v in entries.values():
            v[:] = 0
    for entries in args["loadings"].values():
        for v in entries.values():
            v[:] = 0
    d = stationarity_diagnostics(**args)
    assert d["latent_relative_residual"] == 0
    assert d["loading_relative_residual"] == 0
    assert d["relative_scale_imbalance"] == 0


def test_scale_does_not_clip_a_finite_large_optimum_or_accept_anchors():
    from multimodalsrm.optimization import balance_global_scale

    args = fixture("population")
    args["latents"] = {
        s: {r: v * 1e-20 for r, v in runs.items()} for s, runs in args["latents"].items()
    }
    args["loadings"] = {
        s: {m: v * 1e20 for m, v in mods.items()} for s, mods in args["loadings"].items()
    }
    z, w, d = balance_global_scale(**args)
    assert d["accepted"]
    assert d["scale"] > 1e19
    assert d["relative_scale_imbalance"] < 1e-12
    assert complete_objective(**(args | dict(latents=z, loadings=w))) <= complete_objective(**args)
    with pytest.raises(TypeError):
        balance_global_scale(**args, population_reference={})


def test_nonfinite_penalties_and_invalid_shared_state_are_not_scaled():
    from multimodalsrm.optimization import balance_global_scale

    args = fixture("shared")
    args["latents"]["b"]["short"][0, 0] += 1
    z, w, d = balance_global_scale(**args)
    assert not d["accepted"]
    assert d["scale"] == 1
    args = fixture("population")
    args["loadings"]["a"]["x"][0, 0] = np.inf
    z, w, d = balance_global_scale(**args)
    assert not d["accepted"]
    assert d["scale"] == 1
