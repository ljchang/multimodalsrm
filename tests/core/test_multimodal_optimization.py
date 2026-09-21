"""Regression evidence for fit selection, state consistency and diagnostics."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import Gaussian, Identity, MultimodalSRM, Response


def delay_fixture(seed=43):
    from ..reference import optimization_fixture as bench

    maps = {
        f"s{i}": {
            "brain": np.array([[1.0 + i * 0.2]]),
            "rating": np.array([[0.8 + i * 0.1]]),
        }
        for i in range(3)
    }
    kernels = {s: {"brain": Gaussian(width=0.25, lag=1), "rating": Identity()} for s in maps}
    data, _ = bench.generate(seed, maps, kernels, dt={"brain": 1, "rating": 0.5})
    return bench, data


def delay_estimator(seed=43, **kwargs):
    return MultimodalSRM(
        features=1,
        latent_dt=0.5,
        latent_pooling="shared",
        latent_strength=1,
        loading_ridge=0.001,
        latent_ridge=0.0001,
        temporal_strength=0.001,
        kernel_max_iter=12,
        responses={
            "brain": Response(
                Gaussian(width=0.25, lag=0),
                pooling="shared",
                fixed={"width": 0.25},
                bounds={"lag": (-1.5, 1.5)},
            ),
            "rating": Response(Identity(), estimate=False),
        },
        random_state=seed,
        **kwargs,
    )


def test_fit_keeps_lower_objective_candidate_despite_iteration_limit():
    """Changing selection back to converged-first recreates seed 43's failure."""
    bench, data = delay_fixture()
    model = delay_estimator(n_init=5, max_iter=300, tol=1e-6, balance_scale=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(bench.subset(data, {"train-A", "train-B"}))
    best = min(model.restart_diagnostics_, key=lambda r: r["objective"])
    assert model.objective_history_[-1] == best["objective"]
    assert model.best_restart_ == best["restart"]
    assert model.converged_ == best["converged"]
    assert model.n_iter_ == best["iterations"]
    if not model.converged_:
        assert any(issubclass(w.category, ConvergenceWarning) for w in caught)
    test = bench.subset(data, {"test-C"})
    scores, _ = bench.predicted_scores(
        model, test, test["s0"]["test-C"]["brain"], sources=("within",), window=(4, 28)
    )
    assert scores["within"]["r2"] > 0.8


def test_candidate_selection_rejects_nonfinite_and_breaks_ties_stably():
    from multimodalsrm.estimator import select_candidate

    candidates = [
        ({"objective": value, "restart": i, "converged": converged}, object())
        for i, value, converged in [
            (4, np.nan, True),
            (3, np.inf, True),
            (2, 0.01, False),
            (1, 0.01, False),
            (0, 1.0, True),
        ]
    ]
    for order in [
        candidates,
        list(reversed(candidates)),
        candidates[2:] + candidates[:2],
    ]:
        assert select_candidate(order) is candidates[3]
    assert select_candidate(candidates, policy="converged") is candidates[-1]
    assert select_candidate(candidates[2:4], policy="converged") is candidates[3]
    with pytest.raises(RuntimeError, match="finite"):
        select_candidate(candidates[:2])
    with pytest.raises(ValueError, match="policy"):
        select_candidate(candidates, policy="prediction")


def test_scale_and_final_state_diagnostics_are_exposed_without_candidate_arrays():
    bench, data = delay_fixture(seed=17)
    model = delay_estimator(seed=17, n_init=1, max_iter=50, tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(bench.subset(data, {"train-A", "train-B"}))
    assert "optimization_diagnostics_" in vars(model)
    d = model.optimization_diagnostics_
    assert d["components"]["total"] == pytest.approx(model.objective_history_[-1])
    assert d["stationarity"]["relative_scale_imbalance"] < 1e-10
    assert d["stationarity"]["scope"] == "fixed_kernel_training_quadratic"
    assert d["stopping_rule"] == "objective_decrement_and_kernel_optimizer_success"
    assert model.selection_diagnostics_["policy"] == "minimum_finite_training_objective"
    assert not any("candidates" in key for key in vars(model))
    assert np.max(np.diff(model.objective_history_)) <= 1e-10
    for record in model.restart_diagnostics_:
        assert "stationarity" in record
        assert "objective_components" in record


@pytest.mark.parametrize("lag,target,expected", [(0.5, 1, -2), (0, -1, 0), (2, 3, 0)])
def test_kernel_projection_accounts_for_shared_parameters_and_active_bounds(lag, target, expected):
    from multimodalsrm.kernel_optimization import kernel_stationarity

    kernels = {s: {"brain": Gaussian(width=0.25, lag=lag)} for s in ["a", "b"]}
    responses = {
        "brain": Response(
            Gaussian(width=0.25),
            pooling="shared",
            fixed={"width": 0.25},
            bounds={"lag": (0, 2)},
        )
    }
    d = kernel_stationarity(
        kernels,
        responses,
        lambda k: sum((mods["brain"].lag - target) ** 2 for mods in k.values()),
    )
    assert d["success"]
    assert len(d["parameters"]) == 1
    assert d["parameters"][0]["projected_gradient"] == pytest.approx(expected, abs=1e-5)
    assert d["max_abs_projected_gradient"] == pytest.approx(abs(expected), abs=1e-5)
    assert all(mods["brain"].lag == lag for mods in kernels.values())


def test_kernel_gradient_uses_log_shape_coordinates_and_reports_unavailable_values():
    from multimodalsrm.kernel_optimization import kernel_stationarity

    kernels = {s: {"brain": Gaussian(width=0.3, lag=0)} for s in ["a", "b"]}
    responses = {
        "brain": Response(
            Gaussian(width=0.3),
            pooling="shared",
            fixed={"lag": 0},
            bounds={"width": (0.1, 2)},
        )
    }
    d = kernel_stationarity(
        kernels,
        responses,
        lambda k: sum(mods["brain"].width ** 2 for mods in k.values()),
    )
    assert d["parameters"][0]["coordinate"] == "log"
    assert d["parameters"][0]["gradient"] == pytest.approx(0.36, rel=1e-6)
    invalid = kernel_stationarity(kernels, responses, lambda k: np.inf)
    assert not invalid["success"]
    assert invalid["max_abs_projected_gradient"] is None
