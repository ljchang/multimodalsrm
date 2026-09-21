"""Feature-count scaling for the regularized multimodal loading penalty."""

import copy
import pickle
import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.objective import (
    ObservationBlock,
    complete_objective,
    solve_loadings,
)
from multimodalsrm.optimization import (
    balance_global_scale,
    training_gradients,
)


def _quadratic_fixture():
    grid = np.arange(4.0)
    latent = np.array([[0.4], [1.2], [-0.7], [0.3]])
    values = np.array([[1.0, -0.5], [0.3, 1.1], [-0.8, 0.2], [0.6, -1.0]])
    mask = np.ones_like(values, dtype=bool)
    block = ObservationBlock(
        "s",
        "r",
        "x",
        values,
        mask,
        grid,
        sparse.eye(len(grid), format="csr"),
        np.full_like(values, 1 / values.size),
        np.ones(len(grid), dtype=bool),
    )
    return dict(
        blocks=[block],
        grids={"r": grid},
        latents={"s": {"r": latent}},
        loadings={"s": {"x": np.array([[0.8], [-0.35]])}},
        pooling="population",
        strength=0.2,
        ridge=0.11,
        temporal=0.07,
        loading_ridge=0.4,
        loading_penalty_scaling="feature",
    )


def _duplicate_features(args):
    duplicate = copy.deepcopy(args)
    source = duplicate["blocks"][0]
    source.values = np.tile(source.values, (1, 2))
    source.mask = np.tile(source.mask, (1, 2))
    source.coefficients = np.tile(source.coefficients / 2, (1, 2))
    duplicate["loadings"]["s"]["x"] = np.tile(duplicate["loadings"]["s"]["x"], (2, 1))
    return duplicate


def test_feature_scaling_makes_objective_and_loading_solve_duplication_invariant():
    """Catches omission of feature count in either objective or normal equations."""
    base = _quadratic_fixture()
    duplicate = _duplicate_features(base)

    assert complete_objective(**base) == pytest.approx(complete_objective(**duplicate))
    fitted = solve_loadings(
        base["blocks"],
        base["latents"],
        base["loading_ridge"],
        loading_penalty_scaling="feature",
    )["s"]["x"]
    duplicated = solve_loadings(
        duplicate["blocks"],
        duplicate["latents"],
        duplicate["loading_ridge"],
        loading_penalty_scaling="feature",
    )["s"]["x"]
    np.testing.assert_allclose(duplicated, np.tile(fitted, (2, 1)))

    pair_base = complete_objective(**(base | {"loading_penalty_scaling": "pair"}))
    pair_duplicate = complete_objective(**(duplicate | {"loading_penalty_scaling": "pair"}))
    assert pair_duplicate > pair_base
    assert complete_objective(
        **{k: v for k, v in base.items() if k != "loading_penalty_scaling"}
    ) == pytest.approx(pair_base)


def test_feature_scaled_loading_and_latent_gradients_match_finite_differences():
    """Catches a feature-scaled objective paired with legacy analytic gradients."""
    args = _quadratic_fixture()
    gw, gz = training_gradients(**args)
    for kind, gradients in (("loadings", gw), ("latents", gz)):
        for subject, entries in gradients.items():
            for key, gradient in entries.items():
                for index in np.ndindex(gradient.shape):
                    plus, minus = copy.deepcopy(args), copy.deepcopy(args)
                    plus[kind][subject][key][index] += 1e-6
                    minus[kind][subject][key][index] -= 1e-6
                    numerical = (complete_objective(**plus) - complete_objective(**minus)) / 2e-6
                    assert gradient[index] == pytest.approx(numerical, abs=2e-7, rel=2e-6)


def test_feature_scaled_balance_preserves_reconstruction_and_decreases_objective():
    """Catches scale balancing that computes the legacy loading penalty."""
    args = _quadratic_fixture()
    before = complete_objective(**args)
    z, w, diagnostic = balance_global_scale(**args)
    after = complete_objective(**(args | {"latents": z, "loadings": w}))

    assert diagnostic["accepted"]
    assert after < before
    assert diagnostic["after"] == pytest.approx(after)
    block = args["blocks"][0]
    original = (block.H @ args["latents"]["s"]["r"]) @ args["loadings"]["s"]["x"].T
    balanced = (block.H @ z["s"]["r"]) @ w["s"]["x"].T
    np.testing.assert_allclose(balanced, original, atol=1e-13)


def _estimator_data():
    t = np.arange(8.0)
    data = {
        subject: {
            "r": {
                "x": TimeSeries(np.column_stack([np.sin(t / 2) + offset, np.cos(t / 3)]), t),
                "y": TimeSeries(np.cos(t / 2 + offset)[:, None], t),
            }
        }
        for subject, offset in (("a", 0.0), ("b", 0.2))
    }
    return data


def _fit_estimator(policy="feature"):
    data = _estimator_data()
    model = MultimodalSRM(
        features=1,
        latent_dt=1,
        responses={m: Response(Identity(), estimate=False) for m in ("x", "y")},
        loading_penalty_scaling=policy,
        max_iter=12,
        tol=1e-3,
        random_state=4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(data)
    return model, data


def test_fitted_and_calibrated_estimators_record_loading_scaling_policy():
    """Catches a constructor-only option that is lost in fitting or calibration."""
    model, data = _fit_estimator("feature")
    assert model.loading_penalty_scaling == "feature"
    assert model.configuration_["loading_penalty_scaling"] == "feature"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        calibrated = model.calibrate({"c": data["a"]}, reference="training")
    assert calibrated.loading_penalty_scaling == "feature"
    assert calibrated.configuration_["loading_penalty_scaling"] == "feature"


def test_calibration_of_old_model_without_policy_defaults_to_pair():
    """Catches an AttributeError when calibrating a pre-policy joblib model."""
    model, data = _fit_estimator("pair")
    del model.loading_penalty_scaling
    model.configuration_.pop("loading_penalty_scaling")
    model = pickle.loads(pickle.dumps(model))
    assert model.get_params()["loading_penalty_scaling"] == "pair"
    assert "loading_penalty_scaling" not in model.configuration_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        calibrated = model.calibrate({"c": data["a"]}, reference="training")
    assert calibrated.loading_penalty_scaling == "pair"
    assert calibrated.configuration_["loading_penalty_scaling"] == "pair"


def test_invalid_loading_penalty_scaling_is_rejected_at_public_boundaries():
    """Catches misspelled policies being silently treated as pair scaling."""
    args = _quadratic_fixture()
    with pytest.raises(ValueError, match="loading_penalty_scaling"):
        complete_objective(**(args | {"loading_penalty_scaling": "features"}))
    with pytest.raises(ValueError, match="loading_penalty_scaling"):
        solve_loadings(args["blocks"], args["latents"], args["loading_ridge"], "features")
    data = _estimator_data()
    with pytest.raises(ValueError, match="loading_penalty_scaling"):
        MultimodalSRM(
            features=1,
            latent_dt=1,
            loading_penalty_scaling="features",
            max_iter=1,
        ).fit(data)
