"""Raw posterior retention and independently conditioned reporting coordinates."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import Identity, TimeSeries

from ..reference.continuous_covariance import response_covariance
from .test_bayesian_multifactor import fixture, independent
from .test_bayesian_problem import api

ANCHORS = (("a", "brain", 0), ("a", "brain", 1))


@pytest.fixture(scope="module")
def posterior():
    b = api()
    times = np.arange(6.0)
    data = {
        "a": {
            "train": {"brain": TimeSeries(np.column_stack((np.sin(times), np.cos(times))), times)}
        }
    }
    model = b.BayesianMultimodalSRM(
        features=2,
        factor_anchors=ANCHORS,
        inference="posterior",
        linear_algebra="grouped",
        random_state=617,
        priors=b.BayesianPriors(noise=b.Prior.lognormal(-1.0, 0.5)),
        search=b.SearchConfig(starts=1, maxiter=80, refine_maxiter=0),
        sampler=b.SamplerConfig(
            chains=2, warmup=8, draws=8, max_tree_depth=3, mass_matrix="diagonal"
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data)
    return model, data, caught


def test_public_grouped_posterior_retains_raw_draws_and_failed_diagnostics(posterior):
    model, _, caught = posterior
    assert model.parameter_draws_.shape == (2, 8, len(model.parameter_names_))
    assert np.isfinite(model.parameter_draws_).all()
    assert np.isfinite(model.log_likelihood_draws_).all()
    assert not model.sampling_diagnostics_["passes"]
    assert any("diagnostic" in str(w.message) for w in caught)
    assert model.sample_stats_["diverging"].shape == (2, 8)
    assert model.sampling_diagnostics_["metric"]["actual"] == "diagonal"
    reported = model.reported_parameter_draws()
    assert reported.shape == model.parameter_draws_.shape
    assert not reported.flags.writeable
    assert not np.shares_memory(reported, model.parameter_draws_)
    changed = copy.deepcopy(model)
    changed.factor_anchors = (("missing", "brain", 0),) * 2
    changed.features, changed.inference = 5, "map"
    assert_array_equal(changed.reported_parameter_draws(), reported)
    assert changed.infer_latent(times=[1.0, 2.0])["train"].values.shape == (2, 2)


def physical_model(algebra):
    """Use independent physical draws, with known equivalent rotated loadings."""
    p, x, _ = fixture(algebra=algebra)
    b = api()
    model = b.BayesianMultimodalSRM(priors=p.priors, features=2, factor_anchors=ANCHORS)
    model.problem_, model.adapter_ = p, p.adapter
    model.prediction_runs_ = dict(p.adapter.domains_)
    model.targets_, model.sampling_diagnostics_ = None, {"passes": False}
    # Fitted conventions, separate from mutable estimator options.
    model._factor_anchor_keys_ = ANCHORS
    model.configuration_ = {
        "inference": "posterior",
        "features": 2,
        "uncertainty": "parameter_posterior_mixture",
        "reference_convention": p.reference_convention,
        "factor_orientation": {"method": "draw_specific_anchor_QR_positive_diagonal"},
    }
    W = independent(p, x)[1]
    R = np.array([[0.6, -0.8], [0.8, 0.6]])
    rotated = x.copy()
    rotated[: W.size] = (W @ R).ravel()
    model.parameter_draws_ = np.array([[x, rotated], [rotated, x]])
    return model


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_reporting_matches_full_gaussian_with_cross_factor_covariance(algebra):
    from multimodalsrm.bayesian.prediction import (
        result as prediction_result,
    )

    model = physical_model(algebra)
    p = model.problem_
    raw = model.parameter_draws_.copy()
    x, rotated = raw[0]
    C, W, offsets, idx, kernels = independent(p, x)
    assert_allclose(independent(p, rotated)[0], C, atol=1e-10)
    assert_allclose(p.log_prior(x), p.log_prior(rotated), atol=1e-10)
    reported = model.reported_parameter_draws()
    assert_allclose(reported[0, 0], reported[0, 1], atol=1e-10)
    assert_array_equal(reported[..., W.size :], raw[..., W.size :])
    query = np.array([4.0, 7.1, 10.0])
    system = p.systems["train"]
    cross = np.stack(
        [
            response_covariance(
                query, [t], Identity(), kernels[k[1]], p.length_scale, tolerance=1e-10
            )[:, 0, None]
            * W[i]
            for t, k, i in zip(system.times, system.keys, idx)
        ],
        axis=-1,
    )
    mean = np.einsum("tfn,n->tf", cross, np.linalg.solve(C, system.values - offsets[idx]))
    covariance = np.array([np.eye(2) - c @ np.linalg.solve(C, c.T) for c in cross])
    assert np.max(np.abs(covariance[:, 0, 1])) > 1e-3
    # Unique QR convention computed from independent named physical coordinates.
    Q, upper = np.linalg.qr(W[[p.keys.index(k) for k in ANCHORS]].T)
    Q *= np.sign(np.diag(upper))[None, :]
    expected_var = np.einsum("fi,tfg,gi->ti", Q, covariance, Q)
    wrong_var = np.diagonal(covariance, axis1=1, axis2=2) @ Q**2
    assert np.max(np.abs(wrong_var - expected_var)) > 1e-3
    result = model.infer_latent(times=query, max_draws=None)["train"]
    for m, v in zip(result.component_means, result.component_variances):
        assert_allclose(m, mean @ Q, atol=1e-8)
        assert_allclose(v, expected_var, atol=1e-8)
    assert result.metadata["sampling_diagnostics_passed"] is False
    assert result.metadata["parameter_draw_indices"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert result.metadata["fixed_parameter_coordinates"] == "raw_physical"
    observed = prediction_result(
        model,
        "train",
        query,
        raw.reshape(-1, raw.shape[-1]),
        [("a", "brain", 0)],
        include_noise=True,
    )
    raw_index = p.keys.index(("a", "brain", 0))
    raw_loading = W[raw_index]
    expected_observed_mean = mean @ raw_loading + offsets[raw_index]
    expected_observed_variance = (
        np.einsum("f,tfg,g->t", raw_loading, covariance, raw_loading)
        + x[p.indices[("noise", "a", "brain")]]
    )
    for m, v in zip(observed.component_means, observed.component_variances):
        assert_allclose(m[:, 0], expected_observed_mean, atol=1e-8)
        assert_allclose(v[:, 0], expected_observed_variance, atol=1e-8)
    # Fitted dimension, inference and anchors govern queries even after edits.
    model.features, model.factor_anchors, model.inference = (
        3,
        (("bad", "key", 0),),
        "map",
    )
    assert_array_equal(model.reported_parameter_draws(), reported)
    again = model.infer_latent(times=query, max_draws=None)["train"]
    assert_allclose(again.component_variances, result.component_variances)
    assert_array_equal(model.parameter_draws_, raw)


@pytest.mark.parametrize("value, message", [(0.0, "rank deficient"), (np.nan, "nonfinite")])
def test_bad_anchor_preserves_raw_draws_and_original_subsample_ids(value, message):
    model = physical_model("grouped")
    model.parameter_draws_ = np.repeat(model.parameter_draws_[:, :1], 5, axis=1)
    p = model.problem_
    for key in ANCHORS:
        for f in range(2):
            model.parameter_draws_[1, 4, p.indices[("loading", *key, f)]] = value
    raw = model.parameter_draws_.copy()
    with pytest.raises(ValueError, match=rf"{message}.*chain 1, draw 4"):
        model.infer_latent(times=[4.0, 7.0], max_draws=4)
    with pytest.raises(ValueError, match=rf"{message}.*chain 1, draw 4"):
        model.reported_parameter_draws()
    assert_array_equal(model.parameter_draws_, raw)


def test_multifactor_posterior_keeps_map_only_workflow_boundaries(posterior, tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data, _ = posterior
    with pytest.raises(ValueError, match="overlap|no usable donor"):
        model.condition(data, targets={"a": ["brain"]}, mode="joint")
    with pytest.raises(ValueError, match="posterior"):
        model.condition(data, targets={"a": ["brain"]}, mode="frozen")
    with pytest.raises(ValueError, match="MAP"):
        model.transform(data, times=[1.0, 2.0])
    with pytest.raises(ValueError, match="posterior"):
        model.calibrate(data)
    save_model(tmp_path / "posterior", model)
    restored, _ = load_model(tmp_path / "posterior")
    assert_array_equal(restored.parameter_draws_, model.parameter_draws_)


def test_singular_map_anchor_does_not_prevent_raw_sampling(posterior, monkeypatch):
    from multimodalsrm.bayesian import model as module

    fitted, data, _ = posterior
    best = copy.deepcopy(fitted.map_diagnostics_)
    best["parameters"] = fitted.map_parameters_.copy()
    best["parameters"][:4] = 0.0
    # Isolate a singular initialization record; real sampling was exercised above.
    monkeypatch.setattr(module, "search", lambda *args, **kwargs: (best, [best]))
    monkeypatch.setattr(
        module,
        "sample",
        lambda *args, **kwargs: (
            fitted.parameter_draws_,
            fitted.sample_stats_,
            fitted.log_likelihood_draws_,
            copy.deepcopy(fitted.sampling_diagnostics_),
        ),
    )
    model = module.BayesianMultimodalSRM(**fitted.get_params())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert_array_equal(model.parameter_draws_, fitted.parameter_draws_)
    assert np.isfinite(model.reported_parameter_draws()).all()


@pytest.mark.parametrize("features", [1, 2])
def test_reported_map_draws_keep_established_coordinates(features):
    from multimodalsrm.bayesian.multifactor import orientation

    p, x, _ = fixture(features=features)
    model = api().BayesianMultimodalSRM(priors=p.priors, features=features)
    model.problem_, model.parameter_draws_ = p, x[None, None]
    model.configuration_ = {"inference": "map"}
    expected = x.copy()
    if features > 1:
        model.configuration_["factor_orientation"] = orientation(p, x, ANCHORS)
        W = independent(p, x)[1]
        Q = np.asarray(model.configuration_["factor_orientation"]["rotation"])
        expected[: W.size] = (W @ Q).ravel()
    reported = model.reported_parameter_draws()
    assert_allclose(reported[0, 0], expected, atol=1e-12)
    assert_array_equal(model.parameter_draws_, x[None, None])
    assert not reported.flags.writeable
