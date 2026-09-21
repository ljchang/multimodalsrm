"""Frozen inference must condition new runs using the original training MAP."""

import copy
import importlib
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    SamplerConfig,
    SearchConfig,
)
from multimodalsrm.bayesian.quality import summarize_fit
from multimodalsrm.bayesian.workflow import load_results, save_results

from .test_bayesian_problem import problem_fixture


def fitted(mode="dense", baseline=None):
    p, adapter, data = problem_fixture()
    model = BayesianMultimodalSRM(
        priors=p.priors,
        anchor=p.anchor,
        responses=adapter.responses_,
        inference="map",
        random_state=413,
        linear_algebra=mode,
        run_baseline_sd=baseline,
        search=SearchConfig(starts=1, maxiter=500),
    ).fit(data)
    return model, data


def donors(data, run="new"):
    return {s: {run: copy.deepcopy(rs["train"])} for s, rs in data.items()}


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
@pytest.mark.parametrize("baseline", [None, {"ref": 0.6, "signal": 0.8}])
def test_frozen_matches_independent_gaussian_and_preserves_source(
    algebra, baseline, monkeypatch, tmp_path
):
    model, data = fitted(algebra, baseline)
    before = model.map_parameters_.copy()
    original_config = copy.deepcopy(model.configuration_)

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen conditioning must not fit parameters")

    module = importlib.import_module("multimodalsrm.bayesian.model")
    monkeypatch.setattr(module, "search", forbidden)
    monkeypatch.setattr(module, "sample", forbidden)
    supplied = donors(data)
    supplied["b"]["new"]["signal"] = object()
    result = model.condition(supplied, targets={"b": ["signal"]}, mode="frozen")
    assert set(result.problem_.systems) == {"new"}
    assert_allclose(result.map_parameters_, before, atol=0)
    assert not np.shares_memory(result.map_parameters_, model.map_parameters_)
    assert model.configuration_ == original_config
    assert set(model.problem_.systems) == {"train"}
    x = dict(zip(model.parameter_names_, before))
    w = x["loading", "a", "ref", 0]
    v = x["noise", "a", "ref"]
    offset = x["offset", "a", "ref", 0]
    t = data["a"]["train"]["ref"].times
    y = data["a"]["train"]["ref"].values[:, 0]
    q = np.array([5.0, 8.0, 11.0])

    def matern(a, b):
        d = np.sqrt(3) * np.abs(a[:, None] - b[None, :]) / 3
        return (1 + d) * np.exp(-d)

    C = w * w * matern(t, t) + np.eye(len(t)) * v
    if baseline:
        C += baseline["ref"] ** 2
    for latent in (False, True):
        loading = 1.0 if latent else x["loading", "b", "signal", 0]
        mean_offset = 0.0 if latent else x["offset", "b", "signal", 0]
        cross = loading * w * matern(q, t)
        mu = cross @ np.linalg.solve(C, y - offset) + mean_offset
        var = loading**2 - np.sum(cross * np.linalg.solve(C, cross.T).T, axis=1)
        if baseline and not latent:
            var += baseline["signal"] ** 2
        prediction = (
            result.infer_latent(times=q)["new"]
            if latent
            else result.predict(times=q)["b"]["new"]["signal"]
        )
        assert_allclose(prediction.values[:, 0], mu, atol=1e-9)
        assert_allclose(prediction.variance[:, 0], var, atol=1e-9)
        assert prediction.metadata["conditioning_mode"] == "frozen"
        assert prediction.metadata["parameter_conditioning"] == "original_training_MAP"
        assert prediction.metadata["uncertainty"] == "conditional_on_training_MAP"
    assert result.configuration_["parameter_refits"] == 0
    assert set(result.configuration_["linear_system_sizes"]) == {"new"}
    assert result.map_diagnostics_["scope"] == "original_training"
    quality = summarize_fit(
        dict(
            configuration=result.configuration_,
            map=result.map_diagnostics_,
            restarts=result.restart_diagnostics_,
            parameter_names=result.parameter_names_,
        )
    )
    assert quality["map_passed"] is None
    assert quality["training_map_passed"] == model.map_diagnostics_["meets_gradient_tolerance"]
    assert quality["diagnostic_scope"] == "original_training"
    path = tmp_path / "frozen.npz"
    save_results(path, {("b", "new", "signal"): prediction}, metadata=result.configuration_)
    restored, metadata = load_results(path)
    assert metadata["conditioning_mode"] == "frozen"
    assert restored["b", "new", "signal"].metadata["conditioning_mode"] == "frozen"
    result.map_diagnostics_["objective"] = 123
    assert model.map_diagnostics_["objective"] != 123


def test_chains_always_return_to_original_map_and_only_current_donors():
    model, data = fitted()
    first = model.condition(donors(data), targets={"b": ["signal"]}, mode="joint")
    assert np.max(np.abs(first.map_parameters_ - model.map_parameters_)) > 1e-5
    for source in (model, first):
        frozen = source.condition(donors(data, "second"), targets={"b": ["signal"]}, mode="frozen")
        again = frozen.condition(donors(data, "third"), targets={"b": ["signal"]}, mode="frozen")
        assert set(again.problem_.systems) == {"third"}
        assert_allclose(again.map_parameters_, model.map_parameters_, atol=0)
        assert_allclose(frozen.map_parameters_, model.map_parameters_, atol=0)
        assert_allclose(
            again.predict(times=[5.0, 8.0])["b"]["third"]["signal"].values,
            frozen.predict(times=[5.0, 8.0])["b"]["second"]["signal"].values,
        )


def test_mode_and_fitted_inference_validation_precede_donor_inspection():
    model, data = fitted()
    with pytest.raises(ValueError, match="mode"):
        model.condition(object(), targets={}, mode="typo")
    model.inference = "posterior"
    with pytest.raises(ValueError, match="MAP|map"):
        model.condition(object(), targets={}, mode="frozen")


def test_frozen_keeps_validation_missing_modalities_and_capacity_guards():
    model, data = fitted()
    only_a = {"a": donors(data)["a"]}
    result = model.condition(only_a, targets={"b": ["signal"]}, mode="frozen")
    assert np.isfinite(result.predict(times=[5.0])["b"]["new"]["signal"].values).all()
    with pytest.raises(ValueError, match="overlap"):
        model.condition(data, targets={"b": ["signal"]}, mode="frozen")
    with pytest.raises(ValueError, match="mapping"):
        model.condition(object(), targets={"unknown": ["signal"]}, mode="frozen")
    model.length_scale = 9
    with pytest.raises(ValueError, match="model specification"):
        model.condition(object(), targets={}, mode="frozen")


def test_posterior_fit_cannot_masquerade_as_map_by_changing_inference():
    model, data = fitted()
    model.set_params(inference="posterior", sampler=SamplerConfig(chains=1, warmup=5, draws=5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    for inference in ("posterior", "map"):
        model.inference = inference
        with pytest.raises(ValueError, match="MAP|map"):
            model.condition(object(), targets={}, mode="frozen")


def test_frozen_preserves_scalar_observation_guard():
    model, data = fitted()
    model.set_params(max_observations=20).fit(data)
    t = np.linspace(0, 20, 21)
    oversized = {"a": {"new": {"ref": TimeSeries(np.sin(t)[:, None], t)}}}
    with pytest.raises(ValueError, match="max_observations"):
        model.condition(oversized, targets={"b": ["signal"]}, mode="frozen")
