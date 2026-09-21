"""Independent test projections must not share held-out observations."""

import copy
import importlib
from collections.abc import Mapping

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.workflow import (
    _json,
    load_results,
    save_results,
)

from .test_bayesian_frozen import donors, fitted
from .test_bayesian_problem import problem_fixture


@pytest.fixture(scope="module")
def trained():
    return fitted()


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
@pytest.mark.parametrize("baseline", [None, {"ref": 0.6, "signal": 0.8}])
def test_transform_matches_subject_only_gaussian(algebra, baseline):
    model, data = fitted(algebra, baseline)
    assert callable(getattr(model, "transform", None)), "independent transform is missing"
    supplied = donors(data)
    # Different native clocks and a masked endpoint must affect only b's domain.
    ts = supplied["b"]["new"]["signal"]
    mask = np.ones(ts.values.shape, bool)
    mask[0] = False
    supplied["b"]["new"]["signal"] = TimeSeries(ts.values, ts.times + 0.2, mask)
    q = np.array([0.0, 5.0, 8.0, 11.0, 21.0])
    output = model.transform(supplied, times=q)
    x = dict(zip(model.parameter_names_, model.map_parameters_))

    def matern(a, b):
        d = np.sqrt(3) * np.abs(a[:, None] - b[None, :]) / 3
        return (1 + d) * np.exp(-d)

    for s, modality in (("a", "ref"), ("b", "signal")):
        ts = supplied[s]["new"][modality]
        t, y = ts.times[ts.mask[:, 0]], ts.values[ts.mask[:, 0], 0]
        w, offset = x["loading", s, modality, 0], x["offset", s, modality, 0]
        C = w**2 * matern(t, t) + np.eye(len(t)) * x["noise", s, modality]
        if baseline:
            C += baseline[modality] ** 2
        cross = w * matern(q, t)
        mu = cross @ np.linalg.solve(C, y - offset)
        var = 1 - np.sum(cross * np.linalg.solve(C, cross.T).T, axis=1)
        result = output[s]["new"]
        valid = (q >= t.min()) & (q <= t.max())
        assert_array_equal(result.valid, valid)
        assert_allclose(result.values[valid, 0], mu[valid], atol=1e-9)
        assert_allclose(result.variance[valid, 0], var[valid], atol=1e-9)
        assert result.metadata["conditioning_subject"] == s
        assert result.metadata["conditioning_modalities"] == [modality]
        assert result.metadata["independent_subject_inference"] is True
        assert result.metadata["parameter_source"] == "original_training_MAP"
        assert result.metadata["parameter_refits"] == 0
        assert result.metadata["uncertainty"] == "conditional_on_training_MAP"
        assert result.metadata["n_conditioning_observations"] == len(t)


def test_other_subject_values_clocks_and_masks_cannot_change_projection(trained):
    model, data = trained
    supplied = donors(data)
    q = np.arange(3.0, 18.0)
    before = model.transform(supplied, times=q)
    ts = supplied["b"]["new"]["signal"]
    mask = np.ones(ts.values.shape, bool)
    mask[::3] = False
    supplied["b"]["new"]["signal"] = TimeSeries(ts.values * -7, ts.times + 0.3, mask)
    after = model.transform(supplied, times=q)
    assert_array_equal(before["a"]["new"].component_means, after["a"]["new"].component_means)
    assert_array_equal(
        before["a"]["new"].component_variances, after["a"]["new"].component_variances
    )
    assert before["a"]["new"].metadata == after["a"]["new"].metadata
    assert not np.allclose(before["b"]["new"].values, after["b"]["new"].values)


def test_transform_no_refit_selection_and_archive(trained, monkeypatch, tmp_path):
    model, data = trained
    parameters = model.map_parameters_.copy()
    configuration = copy.deepcopy(model.configuration_)

    def forbidden(*args, **kwargs):
        raise AssertionError("transform must not optimize or sample")

    module = importlib.import_module("multimodalsrm.bayesian.model")
    monkeypatch.setattr(module, "search", forbidden)
    monkeypatch.setattr(module, "sample", forbidden)
    supplied = {"a": {"new": {"ref": data["a"]["train"]["ref"], "signal": object()}}}
    result = model.transform(supplied, times={"new": [4.0, 9.0]}, modalities=["ref"])["a"]["new"]
    assert_array_equal(model.map_parameters_, parameters)
    assert model.configuration_ == configuration
    assert set(model.problem_.systems) == {"train"}
    path = tmp_path / "alignment.npz"
    save_results(path, {("a", "new", "latent"): result})
    restored, _ = load_results(path)
    loaded = restored["a", "new", "latent"]
    assert_array_equal(result.component_means, loaded.component_means)
    assert_array_equal(result.component_variances, loaded.component_variances)
    assert _json(result.metadata) == loaded.metadata


@pytest.mark.parametrize("readout", ["gp", "instantaneous"])
def test_transform_chain_uses_original_training_map(trained, readout):
    model, data = trained
    chained = model.condition(donors(data, "first"), targets={"b": ["signal"]})
    q = [4.0, 8.0, 12.0]
    direct = model.transform(donors(data, "second"), times=q, readout=readout)
    later = chained.transform(donors(data, "second"), times=q, readout=readout)
    for s in data:
        assert_array_equal(direct[s]["second"].component_means, later[s]["second"].component_means)
        assert_array_equal(
            direct[s]["second"].component_variances,
            later[s]["second"].component_variances,
        )
        assert direct[s]["second"].metadata == later[s]["second"].metadata


@pytest.mark.parametrize("selection", [[], "ref", ["missing"], ["ref", "ref"]])
def test_transform_invalid_modality_selection(trained, selection):
    model, data = trained
    with pytest.raises(ValueError, match="modalit"):
        model.transform(donors(data), times=[4.0, 8.0], modalities=selection)


def test_transform_rejects_missing_selected_run_and_training_ids(trained):
    model, data = trained
    with pytest.raises(ValueError, match="selected"):
        model.transform(donors(data), times=[4.0, 8.0], modalities=["ref"])
    with pytest.raises(ValueError, match="overlap training"):
        model.transform(data, times=[4.0, 8.0])
    supplied = donors(data)
    supplied["b"] = {"different": supplied["b"]["new"]}
    with pytest.raises(ValueError, match="query runs"):
        model.transform(supplied, times={"new": [4.0, 8.0]})


@pytest.mark.parametrize("change", ["specification", "posterior_snapshot", "configuration"])
def test_transform_rejects_changed_fit_contract(trained, change):
    original, data = trained
    model = copy.deepcopy(original)
    if change == "specification":
        model.length_scale *= 2
    elif change == "posterior_snapshot":
        model.training_fit_["inference"] = "posterior"
    else:
        model.configuration_["inference"] = "posterior"
    with pytest.raises(ValueError, match="specification|MAP"):
        model.transform(donors(data), times=[4.0, 8.0])


def test_learned_filter_projection_matches_continuous_integrator():
    from multimodalsrm import Gaussian, Identity, temporal_isc
    from multimodalsrm.bayesian import BayesianMultimodalSRM, SearchConfig

    from ..reference.continuous_covariance import response_covariance

    p, adapter, data = problem_fixture(gaussian=True)
    model = BayesianMultimodalSRM(
        priors=p.priors,
        anchor=p.anchor,
        responses=adapter.responses_,
        inference="map",
        random_state=413,
        search=SearchConfig(starts=1, maxiter=500),
    ).fit(data)
    supplied = donors(data)
    native = np.arange(0.0, 21, 0.7)
    mask = np.ones((len(native), 1), bool)
    mask[7] = False
    supplied["b"]["new"]["signal"] = TimeSeries(np.cos(native[:, None] / 3), native, mask)
    q = np.arange(5.0, 15.0, 2.0)
    transformed = model.transform(supplied, times=q)
    x = dict(zip(model.parameter_names_, model.map_parameters_))
    kernel = Gaussian(x["filter", "signal", "width"], x["filter", "signal", "lag"])
    t = native[mask[:, 0] & adapter._support(native, "signal", (native[0], native[-1]))]
    w = x["loading", "b", "signal", 0]
    C = w**2 * response_covariance(t, t, kernel, kernel, 3, tolerance=1e-9)
    C += np.eye(len(t)) * x["noise", "b", "signal"]
    cross = w * response_covariance(q, t, Identity(), kernel, 3, tolerance=1e-9)
    expected = cross @ np.linalg.solve(C, np.cos(t / 3) - x["offset", "b", "signal", 0])
    variance = 1 - np.sum(cross * np.linalg.solve(C, cross.T).T, axis=1)
    assert_allclose(transformed["b"]["new"].values[:, 0], expected, atol=1e-7)
    assert_allclose(transformed["b"]["new"].variance[:, 0], variance, atol=1e-7)
    assert transformed["b"]["new"].metadata["n_conditioning_observations"] == len(t)
    # Exercise the actual transform -> evaluator boundary, not fabricated metadata.
    report = temporal_isc({s: runs["new"] for s, runs in transformed.items()})
    assert report["input_source"] == "independent_training_MAP"
    assert report["n_subjects"] == 2


def test_transform_retains_mapping_feature_and_capacity_guards(trained):
    model, data = trained
    with pytest.raises(ValueError, match="mapping"):
        model.transform({"stranger": donors(data)["a"]}, times=[4.0, 8.0])
    ts = data["a"]["train"]["ref"]
    wrong_features = TimeSeries(np.tile(ts.values, (1, 2)), ts.times)
    with pytest.raises(ValueError, match="feature count"):
        model.transform({"a": {"new": {"ref": wrong_features}}}, times=[4.0, 8.0])
    t = np.linspace(0, 20, model.max_observations + 1)
    with pytest.raises(ValueError, match="max_observations"):
        model.transform(
            {"a": {"new": {"ref": TimeSeries(np.sin(t)[:, None], t)}}}, times=[4.0, 8.0]
        )


def test_modality_selection_does_not_read_lazy_excluded_payload(trained):
    model, data = trained

    class LazyModalities(Mapping):
        def __iter__(self):
            return iter(["ref", "signal"])

        def __len__(self):
            return 2

        def __getitem__(self, key):
            if key == "ref":
                return data["a"]["train"]["ref"]
            raise AssertionError("excluded payload was accessed")

    actual = model.transform(
        {"a": {"new": LazyModalities()}}, modalities=["ref"], times=[4.0, 8.0]
    )["a"]["new"]
    expected = model.transform({"a": donors(data)["a"]}, times=[4.0, 8.0])["a"]["new"]
    assert_array_equal(actual.values, expected.values)
