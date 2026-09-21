"""Pointwise decoding uses frozen maps, observed features, and no temporal borrowing."""
# ruff: noqa: F811 - pytest fixtures are imported for module-local reuse

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.workflow import load_results, save_results

from .test_bayesian_multifactor import fitted  # noqa: F401


def test_instantaneous_matches_observation_space_gaussian(fitted, monkeypatch, tmp_path):
    model, data = fitted
    from multimodalsrm.bayesian import model as module

    def forbidden(*args, **kwargs):
        raise AssertionError("pointwise transform must not fit or evaluate a GP")

    monkeypatch.setattr(module, "search", forbidden)
    monkeypatch.setattr(module.BayesianProblem, "nll", forbidden)
    original = model.map_parameters_.copy()
    supplied = {s: {"test": mods["train"]} for s, mods in data.items()}
    q = np.arange(0.0, 17.0)  # Unobserved odd timestamps must be invalid.
    output = model.transform(supplied, times=q, modalities=["brain"], readout="instantaneous")
    params = dict(zip(model.parameter_names_, original))
    rotation = np.asarray(model.configuration_["factor_orientation"]["rotation"])
    for s, runs in supplied.items():
        ts = runs["test"]["brain"]
        result = output[s]["test"]
        w = (
            np.array([[params["loading", s, "brain", f, k] for k in range(2)] for f in range(3)])
            @ rotation
        )
        offset = np.array([params["offset", s, "brain", f] for f in range(3)])
        noise = params["noise", s, "brain"]
        assert_array_equal(result.valid, np.isin(q, ts.times))
        for i, t in enumerate(ts.times):
            mask = ts.mask[i]
            a = w[mask]
            cov_y = a @ a.T + noise * np.eye(mask.sum())
            cross = np.linalg.solve(cov_y, a).T
            expected_mean = cross @ (ts.values[i, mask] - offset[mask])
            expected_var = np.diag(np.eye(2) - cross @ a)
            assert_allclose(result.values[int(t)], expected_mean, atol=1e-10)
            assert_allclose(result.variance[int(t)], expected_var, atol=1e-10)
        assert result.metadata["alignment_readout"] == "instantaneous"
        assert result.metadata["temporal_conditioning"] == "same_timestamp_only"
        assert result.metadata["parameter_refits"] == 0
        assert result.metadata["posterior_scope"] == "instantaneous_marginal_model"
    assert_array_equal(model.map_parameters_, original)
    path = tmp_path / "pointwise.npz"
    save_results(path, {(s, "test", "latent"): runs["test"] for s, runs in output.items()})
    restored, _ = load_results(path)
    assert_allclose(restored["a", "test", "latent"].values, output["a"]["test"].values)


def test_instantaneous_is_subject_and_timestamp_local(fitted):
    model, data = fitted
    supplied = {s: {"test": copy.deepcopy(mods["train"])} for s, mods in data.items()}
    first = model.transform(
        supplied, times=[0.0, 2.0], modalities=["brain"], readout="instantaneous"
    )
    for s in supplied:
        ts = supplied[s]["test"]["brain"]
        values, mask = ts.values.copy(), ts.mask.copy()
        values[1:] *= 7
        if s != "a":
            values[0] *= -4
        mask[1] = False
        supplied[s]["test"]["brain"] = TimeSeries(values, ts.times, mask)
    second = model.transform(
        supplied, times=[0.0, 2.0], modalities=["brain"], readout="instantaneous"
    )
    assert_array_equal(first["a"]["test"].values[0], second["a"]["test"].values[0])
    assert not second["a"]["test"].valid[1]
    assert not np.allclose(first["b"]["test"].values[0], second["b"]["test"].values[0])


def test_readout_validation_and_gp_default(fitted):
    model, data = fitted
    supplied = {"a": {"test": data["a"]["train"]}}
    with pytest.raises(ValueError, match="readout"):
        model.transform(supplied, times=[0.0], readout="ols")
    implicit = model.transform(supplied, times=[0.0, 1.0])["a"]["test"]
    explicit = model.transform(supplied, times=[0.0, 1.0], readout="gp")["a"]["test"]
    assert_array_equal(implicit.values, explicit.values)
    assert implicit.metadata == explicit.metadata


def test_single_observed_timestamp_and_mixed_readout_guard(fitted):
    from multimodalsrm import time_segment_matching

    model, data = fitted
    ts = data["a"]["train"]["brain"]
    supplied = {"a": {"test": {"brain": TimeSeries(ts.values[:1], ts.times[:1])}}}
    result = model.transform(supplied, times=[0.0, 1.0], readout="instantaneous")["a"]["test"]
    assert_array_equal(result.valid, [True, False])
    supplied = {s: {"test": mods["train"]} for s, mods in data.items()}
    gp = model.transform(supplied, times=np.arange(0.0, 17.0, 2.0))
    instant = model.transform(supplied, times=np.arange(0.0, 17.0, 2.0), readout="instantaneous")
    with pytest.raises(ValueError, match="same alignment readout"):
        time_segment_matching({"a": gp["a"]["test"], "b": instant["b"]["test"]}, window_size=3)


def test_scalar_multimodal_native_clocks_use_noise_weighted_same_time_features():
    from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig

    t = np.arange(12.0)
    data = {
        "a": {
            "train": {
                "ref": TimeSeries(np.column_stack([np.sin(t), np.cos(t)]), t),
                "aux": TimeSeries((2 * np.sin(t[::2]) + 0.3)[:, None], t[::2]),
            }
        }
    }
    model = BayesianMultimodalSRM(
        priors=BayesianPriors(noise=Prior.lognormal(np.log(0.3), 1.0)),
        anchor=("a", "ref", 0),
        inference="map",
        search=SearchConfig(starts=1, maxiter=300),
    ).fit(data)
    supplied = {"a": {"test": data["a"]["train"]}}
    result = model.transform(supplied, times=[2.0, 3.0], readout="instantaneous")["a"]["test"]
    p = dict(zip(model.parameter_names_, model.map_parameters_))
    for i, q in enumerate(result.times):
        numerator, precision, count = 0.0, 1.0, 0
        for m, ts in supplied["a"]["test"].items():
            for row in np.flatnonzero(ts.times == q):
                for f in np.flatnonzero(ts.mask[row]):
                    w, b, v = (
                        p["loading", "a", m, f],
                        p["offset", "a", m, f],
                        p["noise", "a", m],
                    )
                    numerator += w * (ts.values[row, f] - b) / v
                    precision += w**2 / v
                    count += 1
        assert_allclose(result.values[i, 0], numerator / precision, atol=1e-10)
        assert_allclose(result.variance[i, 0], 1 / precision, atol=1e-10)
        assert result.metadata["n_conditioning_observations_per_query"][i] == count


@pytest.mark.parametrize("variant", ["filter", "baseline", "noise", "spectral"])
def test_instantaneous_rejects_ineligible_fitted_models(variant):
    from multimodalsrm.bayesian import BayesianMultimodalSRM, SearchConfig

    from .test_bayesian_problem import problem_fixture

    p, adapter, data = problem_fixture(gaussian=variant == "filter")
    kwargs = (
        {"run_baseline_sd": {"ref": 0.5}}
        if variant == "baseline"
        else ({"noise_timescales": {"ref": 2.0}} if variant == "noise" else {})
    )
    if variant == "spectral":
        from multimodalsrm.bayesian.spectral import SpectralConfig

        kwargs = dict(linear_algebra="spectral", spectral=SpectralConfig(32, 12.0))
    model = BayesianMultimodalSRM(
        priors=p.priors,
        anchor=p.anchor,
        responses=adapter.responses_,
        inference="map",
        search=SearchConfig(starts=1, maxiter=50),
        **kwargs,
    ).fit(data)
    subject, modality = ("b", "signal") if variant == "filter" else ("a", "ref")
    with pytest.raises(ValueError, match="instantaneous readout requires"):
        model.transform(
            {subject: {"new": data[subject]["train"]}},
            times=[4.0],
            modalities=[modality],
            readout="instantaneous",
        )
    if variant != "spectral":
        subject, modality = ("a", "ref") if variant == "filter" else ("b", "signal")
        good = {
            subject: {
                "new": {
                    modality: data[subject]["train"][modality],
                    "excluded": object(),
                }
            }
        }
        result = model.transform(good, times=[5.0], modalities=[modality], readout="instantaneous")
        assert result[subject]["new"].valid[0]
