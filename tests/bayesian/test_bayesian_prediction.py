"""Prediction contracts: true mixtures, direct conditioning and no leakage."""

import copy
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

from multimodalsrm import TimeSeries

from .test_bayesian_model import make_model
from .test_bayesian_problem import api, independent_parameters, problem_fixture


def test_mixture_retains_between_parameter_uncertainty_and_true_quantiles():
    b = api()
    assert hasattr(b, "GaussianMixtureSeries"), "mixture result missing"
    means = np.array([[[-2.0]], [[3.0]]])
    variances = np.array([[[0.25]], [[1.0]]])
    r = b.GaussianMixtureSeries(means, variances, [1.0], [True], {})
    assert_allclose(r.values, [[0.5]])
    assert_allclose(r.within_parameter_variance, [[0.625]])
    assert_allclose(r.between_parameter_variance, [[6.25]])
    assert_allclose(r.std**2, [[6.875]])
    lo, hi = r.interval(0.8)
    for quantile, p in [(lo, 0.1), (hi, 0.9)]:
        assert_allclose(ndtr((quantile - means) / np.sqrt(variances)).mean(0), p, atol=1e-10)
    assert_allclose(
        r.log_density(np.array([[1.0]])),
        logsumexp(norm.logpdf(1.0, means, np.sqrt(variances)), axis=0) - np.log(2.0),
    )
    assert not r.component_means.flags.writeable


def test_projection_matches_independent_gaussian_conditioning_and_noise():
    b = api()
    assert hasattr(b, "GaussianMixtureSeries"), "posterior prediction missing"
    projection = importlib.import_module("multimodalsrm.bayesian.prediction")
    p, _, _ = problem_fixture()
    x = independent_parameters(p)
    times = np.array([0.2, 4.1, 12.3])
    result = projection.project(
        p, x[None], "train", times, key=("b", "signal", 0), include_noise=False
    )
    system = p.systems["train"]
    w = np.array([1.2 if k[0] == "a" else -0.7 for k in system.keys])
    mu = np.array([0.12 if k[0] == "a" else -0.2 for k in system.keys])
    noise = np.array([0.15 if k[0] == "a" else 0.3 for k in system.keys])

    def matern(a, b):
        d = np.sqrt(3) * np.abs(a[:, None] - b[None, :]) / 3.0
        return (1 + d) * np.exp(-d)

    C = matern(system.times, system.times) * np.outer(w, w) + np.diag(noise)
    cross = matern(times, system.times) * (-0.7) * w[None, :]
    expected = cross @ np.linalg.solve(C, system.values - mu) - 0.2
    variance = 0.49 - np.einsum("ij,ji->i", cross, np.linalg.solve(C, cross.T))
    assert_allclose(result[0][0], expected, atol=1e-10)
    assert_allclose(result[1][0], variance, atol=1e-10)
    noisy = projection.project(
        p, x[None], "train", times, key=("b", "signal", 0), include_noise=True
    )
    assert_allclose(noisy[1] - result[1], 0.3, atol=1e-10)
    latent = projection.project(p, x[None], "train", times, key=None, include_noise=False)
    latent_cross = matern(times, system.times) * w[None, :]
    assert_allclose(latent[0][0], latent_cross @ np.linalg.solve(C, system.values - mu), atol=1e-10)


def test_joint_condition_excludes_poisoned_targets_and_does_not_double_count():
    model, data = make_model()
    assert hasattr(model, "condition"), "joint conditioning missing"
    model.fit(data)
    donors = {s: {"new": copy.deepcopy(runs["train"])} for s, runs in data.items()}
    clean = model.condition(donors, targets={"b": ["signal"]})
    poisoned = copy.deepcopy(donors)
    poisoned["b"]["new"]["signal"] = object()
    dirty = model.condition(poisoned, targets={"b": ["signal"]})
    assert_allclose(clean.map_parameters_, dirty.map_parameters_, atol=1e-12)
    assert len(clean.problem_.systems) == 2
    assert len(clean.problem_.systems["new"].times) == len(donors["a"]["new"]["ref"].times)
    assert len(model.problem_.systems) == 1
    assert all(k[:2] != ("b", "signal") for k in clean.problem_.systems["new"].keys)
    query = {"new": np.array([-1.0, 6.0, 9.0, 13.0, 21.0])}
    r = clean.predict(times=query)["b"]["new"]["signal"]
    assert r.valid.tolist() == [False, True, True, True, False]
    assert np.isnan(r.values[~r.valid]).all()
    assert_allclose(r.between_parameter_variance[r.valid], 0.0)
    assert r.metadata["parameter_conditioning"] == "training_plus_donors"
    noisy = clean.predict(times=query, include_noise=True)["b"]["new"]["signal"]
    assert np.all(noisy.variance[r.valid] > r.variance[r.valid])
    with pytest.raises(ValueError, match="overlap"):
        model.condition(data, targets={"b": ["signal"]})
    with pytest.raises(ValueError, match="condition"):
        model.predict(times=np.array([5.0, 9.0]))


def test_shared_latent_has_one_result_per_run_and_native_masks():
    model, data = make_model()
    assert hasattr(model, "infer_latent"), "shared latent inference missing"
    ts = data["a"]["train"]["ref"]
    mask = np.ones_like(ts.values, dtype=bool)
    mask[2] = False
    values = ts.values.copy()
    values[2] = np.nan
    data["a"]["train"]["ref"] = TimeSeries(values, ts.times, mask)
    model.fit(data)
    assert len(model.problem_.systems["train"].times) == 8 + 3  # signal fixed support envelope
    q = np.array([0.0, 1.7, 11.3, 20.0])
    out = model.infer_latent(times={"train": q})
    assert set(out) == {"train"}
    assert_allclose(out["train"].times, q)
    assert out["train"].values.shape == (4, 1)
    assert out["train"].metadata["quantity"] == "unfiltered_shared_latent"


def test_condition_rejects_changed_model_specification():
    model, data = make_model()
    model.fit(data)
    donors = {s: {"new": copy.deepcopy(runs["train"])} for s, runs in data.items()}
    model.set_params(length_scale=9.0)
    with pytest.raises(ValueError, match="model specification"):
        model.condition(donors, targets={"b": ["signal"]})


def test_zero_variance_components_have_atom_quantiles_not_a_fake_density():
    b = api()
    r = b.GaussianMixtureSeries(
        np.array([[[1.0]], [[3.0]]]), np.zeros((2, 1, 1)), [1.0], [True], {}
    )
    low, high = r.interval(0.8)
    assert_allclose(low, 1.0)
    assert_allclose(high, 3.0)
    with pytest.raises(ValueError, match="positive component"):
        r.log_density([[1.0]])


def test_new_run_name_can_match_an_excluded_modality_without_inspecting_it():
    model, data = make_model()
    data = {"a": {"train": {**data["a"]["train"], **data["b"]["train"]}}}
    model.fit(data)
    donors = {s: {"signal": copy.deepcopy(runs["train"])} for s, runs in data.items()}
    donors["a"]["signal"]["signal"] = object()
    conditioned = model.condition(donors, targets={"a": ["signal"]})
    assert set(conditioned.prediction_runs_) == {"signal"}
    assert len(conditioned.problem_.systems["signal"].times) == 9
    short = {s: entries["signal"] for s, entries in donors.items()}
    shorthand = model.condition(short, targets={"a": ["signal"]}, donor_layout="shorthand")
    assert set(shorthand.prediction_runs_) == {"run-01"}
