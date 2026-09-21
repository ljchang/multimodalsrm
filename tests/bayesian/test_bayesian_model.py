"""Fitting tests exercise the real probability model and NUTS runtime."""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from .test_bayesian_problem import api, problem_fixture


def make_model(inference="map", **kwargs):
    b = api()
    assert hasattr(b, "BayesianMultimodalSRM"), "Bayesian estimator missing"
    p, adapter, data = problem_fixture(gaussian=True)
    model = b.BayesianMultimodalSRM(
        responses=adapter.responses_,
        priors=p.priors,
        anchor=p.anchor,
        inference=inference,
        random_state=413,
        search=b.SearchConfig(starts=4, maxiter=500),
        **kwargs,
    )
    return model, data


def test_map_uses_same_physical_posterior_and_retains_every_start():
    model, data = make_model()
    model.fit(data)
    assert len(model.restart_diagnostics_) == 4
    assert_allclose(model.objective_, model.problem_.objective(model.map_parameters_), atol=1e-10)
    assert model.objective_ == min(
        d["objective"] for d in model.restart_diagnostics_ if d["objective"] is not None
    )
    assert model.parameter_draws_.shape == (1, 1, len(model.parameter_names_))
    assert model.configuration_["latent_pooling"] == "shared"
    assert model.configuration_["parameter_density"] == "physical"
    assert model.sampling_diagnostics_ is None
    assert len(model.restart_diagnostics_[0]["initial_parameters"]) == len(model.parameter_names_)
    starts = np.array([d["initial_parameters"] for d in model.restart_diagnostics_])
    assert np.all(np.ptp(starts, axis=0) > 0.0)
    assert model.map_diagnostics_["physical_projected_gradient"] < 1e-3
    assert clone(model).get_params() == model.get_params()


def test_nuts_returns_physical_draws_and_reports_short_run_as_unvalidated():
    b = api()
    assert hasattr(b, "SamplerConfig"), "posterior sampler missing"
    model, data = make_model(
        "posterior",
        sampler=b.SamplerConfig(
            chains=2, warmup=80, draws=80, target_accept=0.95, max_tree_depth=7
        ),
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(data)
    draws = model.parameter_draws_
    assert draws.shape == (2, 80, len(model.parameter_names_))
    assert np.isfinite(draws).all()
    assert np.all(draws[..., model.problem_.indices[("noise", "a", "ref")]] > 0.0)
    diag = model.sampling_diagnostics_
    assert diag["chains"] == 2 and diag["draws"] == 80
    assert not diag["passes"]  # Too few draws for the declared ESS >= 400 requirement.
    assert diag["calibration_established"] is False
    assert len(diag["start_indices"]) == 2
    assert any("diagnostic" in str(w.message) for w in recorded)
    assert model.sample_stats_["diverging"].shape == (2, 80)
    assert np.isfinite(model.log_likelihood_draws_).all()


def test_invalid_runtime_and_dense_guard_fail_before_fitting():
    model, data = make_model()
    model.set_params(max_observations=2)
    with pytest.raises(ValueError, match="max_observations"):
        model.fit(data)
    model.set_params(max_observations=800, inference="nonsense")
    with pytest.raises(ValueError, match="inference"):
        model.fit(data)


def test_import_is_optional_and_fit_requires_explicit_float64():
    api()
    script = """
import sys
from multimodalsrm.bayesian import Prior
assert 'jax' not in sys.modules and 'numpyro' not in sys.modules
try:
    Prior.normal().distribution()
except RuntimeError as e:
    assert 'float64' in str(e)
else:
    raise AssertionError('silently used float32')
"""
    env = {**os.environ, "JAX_ENABLE_X64": "false"}
    result = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_nuts_initializes_when_map_lies_exactly_on_uniform_boundary():
    from multimodalsrm import TimeSeries

    b = api()
    t = np.linspace(0.0, 20.0, 50)
    data = {"s": {"ref": TimeSeries(np.zeros((50, 1)), t)}}
    model = b.BayesianMultimodalSRM(
        priors=b.BayesianPriors(noise=b.Prior.uniform(0.1, 1.0)),
        anchor=("s", "ref", 0),
        random_state=0,
        search=b.SearchConfig(starts=2, maxiter=1000),
        sampler=b.SamplerConfig(chains=2, warmup=80, draws=80, max_tree_depth=7),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert_allclose(model.map_parameters_[-1], 0.1, atol=1e-14)
    assert np.isfinite(model.parameter_draws_).all()
    assert np.all(model.parameter_draws_[..., -1] > 0.1)
    assert any(
        a["parameter"] == ["noise", "s", "ref"]
        for a in model.sampling_diagnostics_["initialization_adjustments"]
    )
