"""BachSCR posteriors use the same finite, L2-normalized response as MAP."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import BachSCR, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianPriors, BayesianProblem, Prior
from multimodalsrm.bayesian.persistence import _prepare

from .test_bayesian_gamma_posterior import (
    ELL,
    gamma_fixture,
    independent_temporal,
    reference_kernel,
    reference_moments,
)

SCR_ELL_PRIOR = Prior.lognormal(np.log(3.0), 0.35).bounded(1.0, 6.0)


def bach_fixture(k=3, *, mode="shape_lag", learned=True, order=64):
    model, data = gamma_fixture(k, learned=False, order=order)
    kernel = BachSCR(lag=0.1)
    fixed = {
        "fixed": kernel.parameters,
        "lag": {n: v for n, v in kernel.parameters.items() if n != "lag"},
        "shape_lag": {"t0": kernel.t0},
        "shape_t0": {"lag": kernel.lag},
    }[mode]
    response = Response(
        kernel,
        estimate=True,
        fixed=fixed,
        pooling="shared",
        bounds={
            "t0": (2.5, 3.5),
            "sigma": (0.5, 0.9),
            "lambda1": (0.26, 0.38),
            "lambda2": (0.055, 0.085),
            "lag": (-0.1, 0.3),
        },
    )
    model.responses["signal"] = response
    model.length_scale = SCR_ELL_PRIOR if learned else 3.0
    model.priors = BayesianPriors(
        loading_sd=0.7,
        offset_sd=0.3,
        noise=model.priors.noise,
        filters={
            "signal": {
                n: Prior.normal(v, 0.15 if n == "lag" else 0.15 * v)
                for n, v in kernel.parameters.items()
                if n in response.free_parameters
            }
        },
    )
    times = np.array([0.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    for runs in data.values():
        for modalities in runs.values():
            for m, series in modalities.items():
                modalities[m] = TimeSeries(series.values, times)
    return model, data


def prepared_bach(*args, **kw):
    model, data = bach_fixture(*args, **kw)
    model._config()
    model.adapter_, model.problem_ = _prepare(model, data)
    p = model.problem_
    x = p.initial.copy()
    rng = np.random.default_rng(421)
    for i, n in enumerate(p.names):
        if n[0] == "loading":
            x[i] = rng.normal(scale=0.7)
            if p.features == 1 and n == ("loading", *p.anchor):
                x[i] = abs(x[i])
    return model, p, x, data


@pytest.mark.parametrize("mode", ["fixed", "lag", "shape_lag", "shape_t0"])
@pytest.mark.parametrize("learned", [False, True])
def test_bach_posterior_preserves_map_likelihood_prior_and_parameter_selection(mode, learned):
    model, p, x, data = prepared_bach(mode=mode, learned=learned)
    map_model = copy.deepcopy(model).set_params(inference="map")
    map_model._config()
    _, pm = _prepare(map_model, data)
    assert p.names == pm.names
    assert p.parameter_priors == pm.parameter_priors
    assert (ELL in p.names) == learned
    actual, expected = p.value_gradient(x), pm.value_gradient(x)
    assert np.isfinite(actual[0]) and np.isfinite(actual[1]).all()
    assert_allclose(actual[0], expected[0], atol=1e-10)
    assert_allclose(actual[1], expected[1], atol=1e-10)
    free = set(model.responses["signal"].free_parameters)
    assert {n[2] for n in p.names if n[0] == "filter"} == free


def test_bach_requires_explicit_quadrature_and_identifiable_timing_selection():
    model, data = bach_fixture(learned=False, order=None)
    with pytest.raises(ValueError, match="quadrature"):
        model._config()
    model.response_quadrature_order = 32
    with pytest.warns(UserWarning, match="timing-confounded"):
        model.responses["signal"] = Response(BachSCR(), estimate=True, pooling="shared")
    model.priors.filters["signal"]["t0"] = Prior.normal(3.0745, 0.3)
    with pytest.raises(ValueError, match="fix BachSCR t0"):
        _prepare(model, data)


def test_bach_stability_guard_uses_lower_gp_prior_bound():
    model, data = bach_fixture()
    model.length_scale = Prior.lognormal(np.log(3), 0.2).bounded(0.1, 6)
    with pytest.raises(ValueError, match="support/GP timescale"):
        _prepare(model, data)


@pytest.mark.parametrize("mode", ["shape_lag", "shape_t0"])
def test_bach_shapes_timescale_and_gradients_match_physical_response(mode):
    _, p, x, _ = prepared_bach(mode=mode, order=256)
    for fraction, crossed in ((0.05, False), (0.95, False), (0.05, True), (0.95, True)):
        for i, n in enumerate(p.names):
            if n[0] in ("filter", "gp"):
                lo, hi = p.bounds[i]
                f = 1 - fraction if crossed and n[-1] in ("sigma", "lambda1") else fraction
                x[i] = lo + f * (hi - lo)
        k = reference_kernel(p, x, 1)
        u, coeff = map(np.asarray, p.response_quadrature.nodes(x, 1))
        # Independently normalized production response used by R-MSRM and MAP.
        weights = p.response_quadrature.scr_rule[1]
        assert_allclose(coeff / weights, k.evaluate(u), atol=2e-11, rtol=2e-9)
        ell = x[p.indices[ELL]]
        for delta, left in ((0.0, k), (0.6, k), (-3.0, Identity())):
            actual = p.temporal_covariance(
                x,
                np.array([delta]),
                np.array([-1 if type(left) is Identity else 1]),
                np.array([0.0]),
                np.array([1]),
            )[0, 0]
            expected = independent_temporal(left, k, delta, ell, 1536)
            assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
        _, g = p.value_gradient(x)
        for i, n in enumerate(p.names):
            if n[0] not in ("filter", "gp"):
                continue
            h = 2e-6
            d = np.eye(len(x))[i] * h
            fd = (p.objective(x + d) - p.objective(x - d)) / (2 * h)
            assert_allclose(g[i], fd, atol=3e-5, rtol=3e-5, err_msg=str(n))


@pytest.mark.parametrize("k", [3, 5])
def test_bach_multicomponent_density_joint_moments_and_haar(k):
    from multimodalsrm.bayesian.blocks import ParameterSubspace
    from multimodalsrm.bayesian.orthogonal import (
        validate_orthogonal_target,
    )
    from multimodalsrm.bayesian.trajectories import joint_moments

    _, p, x, _ = prepared_bach(k, order=128)
    times = np.array([96.0, 99.0])
    nll, mean, cov = reference_moments(p, x, "train", times, order=1024)
    actual_mean, actual_cov = joint_moments(p, "train", times)(x, np.eye(k))
    assert_allclose(p.nll(x), nll, atol=2e-5, rtol=1e-7)
    assert_allclose(actual_mean, mean, atol=2e-5, rtol=1e-6)
    assert_allclose(actual_cov, cov, atol=2e-5, rtol=1e-6)
    dense = BayesianProblem(
        p.adapter,
        p.priors,
        anchor=p.anchor,
        response_quadrature_order=128,
        length_scale_prior=SCR_ELL_PRIOR,
    )
    vd, gd = dense.value_gradient(x)
    vg, gg = p.value_gradient(x)
    assert_allclose(vd, vg, atol=1e-8)
    assert_allclose(gd, gg, atol=2e-7)
    validate_orthogonal_target(ParameterSubspace(p, x))
    rotation, _ = np.linalg.qr(np.random.default_rng(716).normal(size=(k, k)))
    count = len(p.keys) * k
    rotated = x.copy()
    rotated[:count] = (x[:count].reshape(-1, k) @ rotation).ravel()
    assert_allclose(p.objective(rotated), p.objective(x), atol=1e-8)


@pytest.fixture(scope="module")
def bach_workflow():
    # Deliberately tiny execution test; never interpreted as convergence evidence.
    model, data = bach_fixture(5, order=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        donor = model.condition(
            {"a": {"new": {"ref": data["a"]["train"]["ref"], "signal": object()}}},
            targets={"a": ["signal"]},
        )
        calibration = {"c": copy.deepcopy(data["a"])}
        calibrated = model.calibrate_posterior(calibration)
        participant = calibrated.condition_participants({"c": {"new": calibration["c"]["train"]}})[
            "c"
        ]
    return dict(training=model, donor=donor, calibration=calibrated, participant=participant)


@pytest.mark.parametrize("kind", ["training", "donor", "calibration", "participant"])
def test_bach_public_workflow_archives_and_joint_paths(bach_workflow, kind, tmp_path, monkeypatch):
    from scipy.stats import multivariate_normal

    from multimodalsrm.bayesian import _archive, workflow

    from .test_bayesian_multifactor import log_prior
    from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting

    model = bach_workflow[kind]
    p = model.problem_
    x = model.map_parameters_
    assert p.names.count(ELL) == 1
    # Rebuilt evidence targets must count each run and the joint prior once.
    expected = sum(
        -multivariate_normal.logpdf(
            p.systems[r].values,
            mean=np.asarray(p.arrays(x)[1])[p._packed[r][0]],
            cov=np.asarray(p.covariance(x, r)),
        )
        for r in p.systems
    )
    assert_allclose(p.objective(x), expected - log_prior(p, x), atol=1e-7)
    assert model.parameter_draws_.shape[:2] == (2, 4)
    assert np.isfinite(model.parameter_draws_).all()
    assert model.sampling_diagnostics_["passes"] is False
    assert all(list(n) in model.sampling_diagnostics_["active_parameters"] for n in p.names)
    run = next(iter(model.prediction_runs_))
    times = {run: [96.0, 99.0]}
    latent = model.infer_latent(times=times, max_draws=2)[run]
    paths = model.sample_latent(times=times, max_draws=2, random_state=90)[run]
    assert paths.samples.shape == (2, 1, 2, 5)
    assert np.isfinite(paths.samples).all()
    assert paths.metadata["covariance_approximation"] == p.response_quadrature.metadata()
    assert paths.metadata["sampling_diagnostics_passed"] is False
    prediction = model.predict(times=times, max_draws=2) if model.targets_ is not None else {}
    workflow.save_model(tmp_path / kind, model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / kind)
    assert_array_equal(restored.parameter_draws_, model.parameter_draws_)
    assert _archive.same(restored.configuration_, model.configuration_)
    assert_result_equal(restored.infer_latent(times=times, max_draws=2)[run], latent)
    replay = restored.sample_latent(times=times, max_draws=2, random_state=90)[run]
    assert_array_equal(replay.samples, paths.samples)
    assert _archive.same(replay.metadata, paths.metadata)
    actual = restored.predict(times=times, max_draws=2) if prediction else {}
    for s, runs in prediction.items():
        for r, modalities in runs.items():
            for m, v in modalities.items():
                assert_result_equal(actual[s][r][m], v)


def test_bach_workflow_rejects_mutated_scr_quadrature(bach_workflow, tmp_path):
    from multimodalsrm.bayesian import workflow

    for kind, model in bach_workflow.items():
        bad = copy.deepcopy(model)
        bad.problem_.response_quadrature.scr_rule[1][0] *= 2
        with pytest.raises(ValueError, match="quadrature|likelihood|target"):
            workflow.save_model(tmp_path / kind, bad)


def test_scalar_fixed_bach_posterior_and_donor_dispatch():
    model, data = bach_fixture(1, mode="fixed", learned=False, order=16)
    model.linear_algebra = "dense"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        donor = model.condition(
            {"a": {"new": {"ref": data["a"]["train"]["ref"]}}},
            targets={"a": ["signal"]},
        )
    assert not any(n[0] in ("filter", "gp") for n in model.parameter_names_)
    assert donor.posterior_update_["kind"] == "donor"
    paths = donor.sample_latent(times={"new": [96.0, 99.0]}, max_draws=2, random_state=7)["new"]
    assert paths.samples.shape == (2, 1, 2, 1)
    assert np.isfinite(paths.samples).all()
