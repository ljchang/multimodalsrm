"""Gamma posteriors retain the existing physical response and inference contracts."""

import copy
import warnings
from functools import lru_cache

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal

from multimodalsrm import (
    DoubleGamma,
    Gamma,
    Identity,
    Response,
    SampledKernel,
    TimeSeries,
)
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    BayesianPriors,
    BayesianProblem,
    Prior,
    SamplerConfig,
    SearchConfig,
)
from multimodalsrm.bayesian.persistence import _prepare

ELL = ("gp", "length_scale")
ELL_PRIOR = Prior.lognormal(np.log(1.5), 0.35).bounded(0.75, 4)


def gamma_fixture(
    k=3,
    family="gamma",
    *,
    order=32,
    learned=True,
    shapes=True,
    algebra="grouped",
    inference="posterior",
):
    kernel = (
        Gamma(2.5, 0.25, 0.1) if family == "gamma" else DoubleGamma(3.5, 0.18, 7.5, 0.25, 0.2, 0.1)
    )
    bounds = {name: (value * 0.85, value * 1.15) for name, value in kernel.parameters.items()}
    bounds["lag"] = (-0.1, 0.3)
    fixed = (
        {}
        if shapes
        else {name: value for name, value in kernel.parameters.items() if "shape" in name}
    )
    response = Response(kernel, estimate=True, fixed=fixed, pooling="shared", bounds=bounds)
    responses = dict(ref=Response(Identity(), pooling="shared", estimate=False), signal=response)
    priors = BayesianPriors(
        loading_sd=0.7,
        offset_sd=0.3,
        noise=Prior.lognormal(np.log(0.3), 0.35),
        filters={
            "signal": {
                name: Prior.normal(value, 0.15 if name == "lag" else value * 0.15)
                for name, value in kernel.parameters.items()
                if name in response.free_parameters
            }
        },
    )
    model = BayesianMultimodalSRM(
        features=k,
        responses=responses,
        priors=priors,
        factor_anchors=tuple(("a", "ref", f) for f in range(k)) if k > 1 else None,
        anchor=("a", "ref", 0),
        inference=inference,
        linear_algebra=algebra,
        response_quadrature_order=order,
        length_scale=ELL_PRIOR if learned else 1.5,
        search=SearchConfig(starts=1, maxiter=3, refine_maxiter=0),
        sampler=SamplerConfig(
            chains=2,
            warmup=4,
            draws=4,
            max_tree_depth=2,
            mass_matrix="diagonal",
            orientation_refresh="haar" if k > 1 else "none",
        ),
        random_state=901,
    )
    times = np.array([0.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])
    rng = np.random.default_rng(417 + k)
    data = {
        s: {
            "train": {
                m: TimeSeries(
                    rng.normal(size=(len(times), k if (s, m) == ("a", "ref") else 1)),
                    times,
                )
                for m in responses
            }
        }
        for s in ("a", "b")
    }
    return model, data


def prepared_gamma(*args, **kw):
    model, data = gamma_fixture(*args, **kw)
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
    if ELL in p.indices:
        x[p.indices[ELL]] = 1.4
    return model, p, x, data


@lru_cache(None)
def legendre(order):
    return np.polynomial.legendre.leggauss(order)


def reference_kernel(p, x, modality):
    if modality == -1:
        return Identity()
    m = p.modalities[modality]
    k = p.responses[m].initial_kernel()
    return k.with_parameters(
        **{n: x[p.indices["filter", m, n]] for n in p.responses[m].free_parameters}
    )


def independent_temporal(left, right, delta, ell, order=384):
    """Direct double integral, independent of JAX/prefix sums and panel rules."""

    def nodes(k):
        if type(k) is Identity:
            return np.zeros(1), np.ones(1)
        z, w = legendre(order)
        lo, hi = k.support
        t = (lo + hi) / 2 + (hi - lo) * z / 2
        return t, (hi - lo) * w / 2 * k.evaluate(t)

    u, wa = nodes(left)
    v, wb = nodes(right)
    d = np.sqrt(3) * abs(delta - u[:, None] + v[None, :]) / ell
    return float(wa @ ((1 + d) * np.exp(-d)) @ wb)


def reference_moments(p, x, run, times, *, order=384):
    pars = dict(zip(p.names, x))
    ell = pars.get(ELL, p.length_scale)
    s = p.systems[run]
    W = np.array([[pars[("loading", *key, f)] for f in range(p.features)] for key in s.keys])
    offsets = np.array([pars[("offset", *key)] for key in s.keys])
    noise = np.array([pars[("noise", *key[:2])] for key in s.keys])
    kernels = {m: reference_kernel(p, x, i) for i, m in enumerate(p.modalities)}

    @lru_cache(None)
    def temporal(m, n, delta):
        return independent_temporal(kernels[m], kernels[n], delta, ell, order)

    T = np.array(
        [
            [temporal(a[1], b[1], float(ta - tb)) for tb, b in zip(s.times, s.keys)]
            for ta, a in zip(s.times, s.keys)
        ]
    )
    C = T * (W @ W.T) + np.diag(noise)
    residual = s.values - offsets
    cross = np.array(
        [
            [
                independent_temporal(Identity(), kernels[key[1]], float(t - ti), ell, order)
                for ti, key in zip(s.times, s.keys)
            ]
            for t in times
        ]
    )
    A = (cross[:, None, :] * W.T[None, :, :]).reshape(len(times) * p.features, -1)
    prior = np.array(
        [
            [independent_temporal(Identity(), Identity(), float(a - b), ell) for b in times]
            for a in times
        ]
    )
    cf = cho_factor(C, lower=True)
    return (
        -multivariate_normal.logpdf(s.values, mean=offsets, cov=C),
        (A @ cho_solve(cf, residual)).reshape(len(times), p.features),
        np.kron(prior, np.eye(p.features)) - A @ cho_solve(cf, A.T),
    )


@pytest.mark.parametrize("family", ["gamma", "double"])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("shapes", [False, True])
def test_posterior_admits_same_gamma_likelihood_and_prior_as_map(family, learned, shapes):
    model, p, x, data = prepared_gamma(family=family, learned=learned, shapes=shapes)
    map_model = copy.deepcopy(model).set_params(inference="map")
    map_model._config()
    _, pm = _prepare(map_model, data)
    assert p.names == pm.names
    assert p.parameter_priors == pm.parameter_priors
    actual, expected = p.value_gradient(x), pm.value_gradient(x)
    assert_allclose(actual[0], expected[0])
    assert_allclose(actual[1], expected[1])
    assert p.response_quadrature.metadata()["error_bound"] is None


@pytest.mark.parametrize("family", ["gamma", "double"])
def test_learned_timescale_and_shapes_match_physical_curves_and_independent_integrals(
    family,
):
    _, p, x, _ = prepared_gamma(family=family, order=96)
    m = p.modalities.index("signal")
    times = np.array([0.0, 0.6, 1.4])
    mi = np.full(3, m)
    for ell in (0.8, 3.8):
        x[p.indices[ELL]] = ell
        k = reference_kernel(p, x, m)
        actual = np.asarray(p.temporal_covariance(x, times, mi, times, mi))
        expected = np.array(
            [[independent_temporal(k, k, float(a - b), ell, 768) for b in times] for a in times]
        )
        assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        u, coeff = p.response_quadrature.nodes(x, m)
        assert u.min() > k.support[0] and u.max() < k.support[1]
        # Integrated mass uses the same continuously normalized response curve.
        z, w = legendre(512)
        lo, hi = k.support
        t = (lo + hi) / 2 + (hi - lo) * z / 2
        assert_allclose(np.sum(coeff), np.sum((hi - lo) * w / 2 * k.evaluate(t)), atol=1e-7)
    _, g = p.value_gradient(x)
    for i, n in enumerate(p.names):
        if n[0] not in ("filter", "gp"):
            continue
        h = 2e-5
        d = np.eye(len(x))[i] * h
        fd = (p.objective(x + d) - p.objective(x - d)) / (2 * h)
        assert_allclose(g[i], fd, atol=3e-5, rtol=3e-5, err_msg=str(n))
    shorter = x.copy()
    shorter[p.indices[ELL]] = 1.2
    assert abs(p.nll(x) - p.nll(shorter)) > 0.01


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("family", ["gamma", "double"])
def test_density_joint_moments_and_haar_match_independent_gaussian(k, family):
    from multimodalsrm.bayesian.blocks import ParameterSubspace
    from multimodalsrm.bayesian.orthogonal import (
        validate_orthogonal_target,
    )
    from multimodalsrm.bayesian.trajectories import joint_moments

    _, p, x, _ = prepared_gamma(k, family, order=64)
    times = np.array([12.0, 15.0])
    nll, mean, cov = reference_moments(p, x, "train", times)
    assert_allclose(p.nll(x), nll, atol=2e-5, rtol=1e-7)
    actual_mean, actual_cov = joint_moments(p, "train", times)(x, np.eye(k))
    assert_allclose(actual_mean, mean, atol=2e-5, rtol=1e-6)
    assert_allclose(actual_cov, cov, atol=2e-5, rtol=1e-6)
    dense = BayesianProblem(
        p.adapter,
        p.priors,
        anchor=p.anchor,
        response_quadrature_order=64,
        length_scale_prior=ELL_PRIOR,
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


def test_shortest_gp_timescale_guards_quadrature_exponentials():
    model, data = gamma_fixture()
    model.length_scale = Prior.lognormal(np.log(2), 0.1).bounded(0.001, 4)
    with pytest.raises(ValueError, match="support/GP timescale"):
        _prepare(model, data)


def test_sampled_and_missing_quadrature_remain_explicitly_unsupported_for_posterior():
    model, _ = gamma_fixture(learned=False)
    model.responses["signal"] = Response(
        SampledKernel([0.0, 1.0, 2.0], [0.0, 1.0, 0.0]), estimate=False, pooling="shared"
    )
    with pytest.raises(ValueError, match="sampled|posterior response"):
        model._config()
    model, _ = gamma_fixture(learned=False)
    model.response_quadrature_order = None
    with pytest.raises(ValueError, match="quadrature"):
        model._config()


def installed_gamma(k=3, family="gamma", order=16):
    """Deterministic physical draws for evidence-target and query tests."""
    from multimodalsrm.bayesian.posterior_coordinates import metadata

    model, p, x, data = prepared_gamma(k, family, order=order)
    model._factor_anchor_keys_ = tuple(model.factor_anchors)
    model.specification_ = model._specification()
    model.training_data_ = data
    model.prediction_runs_ = dict(model.adapter_.domains_)
    model.targets_ = None
    model.parameter_names_ = p.names
    model.map_parameters_ = x
    model.parameter_draws_ = np.broadcast_to(x, (2, 4, len(x))).copy()
    model.configuration_ = {
        "inference": "posterior",
        "uncertainty": "parameter_posterior_mixture",
        "reference_convention": p.reference_convention,
        "factor_orientation": metadata(model._factor_anchor_keys_),
        "response_quadrature": p.response_quadrature.metadata(),
    }
    model.sampling_diagnostics_ = {"passes": False}
    model.training_fit_ = {
        "inference": "posterior",
        "map_parameters": x,
        "configuration": model.configuration_,
    }
    return model


@pytest.mark.parametrize("kind", ["donor", "calibration", "participant"])
@pytest.mark.parametrize("family", ["gamma", "double"])
def test_updates_rebuild_same_quadrature_target_and_one_prior(kind, family):
    from multimodalsrm.bayesian.blocks import ParameterSubspace
    from multimodalsrm.bayesian.orthogonal import (
        validate_orthogonal_target,
    )
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
        prepare_participant,
    )
    from multimodalsrm.bayesian.posterior_updates import (
        prepare_condition,
        validate_update_target,
    )

    model = installed_gamma(family=family)
    donor = {
        "a": {
            "new": {
                "ref": model.training_data_["a"]["train"]["ref"],
                "signal": object(),
            }
        }
    }
    if kind == "donor":
        result = prepare_condition(model, donor, targets={"a": ["signal"]})
    elif kind == "participant":
        donor["a"]["new"].pop("signal")
        result = prepare_participant(model, donor, participant="a")
    else:
        result = prepare_calibration(model, {"c": copy.deepcopy(model.training_data_["a"])})
    p = result.problem_
    x = p.initial.copy()
    old = dict(zip(model.parameter_names_, model.map_parameters_))
    for i, n in enumerate(p.names):
        replacement = (n[0], "a", *n[2:]) if len(n) > 2 and n[1] == "c" else n
        x[i] = old.get(replacement, x[i])
    assert p.names.count(ELL) == 1
    assert p.response_quadrature_order == 16
    assert p.response_quadrature.metadata() == model.problem_.response_quadrature.metadata()
    validate_update_target(p)
    validate_orthogonal_target(ParameterSubspace(p, x))
    # Independent observation-space Gaussian target, one normalized joint prior.
    from .test_bayesian_multifactor import log_prior

    expected = sum(
        -multivariate_normal.logpdf(
            p.systems[r].values,
            mean=np.asarray(p.arrays(x)[1])[p._packed[r][0]],
            cov=np.asarray(p.covariance(x, r)),
        )
        for r in p.systems
    )
    assert_allclose(p.objective(x), expected - log_prior(p, x), atol=1e-7)
    p.response_quadrature.positive_rule[1][0] *= 2
    with pytest.raises(ValueError, match="quadrature|target"):
        validate_update_target(p)


@pytest.mark.parametrize("family", ["gamma", "double"])
def test_joint_paths_accept_gamma_and_preserve_approximation_metadata(family):
    model = installed_gamma(family=family)
    before = model.parameter_draws_.copy()
    paths = model.sample_latent(times=[12.0, 14.0], max_draws=2, random_state=87)["train"]
    assert paths.samples.shape == (2, 1, 2, 3)
    assert np.isfinite(paths.samples).all()
    assert (
        paths.metadata["covariance_approximation"] == model.problem_.response_quadrature.metadata()
    )
    assert paths.metadata["sampling_diagnostics_passed"] is False
    assert_array_equal(model.parameter_draws_, before)


@pytest.fixture(scope="module", params=[(3, "gamma"), (5, "double")])
def gamma_workflow(request):
    k, family = request.param
    model, data = gamma_fixture(k, family, order=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        donors = {"a": {"new": {"ref": data["a"]["train"]["ref"], "signal": object()}}}
        donor = model.condition(donors, targets={"a": ["signal"]})
        calibration = {"c": copy.deepcopy(data["a"])}
        calibrated = model.calibrate_posterior(calibration)
        participant = calibrated.condition_participants({"c": {"new": calibration["c"]["train"]}})[
            "c"
        ]
    return dict(training=model, donor=donor, calibration=calibrated, participant=participant)


@pytest.mark.parametrize("kind", ["training", "donor", "calibration", "participant"])
def test_public_sampling_queries_and_archives_keep_gamma_target(
    gamma_workflow, kind, tmp_path, monkeypatch
):
    from multimodalsrm.bayesian import _archive, workflow

    from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting

    model = gamma_workflow[kind]
    run = next(iter(model.prediction_runs_))
    times = {run: [12.0, 14.0]}
    assert model.parameter_draws_.shape[:2] == (2, 4)
    assert np.isfinite(model.parameter_draws_).all()
    assert model.sampling_diagnostics_["passes"] is False
    assert ["gp", "length_scale"] in model.sampling_diagnostics_["active_parameters"]
    assert model.configuration_["response_quadrature"]["order_per_panel"] == 16
    assert model.configuration_["response_quadrature"]["error_bound"] is None
    assert all(
        list(n) in model.sampling_diagnostics_["active_parameters"]
        for n in model.problem_.names
        if n[0] == "filter"
    )
    latent = model.infer_latent(times=times, max_draws=2)[run]
    paths = model.sample_latent(times=times, max_draws=2, random_state=90)[run]
    predicted = (
        model.predict(times=times, include_noise=True, max_draws=2)
        if model.targets_ is not None
        else {}
    )
    workflow.save_model(tmp_path / kind, model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / kind)
    assert_array_equal(restored.parameter_draws_, model.parameter_draws_)
    assert _archive.same(restored.configuration_, model.configuration_)
    assert_result_equal(restored.infer_latent(times=times, max_draws=2)[run], latent)
    actual = restored.sample_latent(times=times, max_draws=2, random_state=90)[run]
    assert_array_equal(actual.samples, paths.samples)
    assert _archive.same(actual.metadata, paths.metadata)
    replay = (
        restored.predict(times=times, include_noise=True, max_draws=2)
        if restored.targets_ is not None
        else {}
    )
    for s, runs in predicted.items():
        for r, modalities in runs.items():
            for m, v in modalities.items():
                assert_result_equal(replay[s][r][m], v)


def test_archive_rejects_changed_quadrature_rules(gamma_workflow, tmp_path):
    from multimodalsrm.bayesian import workflow

    for kind, model in gamma_workflow.items():
        bad = copy.deepcopy(model)
        bad.problem_.response_quadrature.positive_rule[1][0] *= 2
        with pytest.raises(ValueError, match="quadrature|likelihood|target"):
            workflow.save_model(tmp_path / kind, bad)


@pytest.mark.parametrize("family", ["gamma", "double"])
def test_fixed_response_scalar_posterior_and_donor_dispatch(family):
    model, data = gamma_fixture(1, family, order=16, learned=False, algebra="dense")
    model.responses["signal"] = Response(
        model.responses["signal"].kernel, estimate=False, pooling="shared"
    )
    model.priors = BayesianPriors(loading_sd=0.7, offset_sd=0.3, noise=model.priors.noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        donor = model.condition(
            {"a": {"new": {"ref": data["a"]["train"]["ref"]}}},
            targets={"a": ["signal"]},
        )
    assert not any(n[0] == "filter" for n in model.parameter_names_)
    assert donor.posterior_update_["kind"] == "donor"
    assert np.isfinite(
        donor.sample_latent(times={"new": [12.0, 14.0]}, max_draws=2, random_state=7)["new"].samples
    ).all()
