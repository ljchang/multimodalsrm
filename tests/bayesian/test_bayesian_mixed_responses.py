"""Independent integration and Gaussian conditioning for mixed response families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from multimodalsrm import (
    BatemanSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    Response,
    TimeSeries,
)

from .test_bayesian_problem import api


def mixed_fixture(*, algebra="dense", order=64, features=1):
    b = api()
    from multimodalsrm.bayesian.persistence import _prepare

    kernels = dict(
        brain=DoubleGamma(),
        eda=BatemanSCR(),
        ratings=Gamma(3.0, 0.7, 0.4),
        face=Gaussian(0.8, 0.2),
    )
    responses = {m: Response(k, pooling="shared", estimate=False) for m, k in kernels.items()}
    responses["ratings"] = Response(
        kernels["ratings"],
        pooling="shared",
        bounds=dict(shape=(2.5, 3.5), scale=(0.5, 0.9), lag=(0.0, 0.8)),
    )
    responses["eda"] = Response(
        kernels["eda"],
        pooling="shared",
        fixed={"rise": kernels["eda"].rise},
        bounds=dict(decay=(2.5, 3.5), lag=(-0.5, 0.5)),
    )
    priors = b.BayesianPriors(
        noise=b.Prior.lognormal(np.log(0.3), 0.5),
        filters={
            "ratings": dict(
                shape=b.Prior.normal(3.0, 0.3),
                scale=b.Prior.normal(0.7, 0.1),
                lag=b.Prior.normal(0.4, 0.2),
            ),
            "eda": dict(decay=b.Prior.normal(3.0, 0.3), lag=b.Prior.normal(0.0, 0.2)),
        },
    )
    model = b.BayesianMultimodalSRM(
        priors=priors,
        anchor=("a", "brain", 0),
        features=features,
        factor_anchors=(("a", "brain", 0), ("a", "brain", 1)) if features == 2 else None,
        reference_modality="brain",
        responses=responses,
        inference="map",
        linear_algebra=algebra,
        length_scale=3.0,
        response_quadrature_order=order,
    )
    times = np.array([0.0, 94.0, 96.0, 99.0, 103.0, 108.0, 112.0])
    data = {
        s: {
            "train": {
                m: TimeSeries(
                    np.stack([np.sin(times / 5 + i + j) for j in range(features)], axis=1),
                    times,
                )
                for i, m in enumerate(kernels)
            }
        }
        for s in ("a", "b")
    }
    model._config()
    adapter, p = _prepare(model, data)
    x = p.initial.copy()
    for i, name in enumerate(p.names):
        if name[0] == "loading":
            x[i] = 0.8 if name[1] == "a" else -0.6
        if name[0] == "noise":
            x[i] = 0.3
        if name[0] == "filter":
            x[i] = kernels[name[1]].parameters[name[2]]
    return model, p, x, kernels, data


def reference_entry(delta, left, right, length=3.0):
    rate = np.sqrt(3) / length

    def matern(d):
        z = rate * abs(d)
        return (1 + z) * np.exp(-z)

    if type(left) is Identity and type(right) is Identity:
        return matern(delta)
    if type(left) is Identity:
        return quad(
            lambda v: float(right.evaluate(v)) * matern(delta + v),
            *right.support,
            epsabs=1e-9,
            limit=300,
            points=[v for v in (-delta,) if right.support[0] < v < right.support[1]],
        )[0]
    if type(right) is Identity:
        return reference_entry(-delta, right, left, length)
    return quad(
        lambda u: float(left.evaluate(u)) * reference_entry(delta - u, Identity(), right, length),
        *left.support,
        epsabs=2e-7,
        limit=200,
    )[0]


def test_mixed_covariance_matches_independent_integrals_and_is_psd():
    _, p, x, kernels, _ = mixed_fixture()
    t = np.array([94.0, 96.0, 99.0, 103.0])
    mi = np.arange(4)
    c = np.asarray(p.temporal_covariance(x, t, mi, t, mi))
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 3), (2, 2), (2, 3), (3, 3)):
        want = reference_entry(t[i] - t[j], list(kernels.values())[i], list(kernels.values())[j])
        assert_allclose(c[i, j], want, atol=3e-6, rtol=2e-6)
    assert_allclose(c, c.T, atol=1e-11)
    assert np.linalg.eigvalsh(c).min() > -1e-10
    assert p.covariance_error_bound is None  # quadrature is not a certified bound


def test_mixed_dense_grouped_likelihood_and_filter_gradients_agree():
    _, dense, x, _, _ = mixed_fixture()
    _, grouped, y, _, _ = mixed_fixture(algebra="grouped")
    assert_allclose(x, y)
    vd, gd = dense.value_gradient(x)
    vg, gg = grouped.value_gradient(x)
    assert_allclose(vd, vg, atol=1e-8)
    assert_allclose(gd, gg, atol=2e-7, rtol=2e-7)
    for i, name in enumerate(dense.names):
        if name[0] != "filter":
            continue
        h = 2e-5 * max(abs(x[i]), 0.01)
        direction = np.eye(len(x))[i] * h
        fd = (float(dense.objective(x + direction)) - float(dense.objective(x - direction))) / (
            2 * h
        )
        assert_allclose(gd[i], fd, atol=2e-4, rtol=2e-4, err_msg=str(name))


def test_mixed_responses_require_explicit_quadrature():

    model, _, _, _, data = mixed_fixture()
    model.set_params(response_quadrature_order=None)
    with pytest.raises(ValueError, match="quadrature"):
        model.fit(data)


def test_mixed_path_rejects_spectral_before_fitting():
    model, _, _, _, _ = mixed_fixture()
    model.set_params(inference="map", linear_algebra="spectral")
    with pytest.raises(ValueError, match="dense/grouped"):
        model._config()


def test_fast_matern_sum_matches_direct_sum_for_signed_weights_and_distant_queries():
    api()
    from multimodalsrm.bayesian.response_quadrature import weighted_matern

    nodes = np.array([-90.0, -5.0, -0.2, 0.0, 3.0, 30.0])
    weights = np.array([0.001, 0.1, -0.4, 0.7, -0.2, 0.01])
    queries = np.array([-10000.0, -100.0, -5.0, -0.5, 0.0, 2.0, 30.0, 100.0, 10000.0])
    d = np.sqrt(3.0) * abs(queries[:, None] - nodes[None, :]) / 3.0
    expected = ((1 + d) * np.exp(-d)) @ weights
    assert_allclose(weighted_matern(queries, nodes, weights, 3.0), expected, atol=1e-12, rtol=1e-12)


def test_mixed_conditional_moments_match_direct_gaussian_conditioning():
    from multimodalsrm.bayesian.prediction import project

    _, p, x, _, _ = mixed_fixture()
    system = p.systems["train"]
    t = np.array([94.0, 99.0, 104.0])
    ki, _, mi = p._packed["train"]
    w, offset, noise, _, _ = map(np.asarray, p.arrays(x))
    cov = np.asarray(p.covariance(x, "train"))
    residual = system.values - offset[ki]
    for target in (None, ("a", "ratings", 0), ("a", "eda", 0)):
        qm = -1 if target is None else p.modalities.index(target[1])
        weight = 1.0 if target is None else w[p.keys.index(target)]
        base = 0.0 if target is None else offset[p.keys.index(target)]
        cross = (
            np.asarray(p.temporal_covariance(x, t, np.full(len(t), qm), system.times, mi))
            * weight
            * w[ki][None, :]
        )
        prior = (
            np.diag(p.temporal_covariance(x, t, np.full(len(t), qm), t, np.full(len(t), qm)))
            * weight**2
        )
        mean, var = project(p, x[None], "train", t, key=target, include_noise=False)
        assert_allclose(mean[0], cross @ np.linalg.solve(cov, residual) + base, atol=1e-9)
        assert_allclose(
            var[0],
            prior - np.sum(cross * np.linalg.solve(cov, cross.T).T, axis=1),
            atol=1e-9,
        )


def test_mixed_fit_archive_and_independent_transform(tmp_path, monkeypatch):
    import warnings

    from multimodalsrm.bayesian import SearchConfig
    from multimodalsrm.bayesian.workflow import load_model, save_model

    from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting

    model, _, _, _, data = mixed_fixture(algebra="grouped", features=2, order=16)
    # This is an API/replay test, not a converged recovery experiment.
    data["b"]["train"].pop("face")
    ts = data["a"]["train"]["ratings"]
    mask = np.ones(ts.values.shape, dtype=bool)
    mask[2, 1] = False
    values = ts.values.copy()
    values[~mask] = np.nan
    data["a"]["train"]["ratings"] = TimeSeries(values, ts.times, mask)
    model.set_params(search=SearchConfig(starts=1, maxiter=2), random_state=421)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model.fit(data)
    new = {s: {"new": r["train"]} for s, r in data.items()}
    expected = model.transform(new, times=[96.0, 99.0])
    predicted = model.condition(new, targets={"b": ["ratings"]}, mode="frozen").predict(
        times=[96.0, 99.0]
    )
    save_model(tmp_path / "mixed", model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = load_model(tmp_path / "mixed")
    assert restored.configuration_ == model.configuration_
    assert restored.configuration_["response_quadrature"]["error_bound"] is None
    actual = restored.transform(new, times=[96.0, 99.0])
    for s in actual:
        assert_result_equal(actual[s]["new"], expected[s]["new"])
        assert actual[s]["new"].metadata["covariance_approximation"]["order_per_panel"] == 16
    replay = restored.condition(new, targets={"b": ["ratings"]}, mode="frozen").predict(
        times=[96.0, 99.0]
    )
    assert_result_equal(replay["b"]["new"]["ratings"], predicted["b"]["new"]["ratings"])


def test_structured_quality_reports_shape_and_reference_clock():
    from multimodalsrm.bayesian.quality import summarize_fit

    model, p, x, _, _ = mixed_fixture()
    record = dict(
        configuration=dict(
            response_metadata={m: r.metadata for m, r in model.responses.items()},
            reference_convention=p.reference_convention,
        ),
        parameter_names=p.names,
        map=dict(parameters=x.tolist()),
        restarts=[],
    )
    report = summarize_fit(record)
    assert report["availability"] == "complete"
    rows = {r["modality"]: r for r in report["filters"]}
    assert_allclose(rows["ratings"]["peak_delay_seconds"], 1.8, atol=1e-5)
    assert 4.9 < rows["brain"]["peak_delay_seconds"] < 5.1
    assert rows["brain"]["has_negative_lobe"] is True
    assert rows["eda"]["fwhm_seconds"] > 0


def test_structured_reference_does_not_claim_fixed_dispersion():
    from multimodalsrm.bayesian.reference import reference_convention

    _, p, _, _, _ = mixed_fixture()
    from dataclasses import replace

    p.adapter.responses_["ratings"] = replace(p.responses["ratings"], fixed={"lag": 0.4})
    reference = reference_convention(p.adapter, p.keys, p.anchor, "ratings", p.systems)
    assert reference["reference_width"] == "not_applicable"
    assert reference["reference_shape_parameters_estimated"] == ["shape", "scale"]
