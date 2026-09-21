"""Clock conventions are explicit and distinct from physiological recovery."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from multimodalsrm import Gaussian, Response, TimeSeries
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

from .test_bayesian_problem import api


def filtered_fixture(shift=0.0):
    b = api()
    responses = {
        "ref": Response(
            Gaussian(0.35, shift),
            pooling="shared",
            fixed={"lag": shift},
            bounds={"width": (0.2, 0.5)},
        ),
        "signal": Response(
            Gaussian(0.45, 0.6 + shift),
            pooling="shared",
            bounds={"width": (0.2, 0.7), "lag": (-0.2 + shift, 1.4 + shift)},
        ),
    }
    priors = b.BayesianPriors(
        noise=b.Prior.lognormal(np.log(0.2), 0.7),
        filters={
            "ref": {"width": b.Prior.normal(0.35, 0.15)},
            "signal": {
                "width": b.Prior.normal(0.45, 0.2),
                "lag": b.Prior.normal(0.6 + shift, 0.4),
            },
        },
    )
    data = {}
    for s, weight in [("a", 1.0), ("b", 0.7)]:
        data[s] = {}
        for run, origin in [("train", 0.0), ("second", 100.0)]:
            data[s][run] = {}
            for m, lag in [("ref", 0.0), ("signal", 0.6)]:
                t = np.linspace(0, 24, 25)
                v = weight * np.c_[np.sin((t - lag) / 3), -np.cos((t - lag) / 4)]
                mask = np.ones_like(v, dtype=bool)
                mask[8, 1] = False
                data[s][run][m] = TimeSeries(v, t + origin, mask)
    adapter = BayesianObservationAdapter(
        features=1,
        latent_dt=1.0,
        responses=responses,
        standardize=False,
    )
    adapter._prepare(data)
    return b, adapter, priors, data


def problem(adapter, priors, **kwargs):
    return api().BayesianProblem(
        adapter, priors, anchor=("a", "signal", 0), reference_modality="ref", **kwargs
    )


def test_filtered_reference_fixes_only_lag_and_separates_sign_anchor():
    b, adapter, priors, _ = filtered_fixture()
    p = problem(adapter, priors)
    assert ("filter", "ref", "width") in p.names
    assert ("filter", "ref", "lag") not in p.names
    assert p.bounds[p.indices[("loading", "a", "signal", 0)]][0] == 0
    assert p.bounds[p.indices[("loading", "a", "ref", 0)]][0] == -np.inf
    assert p.reference_convention["reference_width"] == "learned"
    assert p.reference_convention["reference_lag_seconds"] == 0.0
    automatic = b.BayesianProblem(adapter, priors, anchor=("a", "signal", 0))
    assert automatic.reference_convention["reference_modality"] == "ref"
    assert automatic.reference_convention["mode"] == "automatic_fixed_lag"
    assert_allclose(automatic.value_gradient(p.initial)[0], p.value_gradient(p.initial)[0])


@pytest.mark.parametrize("identity", [False, True])
def test_explicit_reference_accepts_fully_fixed_responses_and_retains_draw_axes(
    identity,
):
    from multimodalsrm import Identity

    b, adapter, priors, _ = filtered_fixture()
    adapter.responses_["ref"] = Response(
        Identity() if identity else Gaussian(0.35, 0.2),
        pooling="shared",
        estimate=False,
    )
    del priors.filters["ref"]
    p = problem(adapter, priors)
    assert p.reference_convention["reference_width"] == "fixed"
    model = b.BayesianMultimodalSRM(priors=priors, anchor=p.anchor, reference_modality="ref")
    with pytest.raises(NotFittedError, match="not fitted"):
        model.relative_lag_draws()
    draws = np.broadcast_to(p.initial, (2, 3, len(p.names))).copy()
    signal_lags = np.array([[0.1, 0.2, 0.5], [0.8, 1.0, 1.2]])
    draws[..., p.indices[("filter", "signal", "lag")]] = signal_lags
    model.problem_, model.parameter_draws_ = p, draws
    derived = model.relative_lag_draws()
    assert_allclose(derived["signal"], signal_lags - (0 if identity else 0.2))
    assert_allclose(derived["ref"], np.zeros((2, 3)))
    assert not derived["signal"].flags.writeable


@pytest.mark.parametrize(
    "failure", ["free_lag", "missing_reference", "fixed_prior", "missing_anchor"]
)
def test_invalid_conventions_fail_before_optimization(failure):
    b, adapter, priors, _ = filtered_fixture()
    kwargs = dict(anchor=("a", "signal", 0), reference_modality="ref")
    match = "fixed lag"
    if failure == "free_lag":
        adapter.responses_["ref"] = Response(Gaussian(0.35), pooling="shared")
    elif failure == "missing_reference":
        kwargs["reference_modality"] = "absent"
        match = "reference_modality"
    elif failure == "fixed_prior":
        priors = copy.deepcopy(priors)
        priors.filters["ref"]["lag"] = b.Prior.normal()
        match = "filter priors"
    else:
        kwargs["anchor"] = ("a", "signal", 9)
        match = "anchor"
    with pytest.raises(ValueError, match=match):
        b.BayesianProblem(adapter, priors, **kwargs)


@pytest.mark.parametrize("role", ["anchor", "reference"])
def test_anchor_and_reference_require_support_eligible_training_observations(role):
    _, adapter, priors, data = filtered_fixture()
    for s, runs in data.items():
        for modalities in runs.values():
            m = "signal" if role == "anchor" else "ref"
            ts = modalities[m]
            mask = ts.mask.copy()
            # Observations exist at raw edges, but all are outside filter support.
            if role == "reference":
                mask[1:-1] = False
            elif s == "a":
                mask[1:-1, 0] = False
            modalities[m] = TimeSeries(ts.values, ts.times, mask)
    with pytest.raises(ValueError, match="support-eligible"):
        adapter._prepare(data)
        problem(adapter, priors)


def test_all_filtered_grouped_density_gradients_and_predictions_match_dense():
    from multimodalsrm.bayesian.prediction import project

    _, adapter, priors, _ = filtered_fixture()
    dense = problem(adapter, priors)
    grouped = problem(adapter, priors, linear_algebra="grouped")
    x = dense.initial.copy()
    rng = np.random.default_rng(318)
    for i, name in enumerate(dense.names):
        if name[0] == "loading" and name != ("loading", *dense.anchor):
            x[i] = rng.normal()
    vd, gd = dense.value_gradient(x)
    vg, gg = grouped.value_gradient(x)
    assert_allclose(vg, vd, atol=1e-9)
    assert_allclose(gg, gd, atol=1e-8, rtol=1e-8)
    for key in [None, ("b", "signal", 1)]:
        args = dict(key=key, include_noise=key is not None)
        assert_allclose(
            project(grouped, x[None], "train", np.array([6.1, 12.8]), **args),
            project(dense, x[None], "train", np.array([6.1, 12.8]), **args),
            atol=1e-9,
        )


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_common_lag_shift_preserves_density_predictions_and_relative_lags(algebra):
    from multimodalsrm.bayesian.prediction import project

    _, adapter, priors, _ = filtered_fixture()
    old = problem(adapter, priors, linear_algebra=algebra)
    shift = 1.25
    _, shifted_adapter, shifted_priors, _ = filtered_fixture(shift)
    # Hold the included observations fixed; shifting support bounds can select
    # different edges if systems are independently rebuilt.
    new = problem(shifted_adapter, shifted_priors, systems=old.systems, linear_algebra=algebra)
    x, y = old.initial.copy(), old.initial.copy()
    y[old.indices[("filter", "signal", "lag")]] += shift
    assert_allclose(new.value_gradient(y)[0], old.value_gradient(x)[0], atol=1e-9)
    assert_allclose(new.value_gradient(y)[1], old.value_gradient(x)[1], atol=1e-9)
    for run in old.systems:
        assert_allclose(new.covariance(y, run), old.covariance(x, run), atol=1e-12)
    q = np.array([7.1, 12.3, 15.8])
    for key in [None, ("b", "signal", 0)]:
        args = dict(key=key, include_noise=False)
        assert_allclose(
            project(new, y[None], "train", q - shift if key is None else q, **args),
            project(old, x[None], "train", q, **args),
            atol=1e-10,
        )
    for p, v in [(old, x), (new, y)]:
        m = api().BayesianMultimodalSRM(priors=p.priors, anchor=p.anchor)
        m.problem_, m.parameter_draws_ = p, v[None, None]
        lags = m.relative_lag_draws()
        assert lags["signal"].shape == (1, 1)
        assert_allclose(lags["signal"], 0.6, atol=1e-12)
        assert_allclose(lags["ref"], 0.0)
        assert not lags["ref"].flags.writeable


def test_convention_survives_clone_donor_exclusion_and_result_archive(tmp_path):
    from multimodalsrm.bayesian.workflow import load_results, save_results

    b, adapter, priors, data = filtered_fixture()
    model = b.BayesianMultimodalSRM(
        priors=priors,
        responses=adapter.responses_,
        anchor=("a", "signal", 0),
        reference_modality="ref",
        inference="map",
        linear_algebra="grouped",
        search=b.SearchConfig(starts=1, maxiter=150, refine_maxiter=30),
    )
    assert clone(model).get_params() == model.get_params()
    train = {s: {"train": runs["train"]} for s, runs in data.items()}
    model.fit(train)
    # Reference and sign anchor can both be excluded from new donors; training
    # still defines the convention. Poisoned target payloads must not be read.
    donors = {s: {"new": copy.deepcopy(runs["second"])} for s, runs in data.items()}
    targets = {"a": ["ref", "signal"], "b": ["ref"]}
    for s, ms in targets.items():
        for m in ms:
            donors[s]["new"][m] = object()
    conditioned = model.condition(donors, targets=targets)
    assert conditioned.reference_modality == "ref"
    convention = model.configuration_["reference_convention"]
    assert conditioned.configuration_["reference_convention"] == convention
    assert all(k[1] != "ref" for k in conditioned.problem_.systems["new"].keys)
    q = {"new": np.array([108.0, 112.0, 116.0])}
    pred = conditioned.predict(times=q)["a"]["new"]["ref"]
    latent = conditioned.infer_latent(times=q)["new"]
    assert pred.metadata["reference_convention"] == convention
    assert latent.metadata["reference_convention"] == convention
    save_results(tmp_path / "results", {("a", "new", "ref"): pred}, metadata={})
    restored, _ = load_results(tmp_path / "results")
    assert restored[("a", "new", "ref")].metadata["reference_convention"] == convention
    model.set_params(reference_modality="signal")
    with pytest.raises(ValueError, match="specification"):
        model.condition(donors, targets=targets)
