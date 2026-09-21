"""Optional conventions preserve the model and its identifiable contrasts."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from multimodalsrm import Gaussian, Response, TimeSeries

from .test_bayesian_problem import api


def setup(features=1):
    b = api()
    responses = {
        m: Response.lag_only(Gaussian(0.15, lag), pooling="shared", bounds={"lag": (-1.0, 1.0)})
        for m, lag in (("brain", 0.4), ("physio", 0.1))
    }
    priors = b.BayesianPriors(
        noise=b.Prior.lognormal(-2.0, 0.5),
        filters={m: {"lag": b.Prior.normal(0.0, 0.6)} for m in responses},
    )
    t = np.arange(12.0)
    data = {
        s: {
            "train": {
                m: TimeSeries((np.sin((t - lag) / 2) + w * np.cos(t / 3))[:, None], t)
                for m, lag in (("brain", 0.4), ("physio", 0.1))
            }
        }
        for s, w in (("b", -0.4), ("a", 0.7))
    }
    model = b.BayesianMultimodalSRM(
        features=features,
        priors=priors,
        responses=responses,
        inference="map",
        linear_algebra="grouped",
        random_state=452,
        search=b.SearchConfig(starts=1, maxiter=40, refine_maxiter=10),
    )
    return b, model, data


def prepared(features=1, **problem_options):
    from multimodalsrm.bayesian.observation_adapter import (
        BayesianObservationAdapter,
    )

    b, model, data = setup(features)
    adapter = BayesianObservationAdapter(
        features=features,
        latent_dt=1.0,
        responses=model.responses,
        length_scale=model.length_scale,
        standardize=False,
    )
    adapter._prepare(data)
    return model, b.BayesianProblem(adapter, model.priors, **problem_options), data


@pytest.mark.parametrize("features", [1, 2])
def test_constructor_and_clone_do_not_require_feature_or_modality_reference(features):
    _, model, _ = setup(features)
    model._config()
    assert model.anchor is None and model.factor_anchors is None
    assert model.reference_modality is None
    assert clone(model).get_params() == model.get_params()


def test_all_filtered_lags_remain_free_with_their_original_priors_and_bounds():
    model, problem, _ = prepared()
    assert problem.anchor == ("a", "brain", 0)
    for modality in model.responses:
        index = problem.indices[("filter", modality, "lag")]
        assert problem.bounds[index] == (-1.0, 1.0)
        assert problem.parameter_priors[index] == model.priors.filters[modality]["lag"].bounded(
            -1, 1
        )
    assert problem.reference_convention["reference_modality"] is None
    assert problem.reference_convention["absolute_physiological_timing_established"] is False


def test_centered_lag_draws_and_covariance_ignore_common_lag_shift():
    model, problem, _ = prepared()
    point = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            point[i] = 0.7
    indices = [problem.indices[("filter", m, "lag")] for m in model.responses]
    point[indices] = [0.4, -0.2]
    draws = np.broadcast_to(point, (2, 3, len(point))).copy()
    draws[..., indices] += np.array([[0.0, 0.1, -0.1], [0.2, 0.3, 0.0]])[..., None]
    model.problem_, model.parameter_draws_ = problem, draws
    relative = model.relative_lag_draws()
    assert_allclose(relative["brain"], 0.3)
    assert_allclose(relative["physio"], -0.3)
    assert relative["brain"].shape == (2, 3)
    assert not relative["brain"].flags.writeable
    shifted = point.copy()
    shifted[indices] += 0.25
    assert_allclose(
        problem.covariance(point, "train"),
        problem.covariance(shifted, "train"),
        atol=1e-12,
    )
    assert_allclose(problem.nll(point), problem.nll(shifted), atol=1e-10)


def test_omitted_multifactor_anchors_allow_rank_deficient_loading_block():
    from multimodalsrm.bayesian.multifactor import orientation

    _, problem, _ = prepared(features=2)
    point = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            point[i] = 0.0
    convention = orientation(problem, point, None)
    assert convention["method"] == "optimizer_coordinates"
    assert convention["anchors"] == []
    assert_array_equal(convention["rotation"], np.eye(2))


def test_spectral_common_offset_metadata_acknowledges_boundary_dependence():
    from multimodalsrm.bayesian import SpectralConfig

    _, problem, _ = prepared(
        linear_algebra="spectral", spectral=SpectralConfig(rank=16, padding=1.0)
    )
    point = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            point[i] = 0.7
    indices = [problem.indices[("filter", m, "lag")] for m in problem.responses]
    point[indices] = [0.4, -0.2]
    shifted = point.copy()
    shifted[indices] += 0.25
    # The finite, bounded sine basis is not exactly stationary.
    assert not np.isclose(problem.nll(point), problem.nll(shifted))
    assert problem.reference_convention["common_lag_offset"] == (
        "prior_and_spectral_boundary_dependent"
    )
    assert not problem.reference_convention["absolute_physiological_timing_established"]


@pytest.fixture(scope="module", params=[1, 2])
def fitted_optional(request):
    _, model, data = setup(request.param)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    return model, data


def test_optional_conventions_survive_fit_archive_and_independent_readout(
    fitted_optional, tmp_path
):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    model, data = fitted_optional
    assert model.anchor is None and model.factor_anchors is None
    assert model.reference_modality is None
    path = tmp_path / "optional.zip"
    save_model(path, model)
    restored, _ = load_model(path)
    assert restored.get_params() == model.get_params()
    q = np.array([3.0, 5.0, 7.0])
    donor = {s: {"heldout": copy.deepcopy(r["train"])} for s, r in data.items()}
    # Excluding the auto sign-anchor participant must not change conventions.
    donor = {"b": donor["b"]}
    before = model.transform(donor, times=q)["b"]["heldout"]
    after = restored.transform(donor, times=q)["b"]["heldout"]
    assert_array_equal(before.values, after.values)
    assert_array_equal(before.variance, after.variance)
    assert_array_equal(before.valid, after.valid)
    assert before.metadata["reference_convention"] == model.configuration_["reference_convention"]
    assert restored.configuration_ == model.configuration_


def test_fit_quality_reports_centered_lags_without_a_reference(fitted_optional):
    from multimodalsrm.bayesian.quality import summarize_fit

    model, _ = fitted_optional
    report = summarize_fit(
        dict(
            configuration=model.configuration_,
            parameter_names=model.parameter_names_,
            map=model.map_diagnostics_,
            restarts=model.restart_diagnostics_,
        )
    )
    assert report["availability"] == "complete"
    assert report["reference"]["relative_lag_definition"] == "modality_lag_minus_mean_lag"
    assert_allclose(sum(row["peak_delay_seconds"] for row in report["filters"]), 0, atol=1e-12)
    assert len(report["relative_delays"]) == 1


def test_calibration_preserves_optional_training_conventions(fitted_optional):
    from multimodalsrm.bayesian.calibration import prepare

    model, data = fitted_optional
    new = {"0-new": copy.deepcopy(data["a"])}
    _, _, problem = prepare(model, new)
    assert problem.joint.anchor == model.problem_.anchor
    assert problem.joint.reference_convention == model.problem_.reference_convention
    for name in problem.names:
        if name[0] == "loading":
            assert problem.bounds[problem.indices[name]] == (-np.inf, np.inf)


def test_explicit_factor_anchors_can_supply_first_anchor_implicitly():
    _, model, data = setup(features=2)
    model.set_params(factor_anchors=(("b", "brain", 0), ("a", "physio", 0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert model.anchor is None
    assert model.problem_.anchor == ("b", "brain", 0)
    assert model.factor_orientation_["method"] == "training_anchor_QR_positive_diagonal"
