"""A shared GP timescale is a physical posterior coordinate, not a plug-in fit."""

import copy
import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import lognorm, multivariate_normal

from multimodalsrm.bayesian import BayesianProblem, Prior
from multimodalsrm.bayesian.blocks import ParameterSubspace

from .test_bayesian_multifactor import fixture
from .test_bayesian_posterior_updates import prepared
from .test_bayesian_trajectories import integrated_temporal, reference_joint

LENGTH_PRIOR = Prior.lognormal(np.log(2.0), 0.5).bounded(0.5, 6.0)
NAME = ("gp", "length_scale")


def learned_problem(features=3, algebra="grouped", prior=LENGTH_PRIOR, gaussian=True):
    assert "length_scale_prior" in inspect.signature(BayesianProblem).parameters, (
        "learned GP timescale support missing"
    )
    base, x, data = fixture(features, algebra, gaussian=gaussian)
    base.adapter.length_scale = float(prior.ppf(0.5))
    p = BayesianProblem(
        base.adapter,
        base.priors,
        anchor=base.anchor,
        linear_algebra=algebra,
        length_scale_prior=prior,
    )
    return p, np.append(x, 1.75), base, data


def independent_nll(p, x, run="train"):
    """Named physical values and numerical convolution, independent of JAX."""
    pars = dict(zip(p.names, x, strict=True))
    ell = pars.get(NAME, p.length_scale)
    system = p.systems[run]
    w = np.array([[pars[("loading", *key, f)] for f in range(p.features)] for key in system.keys])
    offset = np.array([pars[("offset", *key)] for key in system.keys])
    noise = np.array([pars[("noise", *key[:2])] for key in system.keys])
    parameters = {}
    for modality, response in p.responses.items():
        kernel = response.initial_kernel()
        parameters[modality] = tuple(
            pars.get(("filter", modality, name), getattr(kernel, name, 0.0))
            for name in ("width", "lag")
        )
    temporal = np.empty((len(offset), len(offset)))
    for i, (ta, ka) in enumerate(zip(system.times, system.keys)):
        wa, la = parameters[ka[1]]
        for j, (tb, kb) in enumerate(zip(system.times, system.keys)):
            wb, lb = parameters[kb[1]]
            temporal[i, j] = integrated_temporal(
                float(abs(ta - tb - la + lb)), float(wa), float(wb), float(ell)
            )
    C = temporal * (w @ w.T) + np.diag(noise)
    return -multivariate_normal.logpdf(system.values, mean=offset, cov=C)


@pytest.mark.parametrize("features", [3, 5])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_density_and_lengthscale_gradient_match_independent_integrals(features, algebra):
    p, x, base, _ = learned_problem(features, algebra)
    assert p.names == [*base.names, NAME]
    assert p.parameter_priors[-1] == LENGTH_PRIOR
    distribution = lognorm(LENGTH_PRIOR.scale, scale=np.exp(LENGTH_PRIOR.loc))
    normalizer = distribution.cdf(6.0) - distribution.cdf(0.5)

    def gp_log_prior(value):
        return distribution.logpdf(value) - np.log(normalizer)

    assert_allclose(p.nll(x), independent_nll(p, x), atol=2e-9, rtol=1e-10)
    assert_allclose(p.log_prior(x), base.log_prior(x[:-1]) + gp_log_prior(x[-1]), atol=1e-10)
    value, gradient = p.value_gradient(x)
    assert_allclose(value, independent_nll(p, x) - p.log_prior(x), atol=2e-9)
    h = 2e-5
    plus, minus = x.copy(), x.copy()
    plus[-1] += h
    minus[-1] -= h
    expected = (
        (independent_nll(p, plus) - gp_log_prior(plus[-1]))
        - (independent_nll(p, minus) - gp_log_prior(minus[-1]))
    ) / (2 * h)
    assert_allclose(gradient[-1], expected, rtol=2e-5, atol=2e-5)
    assert abs(gradient[-1]) > 1e-3


def test_timescale_changes_cached_grouped_kernel_and_preserves_haar_symmetry():
    from multimodalsrm.bayesian.orthogonal import (
        validate_orthogonal_target,
    )

    p, x, _, _ = learned_problem(gaussian=False)
    assert p.grouped_systems["train"].covariance_pairs is not None
    other = x.copy()
    other[-1] = 4.0
    assert abs(float(p.nll(x) - p.nll(other))) > 1.0
    rotation, _ = np.linalg.qr(np.random.default_rng(715).normal(size=(3, 3)))
    nw = len(p.keys) * p.features
    rotated = x.copy()
    rotated[:nw] = (x[:nw].reshape(-1, 3) @ rotation).ravel()
    assert_allclose(p.objective(rotated), p.objective(x), atol=1e-9)
    validate_orthogonal_target(ParameterSubspace(p, x))
    assert rotated[-1] == x[-1]


def test_gp_only_parameter_subspace_retains_physical_density_and_transform():
    p, x, _, _ = learned_problem()
    space = ParameterSubspace(p, x, blocks=("gp",))
    assert space.active_parameters == [["gp", "length_scale"]]
    z = space.to_unconstrained(x)
    physical, jacobian = space.from_unconstrained(z)
    assert_allclose(physical, x, atol=1e-12)
    assert_allclose(space.potential(z), p.objective(x) - jacobian, atol=1e-10)
    assert ["gp", "length_scale"] not in [v["name"] for v in space.fixed_parameters]


@pytest.mark.parametrize(
    "prior",
    [
        Prior.lognormal(0, 1),
        Prior.normal(2, 1),
        Prior.uniform(0, 5),
        Prior.uniform(-1, 5),
    ],
)
def test_timescale_prior_requires_finite_positive_domain(prior):
    with pytest.raises(ValueError, match="length_scale.*positive.*finite"):
        learned_problem(prior=prior)


def test_covariance_accuracy_is_checked_at_smallest_allowed_timescale():
    # The median is safe, but the full domain reaches unsupported cancellation.
    prior = Prior.lognormal(np.log(2), 0.2).bounded(0.01, 6)
    with pytest.raises(ValueError, match="covariance accuracy"):
        learned_problem(prior=prior)


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_queries_and_joint_paths_use_each_draws_timescale(algebra):
    from multimodalsrm.bayesian.posterior_coordinates import (
        metadata,
        rotations,
    )
    from multimodalsrm.bayesian.trajectories import joint_moments

    p, x, _, _ = learned_problem(3, algebra)
    model = prepared(3, algebra)
    model.problem_, model.adapter_ = p, p.adapter
    model.parameter_names_ = p.names
    other = x.copy()
    other[-1] = 4.0
    model.parameter_draws_ = np.array([[x], [other]])
    model.configuration_.update(
        uncertainty="parameter_posterior_mixture",
        reference_convention=p.reference_convention,
        factor_orientation=metadata(model._factor_anchor_keys_),
        gp_hyperparameters={"length_scale": {"parameter": list(NAME)}},
    )
    model.sampling_diagnostics_ = {"passes": False}
    query = np.array([4.0, 4.6, 7.1])
    marginal = model.infer_latent(times=query)["train"]
    Q = rotations(p, [x, other], model._factor_anchor_keys_, [(0, 0), (1, 0)])
    evaluate = joint_moments(p, "train", query)
    for i, values in enumerate([x, other]):
        independent_model = copy.copy(model)
        independent_model.problem_ = copy.copy(p)
        independent_model.problem_.length_scale = values[-1]
        mean, covariance = reference_joint(independent_model, values, query)
        actual_mean, actual_covariance = evaluate(values, Q[i])
        assert_allclose(actual_mean, mean, atol=2e-9)
        assert_allclose(actual_covariance, covariance, atol=2e-9)
        assert_allclose(marginal.component_means[i], mean, atol=2e-9)
        assert_allclose(
            marginal.component_variances[i],
            np.diag(covariance).reshape(3, 3),
            atol=2e-9,
        )
    assert np.max(np.abs(marginal.component_means[0] - marginal.component_means[1])) > 0.01
    paths = model.sample_latent(times=query, max_draws=None, random_state=45)["train"]
    assert paths.samples.shape == (2, 1, 3, 3)
    assert paths.metadata["gp_hyperparameters"] == model.configuration_["gp_hyperparameters"]
    assert "fixed_length_scale_seconds" not in paths.metadata["reference_convention"]
    assert_array_equal(model.parameter_draws_, np.array([[x], [other]]))


@pytest.mark.parametrize(
    "options",
    [
        {"linear_algebra": "state_space"},
        {"linear_algebra": "spectral"},
        {"run_baseline_sd": {"brain": 0.2}},
        {"noise_timescales": {"brain": 2.0}},
    ],
)
def test_public_api_rejects_unsupported_learned_timescale_before_preparation(options):
    model = prepared()
    model.set_params(length_scale=LENGTH_PRIOR, inference="map", **options)
    with pytest.raises(ValueError, match="learned length_scale requires"):
        model.fit(None)


@pytest.mark.parametrize("features", [3, 5])
@pytest.mark.parametrize("kind", ["donor", "calibration", "participant"])
def test_updates_keep_one_shared_gp_prior_and_use_all_evidence(features, kind):
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
        prepare_participant,
    )
    from multimodalsrm.bayesian.posterior_updates import (
        prepare_condition,
        validate_update_target,
    )

    from .test_bayesian_multifactor import log_prior
    from .test_bayesian_posterior_participants import calibration, physical
    from .test_bayesian_posterior_updates import donors

    model = prepared(features)
    p, x, _, _ = learned_problem(features)
    model.length_scale = LENGTH_PRIOR
    model.problem_ = p
    model.adapter_ = p.adapter
    model.specification_ = model._specification()
    model.parameter_names_ = p.names
    model.map_parameters_ = x
    model.parameter_draws_ = np.broadcast_to(x, (2, 4, len(x))).copy()
    model.training_fit_["map_parameters"] = x
    if kind == "donor":
        result = prepare_condition(model, donors(model), targets={"a": ["aux"]})
    elif kind == "calibration":
        result = prepare_calibration(model, calibration(model))
    else:
        data = {"a": {"new": {"brain": model.training_data_["a"]["train"]["brain"]}}}
        result = prepare_participant(model, data, participant="a")
    values = physical(result, model)
    target = result.problem_
    assert target.names.count(NAME) == 1
    assert target.length_scale_prior == LENGTH_PRIOR
    assert values[target.indices[NAME]] == x[-1]
    expected = sum(independent_nll(target, values, run) for run in target.systems)
    assert_allclose(target.objective(values), expected - log_prior(target, values), atol=3e-8)
    validate_update_target(target)


@pytest.fixture(scope="module", params=[3, 5])
def learned_workflow(request):
    """Tiny execution smoke, deliberately not a convergence qualification."""
    import warnings

    from .test_bayesian_posterior_updates import donors

    model = prepared(request.param)
    model.length_scale = LENGTH_PRIOR
    data = model.training_data_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        donor = model.condition(donors(model), targets={"a": ["aux"]})
        calibration_data = {"newperson": {"train": {"brain": data["a"]["train"]["brain"]}}}
        calibrated = model.calibrate_posterior(calibration_data)
        participant = calibrated.condition_participants(
            {"newperson": {"new": calibration_data["newperson"]["train"]}}
        )["newperson"]
    return dict(training=model, donor=donor, calibration=calibrated, participant=participant)


@pytest.mark.parametrize("kind", ["training", "donor", "calibration", "participant"])
def test_public_workflow_archive_and_joint_queries_keep_sampled_timescale(
    learned_workflow, kind, tmp_path, monkeypatch
):
    from sklearn.base import clone

    from multimodalsrm.bayesian import _archive, workflow

    from .test_bayesian_persistence import assert_result_equal, forbid_inference_fitting

    model = learned_workflow[kind]
    assert clone(model).length_scale == LENGTH_PRIOR
    assert model.specification_["length_scale"] == LENGTH_PRIOR
    index = model.parameter_names_.index(NAME)
    ell = model.parameter_draws_[..., index]
    assert np.isfinite(ell).all() and np.all((ell > 0.5) & (ell < 6))
    assert np.ptp(ell) > 0
    assert list(NAME) in model.sampling_diagnostics_["active_parameters"]
    assert model.sampling_diagnostics_["passes"] is False
    meta = model.configuration_["gp_hyperparameters"]["length_scale"]
    assert meta["parameter"] == list(NAME)
    assert meta["sharing"] == "all_factors_and_runs"
    assert meta["map"] == model.map_parameters_[index]
    assert model.configuration_["latent_variance"] == 1
    assert "fixed_length_scale_seconds" not in model.problem_.reference_convention
    times = {next(iter(model.prediction_runs_)): [7.0, 9.0]}
    run = next(iter(times))
    expected = model.infer_latent(times=times, max_draws=2)[run]
    paths = model.sample_latent(times=times, max_draws=2, random_state=92)[run]
    workflow.save_model(tmp_path / kind, model)
    forbid_inference_fitting(monkeypatch)
    restored, _ = workflow.load_model(tmp_path / kind)
    assert restored.length_scale == LENGTH_PRIOR
    assert _archive.same(restored.configuration_, model.configuration_)
    assert_array_equal(restored.parameter_draws_, model.parameter_draws_)
    assert_result_equal(restored.infer_latent(times=times, max_draws=2)[run], expected)
    replay = restored.sample_latent(times=times, max_draws=2, random_state=92)[run]
    assert_array_equal(replay.samples, paths.samples)
    assert _archive.same(replay.metadata, paths.metadata)


def test_save_rejects_changed_gp_prior(learned_workflow, tmp_path):
    from multimodalsrm.bayesian import workflow

    for kind, original in learned_workflow.items():
        model = copy.deepcopy(original)
        model.problem_.length_scale_prior = Prior.uniform(0.5, 6)
        with pytest.raises(ValueError, match="length_scale_prior|target|configuration"):
            workflow.save_model(tmp_path / kind, model)


def test_learned_timescale_map_freezing_transform_and_calibration(tmp_path):
    import warnings

    from multimodalsrm.bayesian import workflow

    from .test_bayesian_persistence import assert_result_equal
    from .test_bayesian_posterior_updates import donors

    model = prepared(3)
    model.set_params(length_scale=LENGTH_PRIOR, inference="map")
    data = model.training_data_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
        frozen = model.condition(donors(model), targets={"a": ["aux"]}, mode="frozen")
        individual = model.transform(
            {"a": {"new": {"brain": data["a"]["train"]["brain"]}}},
            times={"new": [7.0, 9.0]},
        )["a"]["new"]
        calibration = model.calibrate(
            {"newperson": {"train": {"brain": data["a"]["train"]["brain"]}}}
        )
    index = model.parameter_names_.index(NAME)
    ell = model.map_parameters_[index]
    assert frozen.map_parameters_[index] == ell
    assert frozen.problem_.length_scale_prior == LENGTH_PRIOR
    assert individual.metadata["gp_hyperparameters"]["length_scale"]["map"] == ell
    assert NAME not in calibration.problem_.names
    joint_index = calibration.problem_.joint.indices[NAME]
    assert calibration.problem_.reference[joint_index] == ell
    assert calibration.problem_.joint.length_scale_prior == LENGTH_PRIOR
    assert_allclose(individual.values, frozen.infer_latent(times={"new": [7.0, 9.0]})["new"].values)
    workflow.save_model(tmp_path / "map", model)
    restored, _ = workflow.load_model(tmp_path / "map")
    assert restored.length_scale == LENGTH_PRIOR
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    workflow.save_model(tmp_path / "calibration", calibration)
    restored_calibration, _ = workflow.load_model(tmp_path / "calibration")
    new_data = {"newperson": {"new": {"brain": data["a"]["train"]["brain"]}}}
    times = {"new": [7.0, 9.0]}
    assert_result_equal(
        restored_calibration.transform(new_data, times=times)["newperson"]["new"],
        calibration.transform(new_data, times=times)["newperson"]["new"],
    )
