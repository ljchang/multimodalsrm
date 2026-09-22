"""R starts must preserve GP units, masks, run sharing and search contracts."""

import copy
import warnings
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    SearchConfig,
    _archive,
    fitting,
    r_initialization,
)
from multimodalsrm.bayesian.persistence import (
    _model_state,
    _restore_model,
    _search_for_validation,
)

from .test_bayesian_multifactor import fixture
from .test_bayesian_problem import problem_fixture


@pytest.mark.parametrize("value", [None, 0, 1, "True", np.bool_(True)])
def test_r_init_requires_boolean(value):
    assert SearchConfig().r_init is False
    with pytest.raises(ValueError, match="r_init"):
        SearchConfig(r_init=value)


@pytest.mark.parametrize("features", [1, 2])
def test_translation_preserves_units_masks_runs_and_anchor(features):
    problem, _, _ = fixture(features=features, gaussian=False)
    # Deliberately unequal feature units and an off-mask outlier. The second
    # run is present for only one participant and must still enter the scale.
    paths = {
        "first": np.array([[-0.5, -0.25], [0.5, 0.25]])[:, :features],
        "second": np.array([[-1.0, -0.5], [1.0, 0.5]])[:, :features],
    }
    latent_scale = np.array([np.sqrt(0.625), np.sqrt(0.15625)])[:features]
    fitted = SimpleNamespace(
        preprocessing_={},
        loadings_={},
        _training_latent_arrays_={},
        run_grids_={run: np.arange(2) for run in paths},
        run_domains_={run: (0, 1) for run in paths},
        observation_blocks_=[],
        group_kernels_={},
    )
    expected_noise = {}
    for subject, modality in problem.groups:
        n = len(problem.adapter.preprocessing_[subject][modality]["mean"])
        units = np.arange(1, n + 1, dtype=float)
        loading = np.full((n, features), -0.2)
        fitted.preprocessing_.setdefault(subject, {})[modality] = dict(
            mean=np.arange(n) * 0.1,
            scale=units,
        )
        fitted.loadings_.setdefault(subject, {})[modality] = loading
        runs = paths if subject == "a" else {"first": paths["first"]}
        fitted._training_latent_arrays_[subject] = runs
        squared, count = 0.0, 0
        for run, latent in runs.items():
            residual = np.full((2, n), 0.2 if run == "first" else 0.4)
            mask = np.ones((2, n), bool)
            mask[0, -1] = False
            squared += np.sum(((residual * units)[mask]) ** 2)
            count += mask.sum()
            residual[0, -1] = 1e20
            fitted.observation_blocks_.append(
                SimpleNamespace(
                    subject=subject,
                    modality=modality,
                    run=run,
                    H=np.eye(2),
                    values=latent @ loading.T + residual,
                    mask=mask,
                    valid=np.ones(2, bool),
                )
            )
        expected_noise[subject, modality] = squared / count
    point = r_initialization._translate(problem, fitted)
    for subject, modality, feature in problem.keys:
        key = subject, modality, feature
        assert_allclose(point[problem.indices[("offset", *key)]], feature * 0.1)
        if features == 1:
            assert_allclose(
                point[problem.indices[("loading", *key)]],
                0.2 * (feature + 1) * latent_scale[0],
            )
        else:
            for factor in range(features):
                assert_allclose(
                    point[problem.indices[("loading", *key, factor)]],
                    -0.2 * (feature + 1) * latent_scale[factor],
                )
    for group, variance in expected_noise.items():
        assert_allclose(point[problem.indices[("noise", *group)]], variance)
    assert np.isfinite(problem.to_unconstrained(point)).all()


@pytest.mark.parametrize("kind", ["multifactor", "scalar_irregular"])
def test_actual_r_fit_transfers_responses_and_is_reproducible(kind):
    problem, _, _ = (
        fixture(algebra="grouped")
        if kind == "multifactor"
        else problem_fixture(gaussian=True, two_runs=True)
    )
    first, report = r_initialization.r_initial_point(problem, 12)
    second, _ = r_initialization.r_initial_point(problem, 12)
    assert report["r_init"] == "used"
    assert 0 < report["iterations"] <= 40
    assert report["elapsed_seconds"] > 0
    assert_array_equal(first, second)
    assert np.isfinite(problem.value_gradient(first)[1]).all()
    for name, index in problem.indices.items():
        if name[0] == "filter":
            lo, hi = problem.bounds[index]
            assert lo < first[index] < hi


def test_bach_initialization_fixes_only_preliminary_response_parameters():
    from .test_bayesian_bach_posterior import prepared_bach

    _, problem, _, _ = prepared_bach(3, order=16)
    response = problem.responses["signal"]
    free = response.free_parameters
    point, report = r_initialization.r_initial_point(problem, 1)
    assert report["r_init"] == "used"
    assert report["fixed_response_modalities"] == ["signal"]
    assert problem.responses["signal"] is response
    assert response.free_parameters == free
    for name in free:
        index = problem.indices[("filter", "signal", name)]
        prior = problem.parameter_priors[index]
        expected = np.clip(
            response.initial_kernel().parameters[name], prior.ppf(1e-4), prior.ppf(1 - 1e-4)
        )
        assert_allclose(point[index], expected, rtol=1e-14, atol=1e-15)


def test_search_changes_only_first_start_and_false_never_fits_r(monkeypatch):
    problem, _, _ = fixture(algebra="grouped", gaussian=False)
    config = SearchConfig(starts=3, maxiter=2, r_init=True)
    historical = fitting.initial_points(problem, config.starts, 7)
    point = problem.initial.copy()
    calls = []

    def initialize(problem, seed):
        calls.append(seed)
        return point.copy(), dict(method="r_msrm", r_init="used")

    monkeypatch.setattr(r_initialization, "r_initial_point", initialize)
    _, records = fitting.search(problem, config, 7)
    assert calls == [7]
    assert_array_equal(records[0]["initial_parameters"], point)
    assert records[0]["initialization"]["r_init"] == "used"
    assert_array_equal([r["initial_parameters"] for r in records[1:]], historical[1:])
    _, old_records = fitting.search(problem, replace(config, r_init=False), 7)
    assert calls == [7]
    assert_array_equal([r["initial_parameters"] for r in old_records], historical)
    assert "initialization" not in old_records[0]


def test_failed_r_fit_keeps_historical_start_with_diagnostic(monkeypatch):
    problem, _, _ = problem_fixture()

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic singular fit")

    monkeypatch.setattr(r_initialization.MultimodalSRM, "fit", fail)
    with pytest.warns(RuntimeWarning, match="using the data-based GP start"):
        best, records = fitting.search(problem, SearchConfig(starts=1, maxiter=2, r_init=True), 9)
    assert best is not None
    assert_array_equal(records[0]["initial_parameters"], fitting.initial_points(problem, 1, 9)[0])
    assert records[0]["initialization"]["r_init"] == "fallback"
    assert "synthetic singular fit" in records[0]["initialization"]["reason"]


def test_finite_objective_with_invalid_gradient_falls_back(monkeypatch):
    problem, _, _ = problem_fixture()
    monkeypatch.setattr(problem, "value_gradient", lambda point: (0.0, np.full_like(point, np.nan)))
    with pytest.warns(RuntimeWarning, match="nonfinite GP objective or gradient"):
        point, report = r_initialization.r_initial_point(problem, 1)
    assert point is None
    assert report["r_init"] == "fallback"


@pytest.mark.parametrize("target", ["conditional", "updated"])
def test_nontraining_targets_do_not_fit_r(target, monkeypatch):
    problem, _, _ = problem_fixture()
    if target == "conditional":
        problem._full_training_target = False
    else:
        problem._posterior_update = {}

    def forbidden(*args, **kwargs):
        pytest.fail("R fit must not run on conditional or updated targets")

    monkeypatch.setattr(r_initialization.MultimodalSRM, "fit", forbidden)
    point, report = r_initialization.r_initial_point(problem, 1)
    assert point is None
    assert report["r_init"] == "skipped"


def test_old_search_records_retain_disabled_initialization():
    config = SearchConfig(starts=2, r_init=False)
    encoded = _archive.encode(config, {})
    del encoded["fields"]["r_init"]
    restored = _archive.decode(encoded, {}, set())
    assert restored == config
    legacy = asdict(config)
    del legacy["r_init"]
    assert _search_for_validation(legacy) == asdict(config)
    assert "r_init" not in legacy


@pytest.mark.parametrize("r_init", [True, False])
def test_map_archive_replays_initialization_setting_and_evidence(r_init):
    problem, _, data = problem_fixture(two_runs=True)
    model = BayesianMultimodalSRM(
        priors=problem.priors,
        anchor=problem.anchor,
        inference="map",
        search=SearchConfig(starts=1, maxiter=2, r_init=r_init),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(data)
    state = copy.deepcopy(_model_state(model))
    restored, _ = _restore_model(state)
    assert restored.search_config_.r_init is r_init
    assert restored.restart_diagnostics_ == model.restart_diagnostics_
    assert_array_equal(restored.map_parameters_, model.map_parameters_)
    if not r_init:
        del state["fit"]["configuration"]["search"]["r_init"]
        # Emulate an old explicit SearchConfig already decoded with defaults.
        restored, _ = _restore_model(state)
        assert restored.search_config_.r_init is False
        assert "r_init" not in restored.configuration_["search"]


def test_legacy_implicit_search_default_is_restored():
    from multimodalsrm.bayesian.persistence import _restore_search_default

    constructor = dict(search=None)
    fit = dict(configuration=dict(search={}))
    _restore_search_default(constructor, fit)
    assert constructor["search"] == SearchConfig(r_init=False)
    assert fit == dict(configuration=dict(search={}))
