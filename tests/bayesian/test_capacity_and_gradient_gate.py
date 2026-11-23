"""The capacity guard bounds what each backend factorizes; the MAP gate is scale-aware."""

import numpy as np
import pytest

from multimodalsrm import Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.fitting import gradient_gate
from multimodalsrm.bayesian.observation_adapter import capacity_unit
from multimodalsrm.bayesian.persistence import (
    _prepare,
    _restore_search_default,
    _search_for_validation,
)


def recording(subjects=3, rows=60, features=5):
    """900 eligible scalars on 60 brain nodes plus 30 rating nodes."""
    rng = np.random.default_rng(4)
    times = np.arange(rows) * 1.5
    data = {}
    for index in range(subjects):
        data[f"s{index}"] = {"run": {"brain": TimeSeries(rng.normal(size=(rows, features)), times)}}
    data["s0"]["run"]["rating"] = TimeSeries(rng.normal(size=(rows // 2, 1)), times[::2] + 0.25)
    return data


def model(algebra, **settings):
    options = dict(
        priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0)),
        responses={
            m: Response(Identity(), estimate=False, pooling="shared") for m in ("brain", "rating")
        },
        inference="map",
        linear_algebra=algebra,
    )
    options.update(settings)
    return BayesianMultimodalSRM(**options)


def test_capacity_unit_follows_the_factorized_system():
    positive, admits_zero = Prior.lognormal(0.0, 1.0), Prior("normal", 0.5, 0.5, lower=0.0)
    assert capacity_unit("dense", noise_prior=positive) == "scalars"
    assert capacity_unit("spectral", noise_prior=positive) == "scalars"
    assert capacity_unit("grouped", noise_prior=positive) == "functionals"
    assert capacity_unit("state_space", noise_prior=positive) == "nodes"
    # Scalar Kalman updates and dense auxiliary terms keep the scalar count.
    assert capacity_unit("state_space", noise_prior=admits_zero) == "scalars"
    assert capacity_unit("state_space") == "scalars"
    assert capacity_unit("grouped", run_baseline_sd={"brain": 1.0}) == "scalars"
    assert capacity_unit("grouped", noise_timescales={"brain": 2.0}) == "scalars"


def test_dense_guard_still_counts_scalars_and_names_the_alternatives():
    with pytest.raises(ValueError, match="eligible scalar observations.*max_observations=800"):
        _prepare(model("dense"), recording())
    with pytest.raises(ValueError, match="grouped"):
        _prepare(model("dense"), recording())


@pytest.mark.parametrize("algebra", ["grouped", "state_space"])
def test_node_backends_admit_many_scalars_on_few_nodes(algebra):
    adapter, problem = _prepare(model(algebra), recording())
    assert sum(len(system.times) for system in problem.systems.values()) == 930
    assert len(problem.grouped_systems["run"].times) == 90
    assert adapter.capacity_unit == ("functionals" if algebra == "grouped" else "nodes")


def test_grouped_guard_counts_nodes_times_factors():
    data = recording()
    _prepare(model("grouped", features=2, max_observations=180), data)
    with pytest.raises(ValueError, match=r"180 grouped functionals.*max_observations=179"):
        _prepare(model("grouped", features=2, max_observations=179), data)


def test_state_space_guard_counts_nodes_once_across_participants_and_factors():
    data = recording()
    _prepare(model("state_space", features=2, max_observations=90), data)
    with pytest.raises(ValueError, match=r"90 modality/time nodes.*max_observations=89"):
        _prepare(model("state_space", features=2, max_observations=89), data)


def test_masked_rows_and_distinct_clocks_define_nodes():
    data = recording()
    brain = data["s1"]["run"]["brain"]
    mask = np.ones_like(brain.values, dtype=bool)
    mask[5] = False  # Still a node: other participants observe this timestamp.
    data["s1"]["run"]["brain"] = TimeSeries(brain.values, brain.times, mask)
    shifted = data["s2"]["run"]["brain"]
    data["s2"]["run"]["brain"] = TimeSeries(shifted.values, shifted.times + 0.5)
    _, problem = _prepare(model("state_space", max_observations=150), data)
    assert len(problem.grouped_systems["run"].times) == 150
    with pytest.raises(ValueError, match="150 modality/time nodes"):
        _prepare(model("state_space", max_observations=149), data)


def test_scalar_state_space_and_baselines_keep_the_scalar_guard():
    zero = BayesianPriors(noise=Prior("normal", 0.5, 0.5, lower=0.0))
    with pytest.raises(ValueError, match="eligible scalar observations"):
        _prepare(model("state_space", priors=zero), recording())
    with pytest.raises(ValueError, match="eligible scalar observations"):
        _prepare(model("grouped", run_baseline_sd={"brain": 1.0}), recording())


def test_default_response_pooling_error_names_the_modality_and_fix():
    responses = {"brain": Response(Identity(), estimate=False)}
    data = {s: {"run": {"brain": runs["run"]["brain"]}} for s, runs in recording().items()}
    with pytest.raises(ValueError, match=r"responses\['brain'\].*pooling='shared'"):
        _prepare(model("grouped", responses=responses), data)


def test_gradient_gate_criteria():
    config = SearchConfig()
    assert gradient_gate(5e-4, 3.0, config) == (True, pytest.approx(5e-4 / 3), "absolute")
    # Above the absolute limit but at rounding scale for a large objective.
    meets, relative, criterion = gradient_gate(0.037, 858_992.0, config)
    assert meets and criterion == "relative_to_objective"
    assert relative == pytest.approx(0.037 / 858_992.0)
    # A stalled start: large gradient at a comparable objective.
    assert gradient_gate(60_800.0, 1_195_617.0, config)[::2] == (False, None)
    assert gradient_gate(0.037, -858_992.0, config)[0]
    assert gradient_gate(0.0, 0.0, config) == (True, float("inf"), "absolute")
    absolute_only = SearchConfig(relative_gradient_tolerance=0.0)
    assert gradient_gate(0.037, 858_992.0, absolute_only)[::2] == (False, None)
    assert gradient_gate(5e-4, 858_992.0, absolute_only)[2] == "absolute"


@pytest.mark.parametrize("value", [-1e-9, np.inf, np.nan, True, "1e-6"])
def test_relative_gradient_tolerance_validation(value):
    with pytest.raises(ValueError, match="relative_gradient_tolerance"):
        SearchConfig(relative_gradient_tolerance=value)


def test_map_diagnostics_report_the_deciding_criterion():
    data = recording()
    fitted = model("state_space", search=SearchConfig(starts=1, maxiter=400), random_state=3).fit(
        data
    )
    diagnostics = fitted.map_diagnostics_
    assert diagnostics["meets_gradient_tolerance"]
    assert diagnostics["gradient_criterion"] in ("absolute", "relative_to_objective")
    assert diagnostics["relative_projected_gradient"] == pytest.approx(
        diagnostics["physical_projected_gradient"] / abs(diagnostics["objective"])
    )
    assert fitted.configuration_["search"]["relative_gradient_tolerance"] == 1e-6


def test_historical_archives_keep_the_absolute_only_gate():
    from dataclasses import asdict

    historical = asdict(SearchConfig())
    del historical["relative_gradient_tolerance"]
    assert _search_for_validation(historical)["relative_gradient_tolerance"] == 0.0
    constructor = _restore_search_default(
        {"search": None}, {"configuration": {"search": historical}}
    )
    assert constructor["search"] == SearchConfig(relative_gradient_tolerance=0.0)
    current = {"configuration": {"search": asdict(SearchConfig())}}
    assert _restore_search_default({"search": None}, current)["search"] is None
