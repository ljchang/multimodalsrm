"""NUTS geometry is explicit, guarded, and reported from adapted state."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from .test_bayesian_model import make_model
from .test_bayesian_problem import api, independent_parameters, problem_fixture


def _sampling_reference():
    problem, _, _ = problem_fixture(True)
    parameters = independent_parameters(problem)
    record = dict(
        start=0,
        parameters=parameters.tolist(),
        objective=float(problem.objective(parameters)),
    )
    return problem, [record]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass_matrix", "full", "mass_matrix"),
        ("max_dense_parameters", 0, "max_dense_parameters"),
        ("max_dense_parameters", True, "max_dense_parameters"),
    ],
)
def test_sampler_geometry_configuration_rejects_invalid_values(field, value, message):
    b = api()
    with pytest.raises(ValueError, match=message):
        b.SamplerConfig(**{field: value})


def test_legacy_positional_sampler_arguments_keep_their_original_order():
    b = api()
    config = b.SamplerConfig(2, 30, 40, 0.95, 7, "vectorized", 3.5, 0.2)

    assert config.chains == 2
    assert config.warmup == 30
    assert config.draws == 40
    assert config.target_accept == 0.95
    assert config.max_tree_depth == 7
    assert config.chain_method == "vectorized"
    assert config.start_objective_window == 3.5
    assert config.start_jitter == 0.2
    assert config.mass_matrix == "dense"
    assert config.max_dense_parameters == 1024


def test_diagonal_public_fit_ignores_dense_guard_and_retains_raw_draws():
    b = api()
    config = b.SamplerConfig(
        chains=2,
        warmup=12,
        draws=12,
        max_tree_depth=6,
        mass_matrix="diagonal",
        max_dense_parameters=1,
    )
    model, data = make_model("posterior", sampler=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)

    assert model.parameter_draws_.shape[:2] == (2, 12)
    assert not model.sampling_diagnostics_["passes"]
    metric = model.sampling_diagnostics_["metric"]
    assert metric["requested"] == metric["actual"] == "diagonal"
    assert metric["active_dimension"] == len(model.parameter_names_)
    assert metric["adapted_inverse_mass_matrix_shape"] == [
        2,
        len(model.parameter_names_),
    ]
    assert metric["estimated_bytes_per_chain"] == 8 * len(model.parameter_names_)


@pytest.mark.parametrize(
    ("mass_matrix", "chain_method", "shape_suffix"),
    [
        ("diagonal", "vectorized", (2,)),
        ("dense", "sequential", (2, 2)),
    ],
)
def test_real_sampler_reports_actual_adapted_metric_shape(mass_matrix, chain_method, shape_suffix):
    from multimodalsrm.bayesian.fitting import sample

    b = api()
    problem, records = _sampling_reference()
    config = b.SamplerConfig(
        chains=2,
        warmup=8,
        draws=8,
        max_tree_depth=5,
        chain_method=chain_method,
        mass_matrix=mass_matrix,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws, _, _, diagnostics = sample(problem, records, config, 941, blocks=("filter",))

    assert draws.shape[:2] == (2, 8)
    metric = diagnostics["metric"]
    assert metric["requested"] == mass_matrix
    assert metric["actual"] == mass_matrix
    assert metric["active_dimension"] == 2
    assert tuple(metric["adapted_inverse_mass_matrix_shape"]) == (2, *shape_suffix)
    assert metric["estimated_bytes_per_chain"] == 8 * int(np.prod(shape_suffix))
    assert metric["adapted_values_stored"] is True
    adaptation = diagnostics["adaptation"]
    step_size = np.asarray(adaptation["step_size"])
    inverse_mass = np.asarray(adaptation["inverse_mass_matrix"])
    assert adaptation["values_stored"] is True
    assert tuple(adaptation["step_size_shape"]) == step_size.shape == (2,)
    assert tuple(adaptation["inverse_mass_matrix_shape"]) == inverse_mass.shape
    assert inverse_mass.shape == (2, *shape_suffix)
    assert np.isfinite(step_size).all() and np.all(step_size > 0)
    assert np.isfinite(inverse_mass).all()
    if mass_matrix == "diagonal":
        assert np.all(inverse_mass > 0)
    else:
        assert np.all(np.linalg.eigvalsh(inverse_mass) > 0)


def test_sampler_separates_initialization_and_transition_randomness():
    from multimodalsrm.bayesian.fitting import sample

    config = api().SamplerConfig(
        chains=2,
        warmup=12,
        draws=12,
        mass_matrix="diagonal",
        max_tree_depth=5,
    )
    problem, records = _sampling_reference()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = sample(problem, records, config, 51)
        explicit = sample(problem, records, config, 51, initialization_seed=51)
        other_stream = sample(problem, records, config, 52, initialization_seed=51)
        other_start = sample(problem, records, config, 51, initialization_seed=53)

    np.testing.assert_array_equal(old[0], explicit[0])
    np.testing.assert_array_equal(
        old[3]["initial_unconstrained"],
        other_stream[3]["initial_unconstrained"],
    )
    assert not np.array_equal(explicit[0], other_stream[0])
    assert not np.array_equal(
        old[3]["initial_unconstrained"],
        other_start[3]["initial_unconstrained"],
    )
    assert old[3]["initialization_seed"] == old[3]["sampler_seed"] == 51
    assert explicit[3]["initialization_seed"] == 51
    assert other_stream[3]["initialization_seed"] == 51
    assert other_stream[3]["sampler_seed"] == 52


@pytest.mark.parametrize("initialization_seed", [True, -1, 1.5, 2**32])
def test_invalid_initialization_seed_fails_before_nuts_construction(
    monkeypatch, initialization_seed
):
    import numpyro.infer

    from multimodalsrm.bayesian.fitting import sample

    problem, records = _sampling_reference()

    def forbidden_nuts(*args, **kwargs):
        raise AssertionError("initialization seed validation must precede NUTS")

    monkeypatch.setattr(numpyro.infer, "NUTS", forbidden_nuts)
    with pytest.raises(ValueError, match="initialization_seed"):
        sample(
            problem,
            records,
            api().SamplerConfig(chains=2, warmup=8, draws=8),
            51,
            initialization_seed=initialization_seed,
        )


def test_large_adapted_state_records_shapes_without_large_value_lists():
    from multimodalsrm.bayesian.fitting import _adaptation_diagnostics

    dimension = 72_000
    adapted_state = SimpleNamespace(
        step_size=np.ones(2),
        inverse_mass_matrix=np.ones((2, dimension)),
    )
    metric, adaptation = _adaptation_diagnostics(
        adapted_state,
        np.zeros((2, dimension)),
        dimension,
        "diagonal",
    )

    assert metric["adapted_values_stored"] is False
    assert adaptation == {
        "values_stored": False,
        "omission_reason": "active_dimension_exceeds_64",
        "step_size_shape": [2],
        "step_size_dtype": "float64",
        "inverse_mass_matrix_shape": [2, dimension],
        "inverse_mass_matrix_dtype": "float64",
    }


def test_dense_dimension_guard_runs_before_nuts_construction(monkeypatch):
    import numpyro.infer

    from multimodalsrm.bayesian.fitting import sample

    b = api()
    problem, records = _sampling_reference()
    config = b.SamplerConfig(
        chains=2,
        warmup=8,
        draws=8,
        mass_matrix="dense",
        max_dense_parameters=1,
    )

    def forbidden_nuts(*args, **kwargs):
        raise AssertionError("dense guard must run before NUTS construction")

    monkeypatch.setattr(numpyro.infer, "NUTS", forbidden_nuts)
    with pytest.raises(ValueError, match="max_dense_parameters"):
        sample(problem, records, config, 942, blocks=("filter",))
