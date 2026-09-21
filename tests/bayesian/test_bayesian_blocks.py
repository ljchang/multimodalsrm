"""Conditional block inference must retain fixed values and active Jacobians."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .test_bayesian_model import make_model
from .test_bayesian_problem import api, independent_parameters, problem_fixture


def space_class():
    b = api()
    assert hasattr(b, "ParameterSubspace"), "conditional parameter subspace missing"
    return b.ParameterSubspace


def test_conditional_density_uses_only_active_coordinate_jacobians():
    p, _, _ = problem_fixture(True)
    reference = independent_parameters(p)
    space = space_class()(p, reference, blocks=("filter",))
    z = np.array([-0.8, 0.7])
    x, jac = space.from_unconstrained(z)
    active = space.active_indices
    frozen = [i for i in range(len(p.names)) if i not in active]
    assert_array_equal(np.asarray(x)[frozen], reference[frozen])
    sigmoid = 1 / (1 + np.exp(-z))
    expected_jac = np.log(np.array([0.5, 1.0]) * sigmoid * (1 - sigmoid)).sum()
    assert_allclose(jac, expected_jac, atol=1e-12)
    assert_allclose(space.potential(z), p.objective(x) - expected_jac, atol=1e-12)
    assert_allclose(space.to_unconstrained(x), z, atol=1e-12)
    assert not space.reference.flags.writeable


def test_all_blocks_reduce_to_original_posterior():
    p, _, _ = problem_fixture(True)
    x = independent_parameters(p)
    space = space_class()(p, x, blocks=None)
    z = p.to_unconstrained(x)
    assert_allclose(space.potential(z), p.potential(z), atol=1e-12)


def test_batched_coordinates_match_independent_scalar_transforms():
    """A wrong prior grouping, scatter, or Jacobian reduction must fail."""
    p, _, _ = problem_fixture(True)
    reference = independent_parameters(p)
    space = space_class()(p, reference)
    z = np.linspace(-1.25, 1.1, len(space.active_indices))

    expected = reference.copy()
    expected_jacobian = 0.0
    for active_position, physical_index in enumerate(space.active_indices):
        transform = p.transforms[physical_index]
        value = transform(z[active_position])
        expected[physical_index] = value
        expected_jacobian += float(transform.log_abs_det_jacobian(z[active_position], value))

    actual, actual_jacobian = space.from_unconstrained(z)
    assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert_allclose(actual_jacobian, expected_jacobian, rtol=0, atol=1e-12)
    assert_allclose(space.to_unconstrained(expected), z, rtol=0, atol=1e-12)


def test_repeated_prior_transform_trace_does_not_grow_per_coordinate():
    """Repeated priors must trace as one vector transform, not N scalar scatters."""
    b = api()
    from multimodalsrm.bayesian._backend import runtime

    jax, jnp, _, dist = runtime()
    prior = b.Prior.normal()
    transform = dist.transforms.biject_to(dist.constraints.real)

    class RepeatedPriorProblem:
        def __init__(self, size):
            self.initial = np.zeros(size)
            self.names = [("loading", "s", "m", i) for i in range(size)]
            self.parameter_priors = [prior] * size
            self.transforms = [transform] * size

        def objective(self, x):
            return jnp.sum(jnp.asarray(x) ** 2)

    def equation_count(size):
        space = space_class()(RepeatedPriorProblem(size), np.zeros(size))
        traced = jax.make_jaxpr(space.from_unconstrained)(jnp.zeros(size))
        return len(traced.jaxpr.eqns)

    assert equation_count(2048) <= equation_count(32) + 10


@pytest.mark.parametrize("blocks", [(), ("filter", "filter"), ("nonsense",), "filter"])
def test_invalid_block_specifications_are_rejected(blocks):
    p, _, _ = problem_fixture(True)
    with pytest.raises(ValueError, match="blocks"):
        space_class()(p, independent_parameters(p), blocks=blocks)


def test_fixed_boundary_is_not_transformed_or_jittered():
    p, adapter, _ = problem_fixture(True)
    b = api()
    priors = b.BayesianPriors(noise=b.Prior.uniform(0.1, 1.0), filters=p.priors.filters)
    p = b.BayesianProblem(adapter, priors, anchor=p.anchor)
    x = p.initial.copy()
    noise = p.indices[("noise", "a", "ref")]
    x[noise] = 0.1
    space = space_class()(p, x, blocks=("filter",))
    rebuilt, _ = space.from_unconstrained(space.to_unconstrained(x))
    assert float(rebuilt[noise]) == 0.1


@pytest.mark.parametrize("blocks", [("filter",), ("loading", "offset", "noise")])
def test_real_conditional_sampler_and_diagnostics_ignore_fixed_coordinates(blocks):
    b = api()
    assert hasattr(b, "ParameterSubspace"), "block sampler missing"
    model, data = make_model(
        "posterior",
        sample_blocks=list(blocks),
        sampler=b.SamplerConfig(chains=2, warmup=80, draws=80, max_tree_depth=7),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    active = [i for i, n in enumerate(model.parameter_names_) if n[0] in blocks]
    fixed = [i for i in range(len(model.parameter_names_)) if i not in active]
    xs = model.parameter_draws_
    assert_array_equal(
        xs[..., fixed],
        np.broadcast_to(model.map_parameters_[fixed], xs[..., fixed].shape),
    )
    assert np.all(np.std(xs[..., active], axis=(0, 1)) > 0)
    diag = model.sampling_diagnostics_
    assert diag["active_parameters"] == [list(model.parameter_names_[i]) for i in active]
    assert len(diag["parameters"]) == len(active) + 1
    assert np.isfinite([r["r_hat"] for r in diag["parameters"]]).all()
    assert len(diag["initial_unconstrained"][0]) == len(active)
    model.sample_blocks.append("mutated_after_fit")
    assert model.configuration_["sample_blocks"] == blocks
    result = model.infer_latent(times={"train": np.array([6.0, 10.0, 14.0])})["train"]
    assert result.metadata["uncertainty"] == "conditional_parameter_posterior_mixture"
    assert result.metadata["fixed_parameters"] == diag["fixed_parameters"]


def test_map_does_not_silently_ignore_requested_sampling_blocks():
    b = api()
    assert hasattr(b, "ParameterSubspace"), "block controls missing"
    model, data = make_model("map", sample_blocks=("filter",))
    with pytest.raises(ValueError, match="posterior"):
        model.fit(data)
