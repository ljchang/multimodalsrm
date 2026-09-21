"""Composition checks for the restricted multifactor Haar/NUTS transition."""

import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .test_bayesian_multifactor import fixture
from .test_bayesian_problem import api


def implementation():
    name = "multimodalsrm.bayesian.orthogonal"
    assert importlib.util.find_spec(name) is not None, "orthogonal kernel is missing"
    return __import__(name, fromlist=["orthogonal_kernel"])


@pytest.fixture(scope="module", params=[2, 3])
def tiny(request):
    from multimodalsrm import Identity, Response, TimeSeries
    from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter

    b = api()
    factors = request.param
    times = np.arange(4.0)
    data = {
        "a": {
            "train": {
                "ref": TimeSeries(
                    (
                        np.array([[0.1, 0.3], [0.2, -0.1], [-0.3, 0.4], [0.2, 0.1]])
                        if factors == 2
                        else np.random.default_rng(427).normal(0, 0.3, (4, factors))
                    ),
                    times,
                )
            }
        }
    }
    adapter = BayesianObservationAdapter(
        features=factors,
        latent_dt=1.0,
        standardize=False,
        responses={"ref": Response(Identity(), estimate=False, pooling="shared")},
    )
    adapter._prepare(data)
    p = b.BayesianProblem(
        adapter,
        b.BayesianPriors(noise=b.Prior.lognormal(-1, 0.5)),
        anchor=("a", "ref", 0),
    )
    x = p.initial.copy()
    x[: factors**2] = [1.0, 0.2, -0.3, 0.8] if factors == 2 else (np.eye(factors) + 0.1).ravel()
    return b.ParameterSubspace(p, x)


def tree_equal(a, b):
    import jax

    assert jax.tree.structure(a) == jax.tree.structure(b)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert_array_equal(left, right)


def kernels(space):
    from numpyro.infer import NUTS

    def make():
        return NUTS(
            potential_fn=space.potential,
            dense_mass=True,
            max_tree_depth=3,
            step_size=0.1,
        )

    return make(), implementation().orthogonal_kernel(make(), space)


def test_warmup_identical_and_retained_refresh_caches_and_statistics(tiny):
    import jax

    factors = tiny.problem.features
    nw = len(tiny.problem.keys) * factors
    plain, composed = kernels(tiny)
    init = tiny.to_unconstrained(tiny.reference)
    args = (jax.random.PRNGKey(41), 30, init, (), {})
    a, b = plain.init(*args), composed.init(*args)
    tree_equal(a, b)
    step_a = jax.jit(lambda state: plain.sample(state, (), {}))
    step_b = jax.jit(lambda state: composed.sample(state, (), {}))
    for _ in range(30):
        a, b = step_a(a), step_b(b)
        tree_equal(a, b)
    assert not np.array_equal(a.adapt_state.inverse_mass_matrix, np.eye(len(init)))
    nuts_output, output = step_a(a), step_b(b)
    assert type(output) is type(nuts_output)
    tree_equal(output.adapt_state, nuts_output.adapt_state)
    assert output.r is None
    assert not np.array_equal(output.z[:nw], nuts_output.z[:nw])
    assert_array_equal(output.z[nw:], nuts_output.z[nw:])
    assert_allclose(
        output.z[:nw].reshape(-1, factors) @ output.z[:nw].reshape(-1, factors).T,
        nuts_output.z[:nw].reshape(-1, factors) @ nuts_output.z[:nw].reshape(-1, factors).T,
        atol=1e-12,
    )
    pe, grad = jax.value_and_grad(tiny.potential)(output.z)
    assert_allclose(output.potential_energy, pe, atol=1e-12)
    assert_allclose(output.z_grad, grad, atol=1e-11)
    assert_allclose(
        output.energy - output.potential_energy,
        nuts_output.energy - nuts_output.potential_energy,
        atol=1e-12,
    )
    for field in ("i", "num_steps", "accept_prob", "mean_accept_prob", "diverging"):
        assert_array_equal(getattr(output, field), getattr(nuts_output, field))
    assert_array_equal(output.rng_key, jax.random.split(nuts_output.rng_key)[0])
    assert not np.array_equal(output.rng_key, nuts_output.rng_key)
    tree_equal(output, step_b(b))  # restarting the same state reproduces the step
    next_output = step_b(output)
    assert not np.array_equal(next_output.rng_key, output.rng_key)
    tree_equal(next_output.adapt_state, output.adapt_state)
    with pytest.raises(ValueError, match="momentum"):
        composed.sample(output._replace(r=np.zeros_like(output.z)), (), {})


@pytest.mark.parametrize("factors", [2, 3, 5])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
@pytest.mark.parametrize("gaussian", [False, True])
def test_real_target_common_orthogonal_preserves_density_gram_and_qr(algebra, gaussian, factors):
    import jax

    from multimodalsrm.bayesian.multifactor import orientation

    p, x, _ = fixture(features=factors, algebra=algebra, gaussian=gaussian)
    space = api().ParameterSubspace(p, x)
    impl = implementation()
    assert_array_equal(impl.validate_orthogonal_target(space), np.arange(len(p.keys) * factors))
    z = space.to_unconstrained(x)
    nw = len(p.keys) * factors
    original_w = x[:nw].reshape(-1, factors)
    anchors = p.keys[:factors]
    original_qr = original_w @ np.asarray(orientation(p, x, anchors)["rotation"])
    for seed in range(4):
        Q = np.asarray(impl._haar_matrix(jax.random.PRNGKey(seed), z.dtype, factors))
        rotated = x.copy()
        rotated[:nw] = (original_w @ Q).ravel()
        assert_allclose(p.log_prior(rotated), p.log_prior(x), atol=1e-11)
        assert_allclose(p.nll(rotated), p.nll(x), atol=1e-10)
        assert_allclose(
            space.potential(space.to_unconstrained(rotated)),
            space.potential(z),
            atol=1e-10,
        )
        w = rotated[:nw].reshape(-1, factors)
        assert_allclose(w @ w.T, original_w @ original_w.T, atol=1e-12)
        assert_allclose(
            w @ np.asarray(orientation(p, rotated, anchors)["rotation"]),
            original_qr,
            atol=1e-12,
        )


def test_haar_angles_and_reflections_are_independent():
    import jax

    impl = implementation()
    qs = np.asarray(
        jax.jit(jax.vmap(lambda key: impl._haar_matrix(key, np.float64)))(
            jax.random.split(jax.random.PRNGKey(812), 12000)
        )
    )
    assert_allclose(qs @ qs.transpose(0, 2, 1), np.broadcast_to(np.eye(2), qs.shape), atol=1e-14)
    signs = np.linalg.det(qs)
    assert abs(np.mean(signs)) < 0.035
    angles = np.arctan2(qs[:, 1, 0], qs[:, 0, 0])
    for selector in (np.ones(len(qs), dtype=bool), signs > 0, signs < 0):
        for harmonic in (1, 2, 3, 4):
            assert abs(np.mean(np.exp(1j * harmonic * angles[selector]))) < 0.04


@pytest.mark.parametrize("method", ["sequential", "vectorized"])
def test_mcmc_split_continuation_matches_uninterrupted(tiny, method):
    import jax
    from numpyro.infer import MCMC

    def make():
        _, kernel = kernels(tiny)
        return MCMC(
            kernel,
            num_warmup=6,
            num_samples=9,
            num_chains=2,
            chain_method=method,
            progress_bar=False,
        )

    initial = np.stack([tiny.to_unconstrained(tiny.reference)] * 2)
    key = jax.random.PRNGKey(104)
    full = make()
    full.run(key, init_params=initial, extra_fields=("energy",))
    split = make()
    split.warmup(key, init_params=initial, extra_fields=("energy",))
    split.run(split.post_warmup_state.rng_key, extra_fields=("energy",))
    tree_equal(full.last_state, split.last_state)
    assert_array_equal(full.get_samples(), split.get_samples())
    assert full.get_samples(group_by_chain=True).shape == (2, 9, len(initial[0]))
    assert not np.array_equal(full.last_state.rng_key[0], full.last_state.rng_key[1])
    assert not np.array_equal(
        full.get_samples(group_by_chain=True)[0],
        full.get_samples(group_by_chain=True)[1],
    )
    # A saved retained state has everything needed for exact continuation.
    split.post_warmup_state = split.last_state
    full.post_warmup_state = full.last_state
    split.run(split.last_state.rng_key)
    full.run(full.last_state.rng_key)
    tree_equal(full.last_state, split.last_state)


@pytest.mark.parametrize(
    "change,match",
    [
        ("subclass", "concrete"),
        ("features", "at least two"),
        ("conditional", "all"),
        ("algebra", "dense/grouped"),
        ("order", "loading"),
        ("prior_mean", "prior"),
        ("prior_scale", "prior"),
        ("prior_bounds", "prior"),
        ("transform", "identity"),
        ("systems", "training"),
        ("quadrature", "response"),
    ],
)
def test_target_guard_rejects_unsupported_changes(tiny, change, match):
    import copy

    from numpyro.distributions.transforms import AffineTransform

    b = api()
    p = copy.copy(tiny.problem)
    p.parameter_priors = list(p.parameter_priors)
    p.transforms = list(p.transforms)
    p.names = list(p.names)
    if change == "subclass":

        class Child(type(p)):
            pass

        p.__class__ = Child
    elif change == "features":
        p.features = 1
    elif change == "algebra":
        p.linear_algebra = "state_space"
    elif change == "order":
        p.names[0], p.names[1] = p.names[1], p.names[0]
    elif change == "prior_mean":
        p.parameter_priors[0] = b.Prior.normal(0.1)
    elif change == "prior_scale":
        p.parameter_priors[0] = b.Prior.normal(sd=4)
    elif change == "prior_bounds":
        p.parameter_priors[0] = b.Prior.normal().bounded(0, np.inf)
    elif change == "transform":
        p.transforms[0] = AffineTransform(0, 2)
    elif change == "systems":
        p.systems = {}
    elif change == "quadrature":
        p.response_quadrature_order = 10
    space = copy.copy(tiny)
    space.problem = p
    if change == "conditional":
        space.active_indices = space.active_indices[:-1]
        space.fixed_indices = (len(p.names) - 1,)
    with pytest.raises(ValueError, match=match):
        implementation().validate_orthogonal_target(space)


def test_kernel_guard_rejects_other_target_and_hmc(tiny):
    from numpyro.infer import HMC, NUTS

    impl = implementation()
    with pytest.raises(ValueError, match="NUTS"):
        impl.orthogonal_kernel(HMC(potential_fn=tiny.potential), tiny)
    with pytest.raises(ValueError, match="potential"):
        impl.orthogonal_kernel(NUTS(potential_fn=lambda z: tiny.potential(z)), tiny)


@pytest.mark.parametrize("origin", ["systems", "conventions_from", "missing_marker"])
def test_target_guard_rejects_nontraining_constructor_origins(tiny, origin):
    import copy

    p = tiny.problem
    if origin == "missing_marker":
        conditional = copy.copy(p)
        del conditional._full_training_target
    else:
        kwargs = {origin: p.systems if origin == "systems" else p}
        conditional = api().BayesianProblem(p.adapter, p.priors, anchor=p.anchor, **kwargs)
    # These deliberately use identical observations/parameters: origin is an
    # additional release-scope guard, not evidence that this orbit differs.
    space = api().ParameterSubspace(conditional, tiny.reference)
    with pytest.raises(ValueError, match="training"):
        implementation().validate_orthogonal_target(space)


@pytest.mark.parametrize("factors", [3, 5, 10])
def test_general_haar_moments_include_both_reflection_components(factors):
    import jax

    impl = implementation()
    qs = np.asarray(
        jax.jit(jax.vmap(lambda key: impl._haar_matrix(key, np.float64, factors)))(
            jax.random.split(jax.random.PRNGKey(817), 12000)
        )
    )
    assert_allclose(
        qs @ qs.swapaxes(-1, -2), np.broadcast_to(np.eye(factors), qs.shape), atol=3e-14
    )
    sign = np.linalg.det(qs)
    assert abs(sign.mean()) < 0.04
    for selector in (np.ones(len(qs), dtype=bool), sign > 0, sign < 0):
        selected = qs[selector]
        assert np.max(np.abs(selected.mean(axis=0))) < 0.04
        assert_allclose((selected**2).mean(axis=0), 1 / factors, atol=0.018)
        assert_allclose((selected**4).mean(axis=0), 3 / (factors * (factors + 2)), atol=0.018)
    # A uniform orthogonal draw must not select a preferred input direction.
    direction = np.arange(1, factors + 1, dtype=float)
    direction /= np.linalg.norm(direction)
    projected = qs @ direction
    assert_allclose(projected.T @ projected / len(qs), np.eye(factors) / factors, atol=0.015)
