"""Independent dense references for the explicit spectral approximation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .test_bayesian_problem import api


def test_spectral_covariance_converges_with_identity_and_filtered_responses():
    api()
    from multimodalsrm.bayesian._backend import temporal
    from multimodalsrm.bayesian.spectral import (
        SpectralBasis,
        SpectralConfig,
    )

    times = np.array([0.0, 0.123, 1.4, 7.12, 19.3, 32.0])
    widths = np.array([0.0, 0.15, 0.45, 0.85, 0.0, 0.35])
    lags = np.array([0.0, -1.8, 0.0, 1.8, 0.0, 0.9])
    exact = temporal(times, times, widths, widths, lags, lags, 3.0)
    errors = []
    for rank in (64, 128, 256):
        basis = SpectralBasis.prepare(times, 3.0, SpectralConfig(rank, 24.0))
        features = basis.features(times, widths, lags)
        errors.append(np.max(np.abs(features @ features.T - exact)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-4


@pytest.mark.parametrize("zero,scale", [(False, 1.0), (True, 1.0), (False, 1e-4)])
def test_feature_solve_and_gradients_match_dense_same_approximation(zero, scale):
    api()
    import jax
    import jax.numpy as jnp

    from multimodalsrm.bayesian.spectral import (
        factor_arrays,
        gaussian_score,
    )

    rng = np.random.default_rng(361)
    features = jnp.asarray(rng.normal(size=(5, 7)))  # rank-deficient is allowed
    nodes = np.array([0, 1, 1, 2, 3, 3, 4, 0])
    weights = jnp.asarray(np.zeros(8) if zero else rng.normal(size=8))
    residual = jnp.asarray(rng.normal(size=8))
    variance = jnp.asarray(rng.uniform(0.5, 2, 8) * scale)

    def dense(f, w, r, v):
        F = f[nodes] * w[:, None]
        C = F @ F.T + jnp.diag(v)
        L = jnp.linalg.cholesky(C)
        y = jax.scipy.linalg.solve_triangular(L, r, lower=True)
        return 0.5 * (y @ y + 2 * jnp.log(jnp.diag(L)).sum() + len(r) * np.log(2 * np.pi))

    def fast(f, w, r, v):
        return gaussian_score()(f, w, r, v, nodes)

    value, gradients = jax.value_and_grad(fast, argnums=(0, 1, 2, 3))(
        features, weights, residual, variance
    )
    expected, expected_gradients = jax.value_and_grad(dense, argnums=(0, 1, 2, 3))(
        features, weights, residual, variance
    )
    assert_allclose(value, expected, rtol=1e-8, atol=1e-8)
    assert_allclose(
        value,
        factor_arrays(features, weights, residual, variance, nodes)[0],
        atol=1e-12,
    )
    for actual, reference in zip(gradients, expected_gradients):
        assert_allclose(actual, reference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "rank,padding", [(0, 24), (True, 24), (4.5, 24), (4, 0), (4, np.inf), (4, True)]
)
def test_spectral_configuration_rejects_invalid_values(rank, padding):
    from multimodalsrm.bayesian.spectral import SpectralConfig

    with pytest.raises(ValueError):
        SpectralConfig(rank, padding)


def spectral_fixture(rank=256):
    from multimodalsrm.bayesian.spectral import SpectralConfig

    from .test_bayesian_grouped import grouped_fixture

    dense, _, x = grouped_fixture()
    spectral = api().BayesianProblem(
        dense.adapter,
        dense.priors,
        anchor=dense.anchor,
        linear_algebra="spectral",
        spectral=SpectralConfig(rank, 24.0),
    )
    return dense, spectral, x


def test_spectral_problem_converges_to_exact_physical_density_and_gradient():
    dense, spectral, x = spectral_fixture()
    assert dense.names == spectral.names
    expected, gradient = dense.value_gradient(x)
    actual, actual_gradient = spectral.value_gradient(x)
    assert abs(expected - actual) < 0.01
    assert np.max(abs(gradient - actual_gradient) / (1 + abs(gradient))) < 0.005
    for run in spectral.systems:
        assert spectral.grouped_systems[run].covariance_pairs is None
        assert_allclose(spectral.covariance(x, run), dense.covariance(x, run), atol=0.001)


@pytest.mark.parametrize(
    "key,include_noise",
    [(None, False), (("b", "signal", 1), False), (("b", "signal", 1), True)],
)
def test_spectral_prediction_converges_to_exact_and_uses_same_finite_prior(key, include_noise):
    from multimodalsrm.bayesian.prediction import project

    dense, spectral, x = spectral_fixture()
    times = np.array([2.7, 7.1, 15.9])
    expected = project(dense, np.array([x]), "train", times, key=key, include_noise=include_noise)
    actual = project(spectral, np.array([x]), "train", times, key=key, include_noise=include_noise)
    assert_allclose(actual, expected, atol=0.001, rtol=0)
    # No observations of the latent when every loading vanishes. Its predictive
    # variance must be the finite basis diagonal, not exact unit GP variance.
    x[: len(spectral.keys)] = 0
    _, low, _ = spectral_fixture(rank=4)
    mean, variance = project(low, np.array([x]), "train", times, key=None, include_noise=False)
    F = low.spectral_bases["train"].features(times, np.zeros(3), np.zeros(3))
    assert_allclose(mean, 0, atol=1e-12)
    assert_allclose(variance[0], np.sum(F**2, axis=1), atol=1e-12)
    assert not np.allclose(variance, 1.0)


def test_spectral_clone_conditioning_metadata_and_target_exclusion():
    from sklearn.base import clone

    from multimodalsrm.bayesian.spectral import SpectralConfig

    from .test_bayesian_model import make_model

    b = api()
    model, data = make_model(linear_algebra="spectral", spectral=SpectralConfig(32, 24))
    assert clone(model).spectral == model.spectral
    model.search = b.SearchConfig(starts=1, maxiter=30)
    model.fit(data)
    donors = {s: {"test": dict(runs["train"])} for s, runs in data.items()}
    donors["b"]["test"]["signal"] = object()
    conditioned = model.condition(donors, targets={"b": ["signal"]})
    assert conditioned.problem_.linear_algebra == "spectral"
    assert set(conditioned.configuration_["covariance_approximation"]) == {
        "train",
        "test",
    }
    assert conditioned.problem_.spectral_bases["train"] == model.problem_.spectral_bases["train"]
    result = conditioned.infer_latent(times=np.array([4.0, 5.0]))["test"]
    assert result.metadata["covariance_approximation"]["rank"] == 32
    assert result.metadata["covariance_approximation"]["error_bound"] is None
    with pytest.raises(ValueError, match="specification changed"):
        model.set_params(spectral=SpectralConfig(64, 24)).condition(
            donors, targets={"b": ["signal"]}
        )


def test_spectral_configuration_cannot_be_silently_ignored():
    from multimodalsrm.bayesian.spectral import SpectralConfig

    from .test_bayesian_model import make_model

    for kwargs in (
        dict(linear_algebra="spectral"),
        dict(spectral=SpectralConfig(32, 24)),
    ):
        model, data = make_model(**kwargs)
        with pytest.raises(ValueError, match="spectral"):
            model.fit(data)


def test_small_padding_exposes_extrapolation_without_changing_native_support():
    from multimodalsrm.bayesian import SpectralConfig

    from .test_bayesian_reference import filtered_fixture

    b, adapter, priors, data = filtered_fixture()
    model = b.BayesianMultimodalSRM(
        responses=adapter.responses_,
        priors=priors,
        anchor=("a", "signal", 0),
        reference_modality="ref",
        inference="map",
        linear_algebra="spectral",
        spectral=SpectralConfig(8, 0.01),
    )
    model.search = api().SearchConfig(starts=1, maxiter=10)
    model.fit(data)
    basis = model.problem_.spectral_bases["train"]
    times = np.array([0.0, (basis.left + basis.right) / 2, 24.0])
    result = model.infer_latent(times={"train": times})["train"]
    meta = result.metadata["covariance_approximation"]
    assert meta["domain_source"] == "support_eligible_conditioning_observations"
    assert meta["extrapolation_validated"] is False
    assert meta["query_outside_basis_domain"] == [True, False, True]
    assert result.valid.all()
    a, b = model.prediction_runs_["train"]
    np.testing.assert_array_equal(result.valid, (times >= a) & (times <= b))
