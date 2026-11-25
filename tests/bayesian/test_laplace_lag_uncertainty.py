"""Model-based (Laplace) lag standard errors in the GP timing report."""

import warnings

import numpy as np
import pytest

from multimodalsrm import Gaussian, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.laplace import lag_standard_errors, objective_hessian

WIDTHS = {"A": 0.4, "B": 0.8, "C": 1.2}
TRUE = {"A": 0.0, "B": 1.0, "C": -0.6}


def recording(features, duration=30.0):
    rng = np.random.default_rng(0)
    frequency = rng.uniform(0.02, 0.3, (features, 12))
    amplitude = rng.normal(size=(features, 12))
    phase = rng.uniform(0, 6.28, (features, 12))
    clocks = {"A": (0.0, 0.5), "B": (0.25, 0.5), "C": (0.0, 1.0)}
    data = {}
    for index in range(2):
        streams = {}
        for modality, (start, step) in clocks.items():
            times = np.arange(start, duration, step)
            shifted = (times - TRUE[modality])[:, None]
            signal = np.stack(
                [
                    (amplitude[k] * np.sin(2 * np.pi * frequency[k] * shifted + phase[k])).sum(-1)
                    for k in range(features)
                ],
                axis=1,
            )
            values = signal @ rng.normal(size=(features, 3)) + rng.normal(size=(len(times), 3))
            streams[modality] = TimeSeries(values, times)
        data[f"s{index}"] = {"run": streams}
    return data


TIGHT = dict(maxiter=3000, gtol=1e-10, ftol=1e-15)


def fit(features=1, fixed=("A",), bounds=None, search=None, flat=False):
    bounds = bounds or {}
    responses = {
        m: Response(Gaussian(w), pooling="shared", estimate=False)
        if m in fixed
        else Response(
            Gaussian(w),
            pooling="shared",
            fixed={"width": w},
            bounds={"lag": bounds.get(m, (-2.0, 2.0))},
        )
        for m, w in WIDTHS.items()
    }
    prior = Prior.uniform(-2.0, 2.0) if flat else Prior.normal(0.0, 1.0)
    filters = {m: {"lag": prior} for m in WIDTHS if m not in fixed}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BayesianMultimodalSRM(
            features=features,
            responses=responses,
            priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0), loading_sd=2.0, filters=filters),
            length_scale=2.0,
            inference="map",
            linear_algebra="grouped",
            max_observations=5000,
            search=SearchConfig(starts=1, **(search or dict(maxiter=600))),
            random_state=0,
        ).fit(recording(features))


def dense_hessian(model):
    import jax
    import jax.numpy as jnp

    hessian = jax.hessian(model.problem_.objective)(jnp.asarray(model.map_parameters_))
    return (np.asarray(hessian) + np.asarray(hessian).T) / 2


def reference_covariance(model):
    """The validated computation: pseudo-inverse of the full dense Hessian."""
    return np.linalg.pinv(dense_hessian(model), rcond=1e-9, hermitian=True)


def covariance_dropping(model, nullity):
    """Pseudo-inverse that drops a stated number of smallest eigenvalues."""
    values, vectors = np.linalg.eigh(dense_hessian(model))
    keep = np.argsort(np.abs(values))[nullity:]
    return (vectors[:, keep] / values[keep]) @ vectors[:, keep].T


def lag_index(model, modality):
    return model.problem_.indices[("filter", modality, "lag")]


@pytest.fixture(scope="module")
def one_factor():
    return fit(features=1)


@pytest.fixture(scope="module")
def all_free():
    return fit(features=2, fixed=(), search=TIGHT)


@pytest.mark.parametrize("features", [1, 2])
def test_lag_standard_errors_match_the_dense_pseudo_inverse(features, one_factor):
    # Tight stopping, so the eigenvalue cut-off of the reference computation
    # finds the rotation direction that the report removes in closed form.
    model = one_factor if features == 1 else fit(features=2, search=TIGHT)
    block = model.timing_report(uncertainty="laplace")["lag_standard_error"]
    covariance = reference_covariance(model)
    assert block["method"] == "laplace" and block["quantity"] == "lag"
    assert "Model-based (Laplace)" in block["label"] and "too narrow" in block["label"]
    assert block["reference"] == "A"
    for m in ("B", "C"):
        expected = np.sqrt(covariance[lag_index(model, m), lag_index(model, m)])
        assert block["lag_model_clock"][m]["standard_error"] == pytest.approx(expected, rel=1e-6)
        # The reference lag is fixed, so it adds no variance.
        assert block["relative_lag"][m]["standard_error"] == pytest.approx(expected, rel=1e-6)
        assert block["relative_lag"][m]["estimate"] == pytest.approx(
            float(model.map_parameters_[lag_index(model, m)]) - 0.0
        )
    assert block["lag_model_clock"]["A"] == dict(estimate=0.0, standard_error=0.0)
    assert block["relative_lag"]["A"]["standard_error"] == 0.0
    assert block["parameters"] == len(model.problem_.names)
    assert block["null_directions"] == features * (features - 1) // 2
    assert block["null_direction_residual"] < 1e-8
    assert block["positive_definite"] is True
    assert block["lags_at_bounds"] == [] and block["not_identified"] == []
    assert block["parameters_at_bounds"] == 0
    assert block["applies_to_peak_delay"] == {"A": True, "B": True, "C": True}
    assert block["hessian_seconds"] > 0


def test_rotation_directions_are_removed_at_default_fit_tolerances():
    # At default stopping tolerances the curvature along the rotation is small
    # but not zero, and an eigenvalue cut-off can miss it.
    model = fit(features=2, fixed=())
    block = model.timing_report(uncertainty="laplace")["lag_standard_error"]
    assert block["null_directions"] == 1
    assert block["positive_definite"] is True and block["min_eigenvalue"] > 0.1
    assert 0 < block["null_direction_residual"] < 1e-5
    assert block["not_identified"] == []
    covariance = covariance_dropping(model, 1)
    for m in WIDTHS:
        i = lag_index(model, m)
        assert block["lag_model_clock"][m]["standard_error"] == pytest.approx(
            np.sqrt(covariance[i, i]), rel=1e-4
        )
    # The closed-form directions are exact invariances of the objective.
    from multimodalsrm.bayesian.laplace import invariant_directions

    x = np.asarray(model.map_parameters_, dtype=float)
    tangent = invariant_directions(model.problem_, x)
    assert tangent.shape == (len(x), 1)
    angle = 0.3
    loadings = x[: len(model.problem_.keys) * 2].reshape(-1, 2)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = x.copy()
    rotated[: loadings.size] = (loadings @ rotation.T).ravel()
    value = model.problem_.value_gradient(x)[0]
    assert model.problem_.value_gradient(rotated)[0] == pytest.approx(value, rel=1e-12)
    step = x + 1e-6 * tangent[:, 0]
    assert model.problem_.value_gradient(step)[0] == pytest.approx(value, abs=1e-9)
    one = fit(features=1)
    assert invariant_directions(one.problem_, np.asarray(one.map_parameters_)).shape[1] == 0


def test_flat_lag_priors_leave_the_common_shift_unidentified():
    model = fit(features=1, fixed=(), flat=True)
    block = model.timing_report(uncertainty="laplace")["lag_standard_error"]
    assert block["lags_at_bounds"] == []
    assert block["null_directions"] == 1 and block["positive_definite"] is True
    assert block["not_identified"] == ["lag A", "lag B", "lag C"]
    assert all(np.isnan(v["standard_error"]) for v in block["lag_model_clock"].values())
    covariance = covariance_dropping(model, 1)
    index = {m: lag_index(model, m) for m in WIDTHS}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        i, j = index[b], index[a]
        expected = np.sqrt(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j])
        assert block["pairwise_lag_difference"][f"{b} - {a}"]["standard_error"] == pytest.approx(
            expected, rel=1e-6
        )


def test_relative_and_pairwise_errors_use_the_lag_covariance(all_free):
    model = all_free
    covariance = reference_covariance(model)
    index = {m: lag_index(model, m) for m in WIDTHS}

    def expected(m, reference):
        i, j = index[m], index[reference]
        return np.sqrt(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j])

    block = model.timing_report(reference="B", uncertainty="laplace")["lag_standard_error"]
    assert block["reference"] == "B"
    assert block["relative_lag"]["B"]["standard_error"] == 0.0
    for m in ("A", "C"):
        assert block["relative_lag"][m]["standard_error"] == pytest.approx(
            expected(m, "B"), rel=1e-6
        )
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        entry = block["pairwise_lag_difference"][f"{b} - {a}"]
        assert entry["standard_error"] == pytest.approx(expected(b, a), rel=1e-6)
        assert entry["estimate"] == pytest.approx(
            float(model.map_parameters_[index[b]] - model.map_parameters_[index[a]])
        )
    # The common offset is set by the priors alone, so each lag is far less
    # certain than any difference: ignoring the covariance would be wrong.
    single = block["lag_model_clock"]["C"]["standard_error"]
    naive = np.hypot(single, block["lag_model_clock"]["B"]["standard_error"])
    assert block["pairwise_lag_difference"]["C - B"]["standard_error"] < 0.5 * naive


def test_a_lag_at_its_bound_is_flagged_and_has_no_standard_error(one_factor):
    free = float(one_factor.map_parameters_[lag_index(one_factor, "B")])
    edge = free - 0.5 if free > 0 else free + 0.5
    model = fit(features=1, bounds={"B": (min(edge, 0.0), max(edge, 0.0))})
    assert float(model.map_parameters_[lag_index(model, "B")]) == pytest.approx(edge, abs=1e-6)
    report = model.timing_report(uncertainty="laplace")
    block = report["lag_standard_error"]
    assert block["lags_at_bounds"] == ["B"] and block["parameters_at_bounds"] == 1
    # The pinned lag is held fixed, so the rest is still a local optimum.
    assert block["positive_definite"] is True
    assert report["lags_at_bounds"] == [["filter", "B", "lag"]]
    assert np.isnan(block["lag_model_clock"]["B"]["standard_error"])
    assert np.isnan(block["relative_lag"]["B"]["standard_error"])
    assert np.isnan(block["pairwise_lag_difference"]["C - B"]["standard_error"])
    assert block["relative_lag"]["B"]["estimate"] == pytest.approx(edge, abs=1e-6)
    assert np.isfinite(block["relative_lag"]["C"]["standard_error"])
    assert np.isfinite(block["pairwise_lag_difference"]["C - A"]["standard_error"])


def test_unsupported_fits_and_options_raise(one_factor):
    model = one_factor
    with pytest.raises(ValueError, match="uncertainty must be None, 'laplace' or 'robust'"):
        model.timing_report(uncertainty="bootstrap")
    with pytest.raises(ValueError, match="limit is 10"):
        lag_standard_errors(model, "A", max_parameters=10)
    # One run cannot serve as clusters.
    with pytest.raises(ValueError, match="needs at least 3; this fit has 1.*split_runs"):
        model.timing_report(uncertainty="robust")
    for key, value, message in (
        ("inference", "posterior", "requires a MAP fit; a posterior fit already has"),
        ("parameter_source", "original_training_MAP", "requires the training fit"),
    ):
        missing = object()
        original = model.configuration_.get(key, missing)
        try:
            model.configuration_[key] = value
            with pytest.raises(ValueError, match=message):
                lag_standard_errors(model, "A")
        finally:
            if original is missing:
                del model.configuration_[key]
            else:
                model.configuration_[key] = original
    algebra = model.problem_.linear_algebra
    try:
        model.problem_.linear_algebra = "spectral"
        with pytest.raises(ValueError, match="dense, grouped or state_space"):
            model.timing_report(uncertainty="laplace")
    finally:
        model.problem_.linear_algebra = algebra


def test_timing_report_passes_the_parameter_cap_through(one_factor):
    count = len(one_factor.problem_.names)
    assert count < 2000
    with pytest.raises(ValueError, match=rf"all {count} parameters; the limit is {count - 1}\."):
        one_factor.timing_report(uncertainty="laplace", max_parameters=count - 1)
    # The message names the keyword and what raising it costs.
    with pytest.raises(ValueError, match=r"timing_report\(\.\.\., max_parameters=N\).*minutes"):
        one_factor.timing_report(uncertainty="laplace", max_parameters=10)
    exact = one_factor.timing_report(uncertainty="laplace", max_parameters=count)
    default = one_factor.timing_report(uncertainty="laplace")
    for report in (exact, default):
        report["lag_standard_error"].pop("hessian_seconds")
    assert exact == default
    # Without an uncertainty request nothing is built, so the cap is not consulted.
    assert "lag_standard_error" not in one_factor.timing_report(max_parameters=1)


def test_default_report_is_unchanged(one_factor):
    default = one_factor.timing_report()
    assert "lag_standard_error" not in default
    assert default["uncertainty"] == "conditional_on_MAP"
    extended = one_factor.timing_report(uncertainty="laplace")
    extended.pop("lag_standard_error")
    assert extended == default


def test_hessian_matches_finite_differences_of_the_gradient(one_factor):
    problem = one_factor.problem_
    x = np.asarray(one_factor.map_parameters_, dtype=float)
    hessian = objective_hessian(problem, x, batch_size=7)
    step = 1e-5
    numeric = np.empty_like(hessian)
    for j in range(len(x)):
        shift = np.zeros(len(x))
        shift[j] = step
        numeric[j] = (
            problem.value_gradient(x + shift)[1] - problem.value_gradient(x - shift)[1]
        ) / (2 * step)
    np.testing.assert_allclose(hessian, hessian.T)
    np.testing.assert_allclose(hessian, numeric, rtol=1e-5, atol=1e-6 * np.abs(hessian).max())
