"""Noise autocorrelation timescales as fitted parameters of the GP model."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.problem import BayesianProblem

from .test_bayesian_multifactor import fixture

BRAIN, AUX = ("noise_timescale", "brain"), ("noise_timescale", "aux")
TRUE_TIMESCALE = -1.0 / np.log(0.7)


def problems(features=1, **kwargs):
    """White, fixed-OU and estimated-OU problems on the same observations.

    ``brain`` is scored every 2 s over 16 s and ``aux`` every 1.5 s over 4.5 s.
    """
    original, x, _ = fixture(features=features)
    shared = dict(anchor=original.anchor, **kwargs)
    fixed = BayesianProblem(
        original.adapter, original.priors, noise_timescales={"brain": 3.0, "aux": 1.2}, **shared
    )
    estimated = BayesianProblem(
        original.adapter, original.priors, estimate_noise_timescales=True, **shared
    )
    return original, fixed, estimated, x


def test_timescales_are_named_bounded_parameters_with_a_prior_and_transform():
    original, _, estimated, x = problems()
    assert estimated.names[: len(original.names)] == original.names
    assert estimated.names[len(original.names) :] == [BRAIN, AUX]
    # A twentieth of the sampling interval to the longest scored span.
    assert_allclose(estimated.bounds[estimated.indices[BRAIN]], (0.1, 16.0))
    assert_allclose(estimated.bounds[estimated.indices[AUX]], (0.075, 4.5))
    for name, interval in ((BRAIN, 2.0), (AUX, 1.5)):
        prior = estimated.parameter_priors[estimated.indices[name]]
        assert (prior.family, prior.scale) == ("lognormal", 1.25)
        assert_allclose(prior.loc, np.log(2 * interval))
        assert_allclose(estimated.initial[estimated.indices[name]], prior.ppf(0.5))
    full = np.concatenate([x, [3.0, 1.2]])
    back, _ = estimated.from_unconstrained(estimated.to_unconstrained(full))
    assert_allclose(back, full, rtol=1e-10)
    # The prior is part of the objective, and support ends at the bounds.
    density = estimated.log_prior(full) - original.log_prior(x)
    expected = sum(
        estimated.distributions[estimated.indices[name]].log_prob(value)
        for name, value in ((BRAIN, 3.0), (AUX, 1.2))
    )
    assert_allclose(density, expected, rtol=1e-10)
    assert np.isneginf(estimated.log_prior(np.concatenate([x, [3.0, 0.05]])))


def test_supplied_timescales_are_starting_values_and_priors_can_be_set():
    original, _, _ = fixture(features=1)
    priors = BayesianPriors(
        noise=original.priors.noise,
        loading_sd=original.priors.loading_sd,
        offset_sd=original.priors.offset_sd,
        filters=original.priors.filters,
        noise_timescale={"aux": Prior.uniform(0.5, 100.0)},
    )
    problem = BayesianProblem(
        original.adapter,
        priors,
        anchor=original.anchor,
        noise_timescales={"brain": 5.0},
        estimate_noise_timescales=True,
    )
    assert problem.initial[problem.indices[BRAIN]] == 5.0
    aux = problem.parameter_priors[problem.indices[AUX]]
    # A supplied prior is truncated to the default interval; the other is default.
    assert (aux.family, aux.lower, aux.upper) == ("uniform", 0.5, 4.5)
    assert problem.parameter_priors[problem.indices[BRAIN]].family == "lognormal"
    assert problem.noise_timescales == {"brain": 5.0, "aux": 2.5}
    with pytest.raises(ValueError, match="starting value"):
        BayesianProblem(
            original.adapter,
            original.priors,
            anchor=original.anchor,
            noise_timescales={"brain": 40.0},
            estimate_noise_timescales=True,
        )
    with pytest.raises(ValueError, match="no support"):
        BayesianProblem(
            original.adapter,
            BayesianPriors(
                noise=original.priors.noise,
                filters=original.priors.filters,
                noise_timescale=Prior.uniform(50.0, 100.0),
            ),
            anchor=original.anchor,
            estimate_noise_timescales=True,
        )
    for invalid in (2.0, {"brain": 2.0}, Prior.uniform(-2.0, -1.0)):
        with pytest.raises(ValueError, match="noise_timescale prior"):
            BayesianPriors(noise=original.priors.noise, noise_timescale=invalid)


@pytest.mark.parametrize("features", [1, 2])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_objective_matches_fixed_timescales_and_gradient_matches_differences(features, algebra):
    _, fixed, estimated, x = problems(features, linear_algebra=algebra)
    full = np.concatenate([x, [3.0, 1.2]])
    # The JAX precision at these timescales is the NumPy one prepared for them.
    assert_allclose(float(estimated.nll(full)), float(fixed.nll(x)), rtol=1e-12)
    value, gradient = estimated.value_gradient(full)
    assert_allclose(gradient[: len(x)], fixed.value_gradient(x)[1], rtol=1e-9, atol=1e-9)
    for name in (BRAIN, AUX):
        i = estimated.indices[name]
        step = np.zeros(len(full))
        step[i] = 1e-5
        expected = (
            float(estimated.objective(full + step)) - float(estimated.objective(full - step))
        ) / 2e-5
        assert abs(gradient[i]) > 1e-3
        assert_allclose(gradient[i], expected, rtol=1e-6, atol=1e-7)
    if algebra == "grouped":
        dense = problems(features)[2].value_gradient(full)
        assert_allclose(value, dense[0], rtol=1e-10)
        assert_allclose(gradient, dense[1], rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_zero_timescale_limit_is_the_white_noise_model(algebra):
    original, _, estimated, x = problems(linear_algebra=algebra)
    white = BayesianProblem(
        original.adapter, original.priors, anchor=original.anchor, linear_algebra=algebra
    )
    tiny = np.concatenate([x, [1e-3, 1e-3]])
    assert_allclose(float(estimated.nll(tiny)), float(white.nll(x)), rtol=1e-12)
    gradient = np.asarray(estimated._vg(tiny)[1])
    assert np.isfinite(gradient).all()
    assert_allclose(gradient[-2:], 0.0, atol=1e-12)
    # At the lower bound, a twentieth of the interval, the two already agree.
    bound = np.concatenate([x, [0.1 + 1e-9, 0.075 + 1e-9]])
    assert_allclose(float(estimated.nll(bound)), float(white.nll(x)), rtol=1e-8)


@pytest.mark.parametrize("features", [1, 2])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
@pytest.mark.parametrize("noisy", [False, True])
def test_predictions_use_the_fitted_timescales(features, algebra, noisy):
    _, fixed, estimated, x = problems(features, linear_algebra=algebra)
    full = np.concatenate([x, [3.0, 1.2]])
    query, key = np.array([2.5, 4.0, 7.1]), ("b", "brain", 1)
    expected = project(fixed, x[None], "train", query, key=key, include_noise=noisy)
    actual = project(estimated, full[None], "train", query, key=key, include_noise=noisy)
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    other = np.concatenate([x, [6.0, 1.2]])
    moved = project(estimated, other[None], "train", query, key=key, include_noise=noisy)
    assert not np.allclose(moved[0], expected[0], rtol=1e-6)


def test_fixed_timescale_values_are_unchanged_from_the_previous_release():
    """Objective values recorded on main before timescales could be estimated."""
    recorded = {
        (1, "dense"): RECORDED_FIXED[0],
        (1, "grouped"): RECORDED_FIXED[1],
        (2, "dense"): RECORDED_FIXED[2],
        (2, "grouped"): RECORDED_FIXED[3],
    }
    for (features, algebra), value in recorded.items():
        _, fixed, _, x = problems(features, linear_algebra=algebra)
        assert not fixed.estimate_noise_timescales
        assert fixed.names[-1][0] != "noise_timescale"
        assert_allclose(fixed.value_gradient(x)[0], value, rtol=1e-12, atol=0)


RECORDED_FIXED = (187.72657818072233, 187.72657818072233, 236.85883720929203, 236.85883720929192)


def recording(participants=3, duration=120.0, seed=5):
    """One smooth shared signal; ``slow`` has AR(1) 0.7 noise and ``white`` white noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, 1.0)
    frequency, phase = rng.uniform(0.01, 0.12, 10), rng.uniform(0, 2 * np.pi, 10)
    signal = np.sin(2 * np.pi * frequency * times[:, None] + phase).sum(1) / np.sqrt(5)
    data = {}
    for p in range(participants):
        streams = {}
        for modality, rho in (("slow", 0.7), ("white", 0.0)):
            innovation = rng.normal(size=(len(times), 3))
            noise = innovation.copy()
            for i in range(1, len(times)):
                noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho**2) * innovation[i]
            streams[modality] = TimeSeries(signal[:, None] * rng.normal(1.0, 0.2, 3) + noise, times)
        data[f"p{p}"] = {"run": streams}
    return data


def model(**kwargs):
    settings = dict(
        anchor=("p0", "slow", 0),
        responses={
            "slow": Response(Identity(), estimate=False),
            "white": Response(Gaussian(0.5), fixed={"width": 0.5}, bounds={"lag": (-2.0, 2.0)}),
        },
        priors=BayesianPriors(
            noise=Prior.lognormal(0.0, 1.0), filters={"white": {"lag": Prior.normal(0.0, 1.0)}}
        ),
        length_scale=3.0,
        linear_algebra="grouped",
        max_observations=100_000,
        search=SearchConfig(starts=2, maxiter=2000),
        random_state=0,
        estimate_noise_timescales=True,
    )
    return BayesianMultimodalSRM(**{**settings, **kwargs})


@pytest.fixture(scope="module")
def fitted():
    data = recording()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return model().fit(data), data


def test_fit_recovers_a_known_timescale_and_leaves_white_noise_white(fitted):
    fit, _ = fitted
    assert fit.map_diagnostics_["meets_gradient_tolerance"]
    timescales = fit.noise_timescales_
    assert set(timescales) == {"slow", "white"}
    assert 0.7 * TRUE_TIMESCALE < timescales["slow"] < 1.5 * TRUE_TIMESCALE
    correlation = fit.noise_autocorrelations_
    assert_allclose(correlation["slow"], np.exp(-1.0 / timescales["slow"]))
    assert abs(correlation["slow"] - 0.7) < 0.1
    # White noise settles inside the bounds, where the process is white.
    assert correlation["white"] < 0.05
    assert 0.05 < timescales["white"] < 0.5


def test_search_starts_from_the_white_noise_map(fitted):
    fit, data = fitted
    first = fit.restart_diagnostics_[0]
    assert first["initialization"]["method"] == "white_noise_map"
    assert first["initialization"]["available"]
    assert "initialization" not in fit.restart_diagnostics_[1]
    assert fit.phase_seconds_["white_noise_map"] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        white = model(estimate_noise_timescales=False).fit(data)
    assert first["initialization"]["objective"] == white.objective_
    count = len(white.map_parameters_)
    assert_allclose(first["initial_parameters"][:count], white.map_parameters_, atol=1e-5)
    start = fit.configuration_["temporal_noise"]["starting_timescales"]
    assert first["initial_parameters"][count:] == [start["slow"], start["white"]]


def test_reports_expose_timescale_autocorrelation_bounds_and_standard_error(fitted):
    fit, _ = fitted
    report = fit.noise_report()
    assert report["estimated"] and report["family"] == "OU" and report["units"] == "seconds"
    assert report["timescale"] == fit.noise_timescales_
    assert report["lag_one_autocorrelation"] == fit.noise_autocorrelations_
    assert report["sampling_interval"] == {"slow": 1.0, "white": 1.0}
    assert report["bounds"] == {"slow": (0.05, 119.0), "white": (0.05, 119.0)}
    assert report["timescales_at_bounds"] == []
    assert "standard_error" not in report
    recorded = fit.configuration_["temporal_noise"]
    assert recorded["timescales_estimated"] and recorded["timescales"] == fit.noise_timescales_
    assert recorded["lag_one_autocorrelations"] == fit.noise_autocorrelations_
    assert recorded["timescale_parameters"] == [
        ["noise_timescale", "slow"],
        ["noise_timescale", "white"],
    ]
    assert fit.parameter_names_[-2:] == [("noise_timescale", "slow"), ("noise_timescale", "white")]

    errors = fit.noise_report(uncertainty="laplace")["standard_error"]
    assert errors["positive_definite"] and errors["timescales_at_bounds"] == []
    slow = errors["modalities"]["slow"]
    assert slow["timescale"] == fit.noise_timescales_["slow"]
    assert 0.05 < slow["standard_error"] < 1.0
    assert abs(slow["timescale"] - TRUE_TIMESCALE) < 4 * slow["standard_error"]
    assert_allclose(
        slow["autocorrelation_standard_error"],
        slow["lag_one_autocorrelation"] / slow["timescale"] ** 2 * slow["standard_error"],
    )
    # The lag report sees the same enlarged parameter vector.
    lag = fit.timing_report(uncertainty="laplace")["lag_standard_error"]
    assert lag["parameters"] == len(fit.parameter_names_) and lag["positive_definite"]
    with pytest.raises(ValueError, match="uncertainty"):
        fit.noise_report(uncertainty="bootstrap")


def test_fixed_and_white_models_report_without_new_parameters():
    data = recording(participants=2, duration=40.0)
    quick = SearchConfig(starts=1, maxiter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = model(
            estimate_noise_timescales=False, noise_timescales={"slow": 2.5}, search=quick
        ).fit(data)
        white = model(estimate_noise_timescales=False, search=quick).fit(data)
    assert fixed.noise_timescales_ == {"slow": 2.5}
    assert_allclose(fixed.noise_autocorrelations_["slow"], np.exp(-1 / 2.5))
    report = fixed.noise_report()
    assert not report["estimated"] and "bounds" not in report
    assert report["sampling_interval"] == {"slow": 1.0}
    assert "timescales_estimated" not in fixed.configuration_["temporal_noise"]
    assert "estimate_noise_timescales" not in fixed.specification_
    assert "noise_timescale" not in fixed.specification_["priors"]
    assert not any(name[0] == "noise_timescale" for name in fixed.parameter_names_)
    with pytest.raises(ValueError, match="estimate_noise_timescales=True"):
        fixed.noise_report(uncertainty="laplace")
    assert white.noise_timescales_ == {} and white.noise_autocorrelations_ == {}
    assert white.noise_report()["family"] == "white"
    assert "temporal_noise" not in white.configuration_


def test_timescale_on_a_bound_is_flagged_and_gets_no_standard_error():
    data = recording(participants=2, duration=60.0)
    priors = BayesianPriors(
        noise=Prior.lognormal(0.0, 1.0),
        filters={"white": {"lag": Prior.normal(0.0, 1.0)}},
        # White noise wants the shortest timescale this prior allows.
        noise_timescale={"white": Prior.uniform(1.0, 50.0)},
    )
    with pytest.warns(UserWarning, match="noise timescales ended on a bound"):
        fit = model(priors=priors).fit(data)
    assert fit.noise_report()["timescales_at_bounds"] == ["white"]
    assert fit.configuration_["temporal_noise"]["timescales_at_bounds"] == ["white"]
    assert_allclose(fit.noise_timescales_["white"], 1.0, rtol=1e-6)
    errors = fit.noise_report(uncertainty="laplace")["standard_error"]
    assert errors["timescales_at_bounds"] == ["white"]
    assert np.isnan(errors["modalities"]["white"]["standard_error"])
    assert np.isfinite(errors["modalities"]["slow"]["standard_error"])


def test_archive_restores_the_fitted_timescales(fitted, tmp_path):
    from multimodalsrm.bayesian.workflow import load_model, save_model

    fit, data = fitted
    save_model(tmp_path / "model", fit)
    restored, _ = load_model(tmp_path / "model")
    assert restored.estimate_noise_timescales is True
    assert restored.noise_timescales_ == fit.noise_timescales_
    assert restored.noise_report() == fit.noise_report()
    assert restored.configuration_ == fit.configuration_
    assert restored.parameter_names_ == fit.parameter_names_
    query = np.array([10.5, 30.0])
    assert_allclose(
        restored.infer_latent(times=query)["run"].component_means,
        fit.infer_latent(times=query)["run"].component_means,
        rtol=1e-10,
    )
    # A changed timescale is a changed parameter, not a changed setting.
    tampered = fit.map_parameters_.copy()
    tampered[-1] = -1.0
    broken = load_model(tmp_path / "model")[0]
    broken.map_parameters_ = tampered
    with pytest.raises(ValueError):
        save_model(tmp_path / "broken", broken)


def test_frozen_conditioning_and_transform_keep_the_training_timescales(fitted):
    fit, data = fitted
    new = {subject: {"new": dict(runs["run"])} for subject, runs in recording(seed=6).items()}
    conditioned = fit.condition(new, targets={"p1": ["white"]}, mode="frozen")
    assert conditioned.noise_timescales_ == fit.noise_timescales_
    assert conditioned.problem_.names == fit.problem_.names
    assert conditioned.problem_.bounds == fit.problem_.bounds
    for noisy in (False, True):
        predicted = conditioned.predict(times=np.array([20.0, 21.5]), include_noise=noisy)
        result = predicted["p1"]["new"]["white"]
        assert np.isfinite(result.component_means).all()
        assert np.all(result.component_variances > 0)
    latent = fit.transform({"p0": new["p0"]}, times=np.array([20.0, 40.0]))["p0"]["new"]
    assert np.isfinite(latent.component_means).all()
    with pytest.raises(ValueError, match="instantaneous readout"):
        fit.transform({"p0": new["p0"]}, times=np.array([20.0]), readout="instantaneous")


@pytest.mark.parametrize(
    "settings,message",
    [
        (dict(inference="posterior"), "estimate_noise_timescales requires dense/grouped MAP"),
        (dict(linear_algebra="state_space"), "estimate_noise_timescales requires dense/grouped"),
        (dict(run_baseline_sd={"slow": 0.1}), "estimate_noise_timescales requires dense/grouped"),
        (dict(length_scale=Prior.uniform(1.0, 5.0)), "independent noise"),
        (dict(estimate_noise_timescales=1), "must be Boolean"),
    ],
)
def test_unsupported_combinations_raise(settings, message):
    with pytest.raises(ValueError, match=message):
        model(**settings).fit(recording(participants=2, duration=20.0))


def test_spectral_algebra_raises():
    from multimodalsrm.bayesian import SpectralConfig

    with pytest.raises(ValueError, match="estimate_noise_timescales requires dense/grouped"):
        model(
            linear_algebra="spectral",
            spectral=SpectralConfig(rank=8, padding=5.0),
            responses={
                "slow": Response(Identity(), estimate=False),
                "white": Response(Identity(), estimate=False),
            },
            priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0)),
        ).fit(recording(participants=2, duration=20.0))


def test_default_is_off_and_cloning_keeps_the_flag():
    from sklearn.base import clone

    from multimodalsrm.bayesian.temporal_noise import ESTIMATE_BY_DEFAULT

    default = BayesianMultimodalSRM(priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0)))
    assert default.estimate_noise_timescales is ESTIMATE_BY_DEFAULT is False
    assert clone(model()).estimate_noise_timescales is True
