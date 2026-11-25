"""Cluster-robust (sandwich) lag standard errors with runs as clusters."""

import warnings

import numpy as np
import pytest

from multimodalsrm import Gaussian, Response, TimeSeries, split_runs
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian.laplace import lag_standard_errors, run_scores

WIDTHS = {"A": 0.4, "B": 0.8, "C": 1.2}
TRUE = {"A": 0.0, "B": 1.0, "C": -0.6}
CLOCKS = {"A": (0.0, 0.5), "B": (0.25, 0.5), "C": (0.0, 1.0)}
LENGTH_SCALE = 2.0
TIGHT = dict(maxiter=3000, gtol=1e-10, ftol=1e-15)


def matern32(rng, times, factors):
    """Exact Matern-3/2 paths on an even grid, through the two-state Markov form."""
    rate, step = np.sqrt(3.0) / LENGTH_SCALE, times[1] - times[0]
    transition = np.exp(-rate * step) * np.array(
        [[1 + rate * step, step], [-(rate**2) * step, 1 - rate * step]]
    )
    stationary = np.diag([1.0, rate**2])
    factor = np.linalg.cholesky(
        stationary - transition @ stationary @ transition.T + 1e-14 * np.eye(2)
    )
    state = rng.normal(size=(factors, 2)) * np.sqrt(np.diag(stationary))
    out = np.empty((len(times), factors))
    for i in range(len(times)):
        out[i] = state[:, 0]
        state = state @ transition.T + rng.normal(size=(factors, 2)) @ factor.T
    return out


def recording(runs, duration, features=1, seed=0):
    """Data drawn from the fitted model: Matern latent, Gaussian responses, white noise."""
    rng = np.random.default_rng(seed)
    fine = np.arange(-8.0, duration + 8.0, 0.05)
    loadings = {(s, m): rng.normal(size=(3, features)) for s in ("s0", "s1") for m in WIDTHS}
    # The one-factor model keeps its anchor loading nonnegative; a clearly
    # positive one keeps the fit away from that bound.
    loadings["s0", "A"][0] = np.abs(loadings["s0", "A"][0]) + 0.5
    data = {"s0": {}, "s1": {}}
    for run in range(runs):
        latent = matern32(rng, fine, features)
        for subject in data:
            streams = {}
            for modality, (start, step) in CLOCKS.items():
                times = np.arange(start, duration, step)
                taus = np.arange(-6.0, 6.0, 0.05) * WIDTHS[modality]
                weights = np.exp(-0.5 * (taus / WIDTHS[modality]) ** 2)
                weights /= np.sqrt((weights**2).sum() * 0.05 * WIDTHS[modality])
                weights *= 0.05 * WIDTHS[modality]
                source = times[:, None] - TRUE[modality] - taus[None, :]
                signal = np.stack(
                    [np.interp(source, fine, latent[:, k]) @ weights for k in range(features)],
                    axis=1,
                )
                values = signal @ loadings[subject, modality].T
                streams[modality] = TimeSeries(values + 0.7 * rng.normal(size=values.shape), times)
            data[subject][f"run{run}"] = streams
    return data


def fit(data, features=1, linear_algebra="grouped", search=None):
    responses = {
        m: Response(Gaussian(w), pooling="shared", estimate=False)
        if m == "A"
        else Response(
            Gaussian(w), pooling="shared", fixed={"width": w}, bounds={"lag": (-2.0, 2.0)}
        )
        for m, w in WIDTHS.items()
    }
    filters = {m: {"lag": Prior.normal(0.0, 1.0)} for m in ("B", "C")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BayesianMultimodalSRM(
            features=features,
            responses=responses,
            priors=BayesianPriors(noise=Prior.lognormal(0.0, 1.0), loading_sd=2.0, filters=filters),
            length_scale=LENGTH_SCALE,
            inference="map",
            linear_algebra=linear_algebra,
            max_observations=5000,
            search=SearchConfig(starts=1, **(search or dict(maxiter=600))),
            random_state=0,
        ).fit(data)


@pytest.fixture(scope="module")
def many_runs():
    # Seed chosen so the single-start search does not stop with the anchor
    # loading on its bound (the mirror-image basin), which seeds 0 and 2 do.
    return fit(recording(runs=12, duration=24.0, seed=1), search=TIGHT)


def test_robust_and_laplace_errors_agree_when_the_model_is_right(many_runs):
    laplace = many_runs.timing_report(uncertainty="laplace")["lag_standard_error"]
    block = many_runs.timing_report(uncertainty="robust")["lag_standard_error"]
    assert block["method"] == "robust" and "Cluster-robust" in block["robust_label"]
    assert block["clusters"] == 12 and block["degrees_of_freedom"] == 11
    assert block["few_clusters"] is False
    assert block["positive_definite"] is True and block["score_seconds"] > 0
    quantities = ("lag_model_clock", "relative_lag", "pairwise_lag_difference")
    for quantity in quantities:
        assert block[quantity].keys() == laplace[quantity].keys()
        for name, entry in block[quantity].items():
            # The Laplace part of the robust block is the Laplace report.
            assert entry["standard_error"] == pytest.approx(
                laplace[quantity][name]["standard_error"], rel=1e-9
            )
            assert entry["estimate"] == laplace[quantity][name]["estimate"]
            assert "robust_standard_error" not in laplace[quantity][name]
    # The reference lag is fixed: no variance of either kind.
    assert block["relative_lag"]["A"] == dict(
        estimate=0.0, standard_error=0.0, robust_standard_error=0.0
    )
    # With 11 degrees of freedom the sandwich itself has a relative error of
    # about 20%, so the two agree only loosely.
    for name in ("B", "C"):
        entry = block["relative_lag"][name]
        assert 0.5 < entry["robust_standard_error"] / entry["standard_error"] < 2.0
    entry = block["pairwise_lag_difference"]["C - B"]
    assert 0.5 < entry["robust_standard_error"] / entry["standard_error"] < 2.0
    # And both describe the actual error of this fit.
    for name in ("B", "C"):
        entry = block["relative_lag"][name]
        assert abs(entry["estimate"] - TRUE[name]) < 4 * entry["robust_standard_error"]


def test_robust_error_is_the_sandwich_of_centred_run_scores(many_runs):
    import jax
    import jax.numpy as jnp

    problem = many_runs.problem_
    x = jnp.asarray(many_runs.map_parameters_)
    hessian = np.asarray(jax.hessian(problem.objective)(x))
    inverse = np.linalg.inv((hessian + hessian.T) / 2)
    scores = np.stack(
        [np.asarray(jax.grad(lambda v, r=run: problem.run_nll(v, r))(x)) for run in problem.systems]
    )
    centred = scores - scores.mean(axis=0)
    covariance = inverse @ centred.T @ centred @ inverse * 12 / 11
    block = many_runs.timing_report(uncertainty="robust")["lag_standard_error"]
    assert block["parameters_at_bounds"] == 0
    b, c = (problem.indices[("filter", m, "lag")] for m in ("B", "C"))
    assert block["relative_lag"]["B"]["robust_standard_error"] == pytest.approx(
        np.sqrt(covariance[b, b]), rel=1e-6
    )
    assert block["pairwise_lag_difference"]["C - B"]["robust_standard_error"] == pytest.approx(
        np.sqrt(covariance[b, b] + covariance[c, c] - 2 * covariance[b, c]), rel=1e-6
    )


@pytest.mark.parametrize(
    "linear_algebra, features",
    [("grouped", 1), ("grouped", 2), ("dense", 1), ("dense", 2), ("state_space", 1)],
)
def test_run_scores_sum_to_the_prior_gradient_at_the_map(linear_algebra, features, many_runs):
    import jax
    import jax.numpy as jnp

    if (linear_algebra, features) == ("grouped", 1):
        model = many_runs
    else:
        data = recording(runs=4, duration=12.0, features=features, seed=1)
        model = fit(data, features=features, linear_algebra=linear_algebra, search=TIGHT)
    problem = model.problem_
    x = jnp.asarray(model.map_parameters_)
    scores = run_scores(problem, x)
    assert scores.shape == (len(problem.systems), len(problem.names))
    # The decomposition itself, away from any optimum.
    shifted = x + 0.01
    total = sum(float(problem.run_nll(shifted, run)) for run in problem.systems)
    assert total == pytest.approx(float(problem.nll(shifted)), rel=1e-12)
    np.testing.assert_allclose(
        run_scores(problem, shifted).sum(axis=0),
        np.asarray(jax.grad(problem.nll)(shifted)),
        rtol=1e-8,
        atol=1e-8,
    )
    # First-order condition of the penalized fit: sum_b g_b = grad log prior.
    prior = np.asarray(jax.grad(problem.log_prior)(x))
    assert np.abs(prior).max() > 1e-3
    assert np.abs(scores.sum(axis=0) - prior).max() < 1e-4 * np.abs(scores).max()
    # No run's score is negligible, so the check is not vacuous.
    assert np.abs(scores).max(axis=1).min() > 1e3 * np.abs(scores.sum(axis=0) - prior).max()
    if linear_algebra != "state_space":
        block = model.timing_report(uncertainty="robust")["lag_standard_error"]
        assert block["clusters"] == len(problem.systems)
        assert block["few_clusters"] is (len(problem.systems) < 10)
        assert block["null_directions"] == features * (features - 1) // 2
        assert block["positive_definite"] is True
        for name in ("B", "C"):
            assert np.isfinite(block["relative_lag"][name]["robust_standard_error"])
            assert block["relative_lag"][name]["robust_standard_error"] > 0


def test_fewer_than_three_runs_raise_and_name_the_split_helper():
    data = recording(runs=1, duration=24.0)
    model = fit(split_runs(data, 2), search=dict(maxiter=20))
    assert len(model.problem_.systems) == 2
    with pytest.raises(ValueError, match="at least 3; this fit has 2.*split_runs"):
        model.timing_report(uncertainty="robust")
    with pytest.raises(ValueError, match="needs at least 3"):
        lag_standard_errors(model, "A", robust=True)
    # The Laplace report does not need clusters.
    assert model.timing_report(uncertainty="laplace")["lag_standard_error"]["method"] == "laplace"


def test_a_lag_at_its_bound_has_no_robust_error(many_runs):
    index = many_runs.problem_.indices[("filter", "B", "lag")]
    parameters = many_runs.map_parameters_
    edge = np.array(parameters, dtype=float)
    edge[index] = 2.0
    try:
        many_runs.map_parameters_ = edge
        block = many_runs.timing_report(uncertainty="robust")["lag_standard_error"]
    finally:
        many_runs.map_parameters_ = parameters
    assert block["lags_at_bounds"] == ["B"]
    for entry in (
        block["relative_lag"]["B"],
        block["lag_model_clock"]["B"],
        block["pairwise_lag_difference"]["C - B"],
    ):
        assert np.isnan(entry["standard_error"]) and np.isnan(entry["robust_standard_error"])
    assert np.isfinite(block["relative_lag"]["C"]["robust_standard_error"])


def test_split_runs_keeps_every_sample_and_its_mask():
    rng = np.random.default_rng(0)
    times = {"A": np.arange(0.0, 30.0, 0.5), "B": np.arange(0.25, 31.0, 1.0)}
    data = {}
    for subject in ("s0", "s1"):
        for run in ("first", "second"):
            streams = {}
            for modality, t in times.items():
                values = rng.normal(size=(len(t), 2))
                mask = rng.uniform(size=values.shape) > 0.2
                values[~mask] = np.nan
                streams[modality] = TimeSeries(values, t, mask)
            data.setdefault(subject, {})[run] = streams
    # One subject has a shorter recording; block edges must still be shared.
    short = data["s1"]["second"]["A"]
    data["s1"]["second"]["A"] = TimeSeries(short.values[:20], short.times[:20], short.mask[:20])
    blocks = split_runs(data, 4)
    for subject, runs in data.items():
        assert (
            list(blocks[subject]) == [f"{run}/block-{k:02d}" for run in runs for k in range(1, 5)]
            or subject == "s1"
        )
        for run, streams in runs.items():
            for modality, series in streams.items():
                parts = [
                    block[modality]
                    for name, block in blocks[subject].items()
                    if name.startswith(f"{run}/") and modality in block
                ]
                np.testing.assert_array_equal(
                    np.concatenate([p.times for p in parts]), series.times
                )
                np.testing.assert_array_equal(np.concatenate([p.mask for p in parts]), series.mask)
                np.testing.assert_array_equal(
                    np.concatenate([p.values for p in parts]), series.values
                )
    # Blocks are contiguous and cover the same stretch of time for everyone.
    edges = np.linspace(0.0, 30.25, 5)
    for subject in blocks:
        for k in range(4):
            for series in blocks[subject][f"first/block-{k + 1:02d}"].values():
                assert series.times[0] >= edges[k] and series.times[-1] <= edges[k + 1]
    # The short series ends at 9.5 s, so it is absent from the later blocks.
    assert "A" in blocks["s1"]["second/block-01"] and "A" in blocks["s1"]["second/block-02"]
    assert "A" not in blocks["s1"]["second/block-03"]
    assert "A" not in blocks["s1"]["second/block-04"]


def test_split_runs_accepts_short_nesting_and_validates_blocks():
    series = TimeSeries(np.arange(20.0)[:, None], np.arange(20.0))
    blocks = split_runs({"s0": {"A": series}}, 2)
    assert list(blocks["s0"]) == ["run-01/block-01", "run-01/block-02"]
    assert [len(b["A"].times) for b in blocks["s0"].values()] == [10, 10]
    assert split_runs({"s0": {"A": series}}, 1) == {"s0": {"run-01": {"A": series}}}
    for bad in (0, 2.0, True, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            split_runs({"s0": {"A": series}}, bad)
