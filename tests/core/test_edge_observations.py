"""Scoring observations whose response support reaches outside the run."""

import warnings

import numpy as np
import pytest

from multimodalsrm import Gamma, Gaussian, Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.objective import response_reach


def recording(seed, duration=60.0, subjects=("a", "b", "c")):
    """A direct stream and a stream delayed and smoothed by a causal Gamma(3, 1)."""
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.02, 0.2, 10)
    amplitude, phase = rng.normal(size=10), rng.uniform(0, 6.28, 10)
    lags = np.arange(0, 25, 0.05) + 0.025
    kernel = lags**2 * np.exp(-lags)
    kernel /= kernel.sum()

    def latent(times):
        return (amplitude * np.sin(2 * np.pi * frequency * np.asarray(times)[:, None] + phase)).sum(
            -1
        )

    def delayed(times):
        return sum(weight * latent(times - lag) for weight, lag in zip(kernel, lags))

    direct_times, slow_times = np.arange(0, duration, 0.5), np.arange(0.25, duration, 1.0)
    data, truth = {}, {}
    for index, subject in enumerate(subjects):
        # Maps belong to the participant and persist across runs; noise does not.
        maps = np.random.default_rng([99, index])
        direct_map, slow_map = maps.normal(size=(1, 3)), maps.normal(size=(1, 4))
        local = np.random.default_rng([seed, index])
        clean = delayed(slow_times)[:, None] * slow_map
        data[subject] = {
            f"run{seed}": {
                "direct": TimeSeries(
                    latent(direct_times)[:, None] * direct_map
                    + local.normal(size=(len(direct_times), 3)) * 0.3,
                    direct_times,
                ),
                "slow": TimeSeries(clean + local.normal(size=clean.shape) * 0.3, slow_times),
            }
        }
        truth[subject] = clean
    return data, truth


def responses(slow=None):
    return {
        "direct": Response(Identity(), estimate=False),
        "slow": slow or Response(Gamma(3.0, 1.0), estimate=False),
    }


def fit(data, **settings):
    options = dict(features=1, latent_dt=0.5, responses=responses(), max_iter=200, random_state=0)
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultimodalSRM(**options).fit(data)


def merged(*recordings):
    data = {}
    for item in recordings:
        for subject, runs in item.items():
            data.setdefault(subject, {}).update(runs)
    return data


@pytest.fixture(scope="module")
def models():
    train = merged(recording(1)[0], recording(2)[0])
    return fit(train, edge_observations="exclude"), fit(train)


def scored(model, modality):
    blocks = [b for b in model.observation_blocks_ if b.modality == modality]
    return sum(int(b.valid.sum()) for b in blocks), sum(len(b.valid) for b in blocks)


def test_reach_follows_the_support_envelope():
    gamma = Response(Gamma(3.0, 1.0), estimate=False)
    assert response_reach({"m": Response(Identity(), estimate=False)}) == (0.0, 0.0)
    before, after = response_reach({"m": gamma})
    assert before == pytest.approx(gamma.support_envelope()[1]) and after == 0.0
    two_sided = Response(Gaussian(1.0), fixed={"width": 1.0}, bounds={"lag": (-3.0, 2.0)})
    assert response_reach({"m": two_sided}) == pytest.approx((8.0, 9.0))
    assert response_reach({"m": gamma, "n": two_sided}, {"m": 0.0, "n": 1.0}) == pytest.approx(
        (8.0, 9.0)
    )


def test_default_scores_every_row_and_exclude_drops_edge_rows(models):
    excluded, included = models
    assert MultimodalSRM(latent_dt=1.0).edge_observations == "include"
    assert included.configuration_["edge_observations"] == "include"
    assert excluded.edge_reach_ == (0.0, 0.0)
    assert excluded.configuration_["edge_observations"] == "exclude"
    used, total = scored(excluded, "slow")
    assert used < 0.7 * total  # A 25 s tail on a 60 s run discards the opening rows.
    assert scored(included, "slow") == (total, total)
    assert scored(included, "direct") == scored(excluded, "direct")
    reach = included.edge_reach_
    assert reach == pytest.approx(response_reach(included.responses_))
    assert included.configuration_["edge_reach"] == reach


def test_padding_extends_the_solve_grid_and_is_cropped_from_reported_latents(models):
    excluded, included = models
    for run, domain in included.run_domains_.items():
        assert domain == excluded.run_domains_[run]
        grid, reach = included.run_grids_[run], included.edge_reach_[0]
        # Whole cells of padding: at least the reach, less than one cell more.
        assert domain[0] - reach - included.latent_dt < grid[0] <= domain[0] - reach
        assert grid[-1] >= domain[1]
        # Like a "same" convolution, results cover the run on its own grid.
        for name in ("training_latents_", "population_latents_"):
            padded, plain = getattr(included, name), getattr(excluded, name)
            padded, plain = (
                (padded["a"][run], plain["a"][run]) if name[0] == "t" else (padded[run], plain[run])
            )
            np.testing.assert_array_equal(padded.times, plain.times)
            np.testing.assert_array_equal(padded.valid, plain.valid)
            assert padded.values.shape == plain.values.shape
            assert np.isfinite(padded.values[padded.valid]).all()
        assert len(included._training_latent_arrays_["a"][run]) == len(grid)


def test_identity_responses_are_unaffected():
    data = merged(recording(1)[0], recording(2)[0])
    identity = {m: Response(Identity(), estimate=False) for m in ("direct", "slow")}
    a = fit(data, responses=identity, edge_observations="exclude")
    b = fit(data, responses=identity)
    np.testing.assert_array_equal(a.objective_history_, b.objective_history_)
    for run in a.run_grids_:
        np.testing.assert_array_equal(a.run_grids_[run], b.run_grids_[run])


def test_included_edge_rows_are_predicted_from_the_other_stream(models):
    excluded, included = models
    test, truth = recording(3)
    times = test["a"]["run3"]["slow"].times
    outputs = {}
    for name, model in (("exclude", excluded), ("include", included)):
        result = model.predict(
            test,
            targets={"a": ["slow"]},
            source="within",
            times={"a": {"run3": {"slow": times}}},
        )["a"]["run3"]["slow"]
        outputs[name] = result
    assert outputs["include"].valid.all()
    assert outputs["exclude"].valid.sum() < 0.7 * len(times)
    shared = outputs["exclude"].valid
    error = lambda result, rows: float(  # noqa: E731
        np.mean((result.values[rows] - truth["a"][rows]) ** 2) / np.var(truth["a"][rows])
    )
    # Rows both modes predict are predicted about equally well.
    assert error(outputs["include"], shared) < error(outputs["exclude"], shared) + 0.05
    assert error(outputs["include"], shared) < 0.3
    # The recovered opening rows carry signal, though less than interior rows.
    assert error(outputs["include"], ~shared) < 0.8


def test_include_keeps_a_learned_lag_and_its_timing_report():
    train = merged(recording(1)[0], recording(2)[0])
    slow = Response(Gamma(3.0, 1.0, 0.5), fixed={"shape": 3.0, "scale": 1.0}, pooling="shared")
    reports = [
        fit(train, responses=responses(slow), edge_observations=edge).timing_report()
        for edge in ("exclude", "include")
    ]
    for report in reports:
        assert report["relative_peak_delay"]["slow"] == pytest.approx(2.0, abs=0.3)
    assert reports[0]["relative_peak_delay"]["slow"] == pytest.approx(
        reports[1]["relative_peak_delay"]["slow"], abs=0.2
    )


def test_calibration_and_latent_inference_work_with_padding(models):
    _, included = models
    data, _ = recording(4, subjects=("a", "b", "c", "new"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = included.calibrate(data)
    assert set(calibrated.loadings_["new"]) == {"direct", "slow"}
    excluded, _ = models
    latent = included.infer_latent(recording(5)[0])["a"]["run5"]
    np.testing.assert_array_equal(
        latent.times, excluded.infer_latent(recording(5)[0])["a"]["run5"].times
    )
    assert latent.times[0] == 0.0 and latent.valid.all()


def test_validation_and_earlier_pickles():
    data = recording(1)[0]
    with pytest.raises(ValueError, match="edge_observations"):
        MultimodalSRM(latent_dt=0.5, edge_observations="pad").fit(data)
    model = fit(data, max_iter=3, edge_observations="exclude")
    state = dict(model.__getstate__())
    for name in ("edge_observations", "edge_reach_"):
        state.pop(name)
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert restored.edge_observations == "exclude" and restored.edge_reach_ == (0.0, 0.0)
    expected = model.infer_latent(recording(2)[0])["a"]["run2"].values
    np.testing.assert_array_equal(
        restored.infer_latent(recording(2)[0])["a"]["run2"].values, expected
    )
