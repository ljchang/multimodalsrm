"""The latent smoothness prior is specified as a correlation time."""

import warnings

import numpy as np
import pytest

from multimodalsrm import MultimodalSRM, TimeSeries


def recording(interval=1.0):
    rng = np.random.default_rng(21)
    times = np.arange(0, 60.01, interval)
    latent = np.sin(2 * np.pi * times / 17) + 0.6 * np.cos(2 * np.pi * times / 5 + 1.0)
    data = {}
    for subject in "abc":
        values = latent[:, None] * rng.normal(size=(1, 4)) + rng.normal(size=(len(times), 4))
        data[subject] = {"run": {"m": TimeSeries(values, times)}}
    return data


def fit(**settings):
    options = dict(features=1, latent_dt=0.5, random_state=0, max_iter=200)
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultimodalSRM(**options).fit(recording())


def roughness(model):
    z = model._training_latent_arrays_["a"]["run"]
    return float(np.mean(np.diff(z, axis=0) ** 2) / np.var(z))


def test_defaults_resolve_to_a_two_unit_correlation_time():
    model = fit()
    assert (model.latent_ridge_, model.temporal_strength_) == (0.1, pytest.approx(0.4))
    penalty = model.configuration_["latent_penalty"]
    assert penalty["latent_timescale"] == pytest.approx(2.0)
    assert penalty["parameterization"] == "latent_timescale"
    assert model.get_params()["latent_timescale"] is None


def test_timescale_is_the_square_root_ratio_of_the_two_strengths():
    new = fit(latent_ridge=0.02, latent_timescale=3.0)
    old = fit(latent_ridge=0.02, temporal_strength=0.02 * 3.0**2)
    np.testing.assert_array_equal(new.objective_history_, old.objective_history_)
    np.testing.assert_array_equal(
        new._training_latent_arrays_["a"]["run"], old._training_latent_arrays_["a"]["run"]
    )
    assert old.configuration_["latent_penalty"]["parameterization"] == "temporal_strength"


def test_deprecated_strength_keeps_the_earlier_ridge_default_and_warns():
    with pytest.warns(FutureWarning, match="latent_timescale"):
        model = MultimodalSRM(
            features=1, latent_dt=0.5, temporal_strength=1e-2, random_state=0, max_iter=200
        ).fit(recording())
    assert (model.latent_ridge_, model.temporal_strength_) == (1e-4, 1e-2)
    assert model.configuration_["latent_penalty"]["latent_timescale"] == pytest.approx(10.0)


def test_both_parameterizations_together_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        MultimodalSRM(latent_dt=0.5, latent_timescale=2.0, temporal_strength=0.1).fit(recording())


@pytest.mark.parametrize(
    "settings",
    [
        dict(latent_timescale=-1.0),
        dict(latent_timescale=np.inf),
        dict(latent_timescale="2"),
        dict(latent_ridge=0.0),
        dict(latent_ridge=True),
        dict(temporal_strength=-0.1),
    ],
)
def test_penalty_settings_are_validated(settings):
    name = next(iter(settings))
    with pytest.raises(ValueError, match=name):
        MultimodalSRM(latent_dt=0.5, **settings).fit(recording())


def test_longer_timescales_give_smoother_latents():
    values = [roughness(fit(latent_timescale=t)) for t in (0.0, 0.5, 2.0, 8.0)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_zero_timescale_leaves_unobserved_nodes_unconstrained():
    # Observations every second on a half-second grid: every other node is free.
    free = fit(latent_timescale=0.0)._training_latent_arrays_["a"]["run"][1:-1:2]
    tied = fit()._training_latent_arrays_["a"]["run"]
    assert np.abs(free).max() < 1e-8
    assert np.abs(tied[1:-1:2]).max() > 0.1 * np.abs(tied).max()


def test_timescale_is_in_timestamp_units():
    def scaled(factor, timescale):
        data = {
            s: {"run": {"m": TimeSeries(runs["run"]["m"].values, runs["run"]["m"].times * factor)}}
            for s, runs in recording().items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return MultimodalSRM(
                features=1,
                latent_dt=0.5 * factor,
                latent_timescale=timescale,
                random_state=0,
                max_iter=200,
            ).fit(data)

    seconds, milliseconds = scaled(1.0, 2.0), scaled(1000.0, 2000.0)
    np.testing.assert_allclose(
        milliseconds._training_latent_arrays_["a"]["run"],
        seconds._training_latent_arrays_["a"]["run"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_new_run_inference_reuses_the_resolved_penalty():
    model = fit(latent_timescale=3.0)
    latent = model.infer_latent(recording())["a"]["run"]
    # Training latents precede the last map update, hence the loose tolerance.
    np.testing.assert_allclose(
        latent.values, model._training_latent_arrays_["a"]["run"], rtol=1e-3, atol=1e-5
    )
    rougher = fit(latent_timescale=0.3).infer_latent(recording())["a"]["run"]
    assert np.abs(rougher.values - latent.values).max() > 1e-3


def test_earlier_pickles_keep_their_fitted_penalty():
    model = fit(temporal_strength=1e-2, max_iter=3)
    state = dict(model.__getstate__())
    for name in ("latent_timescale", "latent_ridge_", "temporal_strength_"):
        state.pop(name)
    state["latent_ridge"] = 1e-4  # Earlier constructors stored the numeric default.
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert (restored.latent_ridge_, restored.temporal_strength_) == (1e-4, 1e-2)
    assert restored.latent_timescale is None
    expected = model.infer_latent(recording())["a"]["run"].values
    np.testing.assert_array_equal(restored.infer_latent(recording())["a"]["run"].values, expected)
