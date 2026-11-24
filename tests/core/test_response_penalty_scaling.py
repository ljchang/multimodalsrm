"""Likelihood-scaled response priors and pooling penalties."""

import copy
import warnings

import numpy as np
import pytest

from multimodalsrm import Gaussian, MultimodalSRM, Normal, Response, TimeSeries
from multimodalsrm.kernel_optimization import (
    _kernel_penalty_gradient,
    kernel_penalty,
    likelihood_penalty_scale,
)
from multimodalsrm.kernels import KernelPrior

WIDTHS = {"A": 0.4, "B": 0.8}
TRUE = {"A": 0.0, "B": 1.0}


def recording(seed, subjects=3, offsets=None):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.02, 0.3, 12)
    amplitude, phase = rng.normal(size=12), rng.uniform(0, 6.28, 12)
    clocks = {"A": np.arange(0, 100, 0.5), "B": np.arange(0.25, 100, 0.5)}
    data = {}
    for index in range(subjects):
        maps = np.random.default_rng([5, index])
        streams = {}
        for modality, times in clocks.items():
            shift = TRUE[modality] + (offsets[index] if offsets and modality == "B" else 0.0)
            angle = 2 * np.pi * frequency * (times - shift)[:, None] + phase
            signal = (amplitude * np.sin(angle)).sum(-1)
            values = signal[:, None] * maps.normal(size=(1, 4))
            streams[modality] = TimeSeries(values + rng.normal(size=values.shape), times)
        data[f"s{index}"] = {"run": streams}
    return data


def responses(sd=None, pooling="shared", strength=1.0):
    return {
        m: Response(
            Gaussian(w),
            pooling=pooling,
            fixed={"width": w},
            lag_prior=None if sd is None else Normal(0.0, sd),
            pooling_strength=strength,
        )
        for m, w in WIDTHS.items()
    }


def fit(data, **settings):
    options = dict(features=1, latent_dt=0.5, max_iter=300, random_state=0)
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultimodalSRM(**options).fit(data)


def difference(model):
    return model.timing_report()["pairwise_difference"]["B - A"]


def test_scale_is_the_objective_weight_per_scored_value():
    model = fit(recording(0), responses=responses(1.0), response_penalty_scaling="likelihood")
    scale = likelihood_penalty_scale(model.observation_blocks_)
    assert model.penalty_scale_ == pytest.approx(scale)
    counts = {m: 0 for m in scale}
    for b in model.observation_blocks_:
        counts[b.modality] += int(np.count_nonzero(b.coefficients))
    # Undoing the scale recovers each modality's share of the unit loss weight.
    assert sum(scale[m] * counts[m] / 2 for m in scale) == pytest.approx(1.0)
    assert likelihood_penalty_scale(model.observation_blocks_, 0.25) == pytest.approx(
        {m: v / 4 for m, v in scale.items()}
    )
    assert model.configuration_["response_penalty_scaling"] == "likelihood"
    assert model.configuration_["response_penalty_scale"] == pytest.approx(scale)
    # The existing description of penalty normalization is still recorded.
    assert "loadings" in model.configuration_["penalty_scaling"]


def test_likelihood_scaling_is_the_default():
    assert MultimodalSRM(latent_dt=0.5).response_penalty_scaling == "likelihood"
    model = fit(recording(0), responses=responses(1.0))
    assert model.penalty_scale_ is not None
    assert model.configuration_["response_penalty_scaling"] == "likelihood"


def test_objective_scaling_keeps_the_original_unscaled_penalties():
    model = fit(recording(0), responses=responses(1.0), response_penalty_scaling="objective")
    assert model.response_penalty_scaling == "objective" and model.penalty_scale_ is None
    kernels = model.subject_kernels_
    assert kernel_penalty(kernels, model.responses_) == kernel_penalty(
        kernels, model.responses_, None
    )


@pytest.mark.parametrize("pooling", ["shared", "partial", "none"])
def test_scaled_penalty_and_its_gradient(pooling):
    config = {
        "A": Response(
            Gaussian(0.5),
            pooling=pooling,
            lag_prior=Normal(0.3, 0.7),
            prior=KernelPrior({"width": 0.8}, strength=2.0),
            pooling_strength=3.0,
            bounds={"width": (0.2, 2.0)},
        )
    }
    lags = (0.1, 0.4, -0.2) if pooling != "shared" else (0.2, 0.2, 0.2)
    kernels = {
        f"s{i}": {"A": Gaussian(0.5 + 0.1 * i * (pooling != "shared"), lag)}
        for i, lag in enumerate(lags)
    }
    scale = {"A": 0.01}
    coordinates = [k["A"].to_coordinates() for k in kernels.values()]
    center = {p: np.mean([c[p] for c in coordinates]) for p in ("width", "lag")}
    expected = 0.0
    if pooling == "partial":
        expected += 0.5 * 3.0 * sum((c[p] - center[p]) ** 2 for c in coordinates for p in center)
    targets = coordinates if pooling == "none" else [center]
    for target in targets:
        expected += 0.5 * ((target["lag"] - 0.3) / 0.7) ** 2
        expected += 0.5 * 2.0 * (target["width"] - np.log(0.8)) ** 2
    assert kernel_penalty(kernels, config, scale) == pytest.approx(0.01 * expected)
    gradient = _kernel_penalty_gradient(kernels, config, scale)
    for subject in kernels:
        for parameter in ("width", "lag"):
            step = 1e-6
            values = []
            for sign in (-1, 1):
                moved = copy.deepcopy(kernels)
                targets = kernels if pooling == "shared" else [subject]
                for name in targets:
                    c = moved[name]["A"].to_coordinates()
                    c[parameter] += sign * step
                    moved[name]["A"] = moved[name]["A"].from_coordinates(c)
                values.append(kernel_penalty(moved, config, scale))
            numeric = (values[1] - values[0]) / (2 * step)
            analytic = (
                sum(gradient[name, "A", parameter] for name in kernels)
                if pooling == "shared"
                else gradient[subject, "A", parameter]
            )
            assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-9)


def test_a_one_second_prior_no_longer_overrides_the_data():
    data = recording(0)
    unpenalized = difference(fit(data, responses=responses()))
    assert unpenalized == pytest.approx(1.0, abs=0.05)
    original = difference(fit(data, responses=responses(1.0), response_penalty_scaling="objective"))
    scaled = difference(fit(data, responses=responses(1.0)))
    assert original < 0.6  # A nominal 1 s prior pulls a 1 s delay most of the way to zero.
    assert scaled == pytest.approx(unpenalized, abs=0.01)


def test_scaled_prior_tightens_smoothly_with_its_standard_deviation():
    data = recording(1)
    values = [difference(fit(data, responses=responses(sd))) for sd in (1.0, 0.1, 0.03, 0.01)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[0] > 0.95 and values[-1] < 0.5
    original = [
        difference(fit(data, responses=responses(sd), response_penalty_scaling="objective"))
        for sd in (0.3, 0.1)
    ]
    assert max(original) < 0.1


def test_scaled_partial_pooling_keeps_participant_differences():
    offsets = (-0.3, 0.0, 0.3)
    data = recording(2, offsets=offsets)

    def spread(**settings):
        model = fit(data, responses=responses(pooling="partial", strength=1 / 0.3**2), **settings)
        lags = [
            model.subject_kernels_[s]["B"].lag - model.subject_kernels_[s]["A"].lag
            for s in model.subjects_
        ]
        return np.std(lags)

    independent = fit(data, responses=responses(pooling="none"))
    reference = np.std(
        [
            independent.subject_kernels_[s]["B"].lag - independent.subject_kernels_[s]["A"].lag
            for s in independent.subjects_
        ]
    )
    assert reference == pytest.approx(np.std(offsets), abs=0.08)
    # Unscaled: effectively complete pooling.
    assert spread(response_penalty_scaling="objective") < 0.2 * reference
    assert spread() > 0.8 * reference


def test_calibration_uses_the_scaled_penalty():
    data = recording(0, subjects=4)
    train = {s: runs for s, runs in data.items() if s != "s3"}
    model = fit(
        train,
        responses=responses(1.0, pooling="partial"),
        response_penalty_scaling="likelihood",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = model.calibrate(data)
    lag = calibrated.subject_kernels_["s3"]["B"].lag - calibrated.subject_kernels_["s3"]["A"].lag
    assert lag == pytest.approx(1.0, abs=0.2)


def test_validation_and_earlier_pickles():
    with pytest.raises(ValueError, match="response_penalty_scaling"):
        MultimodalSRM(latent_dt=0.5, response_penalty_scaling="nll").fit(recording(0))
    model = fit(recording(0), responses=responses(), max_iter=3)
    state = dict(model.__getstate__())
    for name in ("response_penalty_scaling", "penalty_scale_"):
        state.pop(name)
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert restored.response_penalty_scaling == "objective" and restored.penalty_scale_ is None


def test_responses_are_shared_across_participants_by_default():
    from multimodalsrm import Gaussian, Response

    assert Response(Gaussian()).pooling == "shared"
    assert Response.lag_only(Gaussian()).pooling == "shared"
    assert Response(Gaussian(), pooling="partial").pooling == "partial"
