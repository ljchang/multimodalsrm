"""Response-step stopping rule and safeguarded response extrapolation."""

import pickle
import warnings

import numpy as np
import pytest

from multimodalsrm import Gaussian, Identity, MultimodalSRM, Response, TimeSeries
from multimodalsrm.kernel_optimization import (
    coordinate_step,
    extrapolated_kernels,
    free_coordinates,
)


def recording():
    rng = np.random.default_rng(11)
    fine = np.arange(0, 80, 0.25)
    latent = np.convolve(rng.normal(size=len(fine) + 40), np.hanning(41), mode="valid")[: len(fine)]
    latent /= latent.std()
    shifted = np.interp(fine - 0.9, fine, latent)
    smooth = np.convolve(shifted, np.exp(-0.5 * (np.arange(-12, 13) * 0.25 / 0.7) ** 2), "same")
    smooth /= smooth.std()
    data = {}
    for index, subject in enumerate("abc"):
        local = np.random.default_rng(index)
        reference = latent[::4, None] * local.normal(size=(1, 3)) + local.normal(size=(80, 3)) * 0.2
        signal = smooth[::6, None] * local.normal(size=(1, 4))
        signal = signal + local.normal(size=signal.shape) * 0.2
        data[subject] = {
            "run": {
                "reference": TimeSeries(reference, fine[::4]),
                "signal": TimeSeries(signal, fine[::6]),
            }
        }
    return data


def responses():
    return {
        "reference": Response(Identity(), estimate=False),
        "signal": Response(
            Gaussian(width=0.5, lag=0.2),
            pooling="shared",
            bounds={"width": (0.2, 2.0), "lag": (-1.0, 2.5)},
        ),
    }


def fit(**settings):
    options = dict(
        features=1,
        latent_dt=0.5,
        responses=responses(),
        temporal_strength=0.1,
        max_iter=400,
        random_state=0,
    )
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MultimodalSRM(**options).fit(recording())


@pytest.mark.parametrize("value", [0, -1e-3, np.inf, np.nan, True, "1e-4"])
def test_kernel_tol_validation(value):
    with pytest.raises(ValueError, match="kernel_tol"):
        MultimodalSRM(latent_dt=0.5, kernel_tol=value).fit(recording())


def test_kernel_extrapolation_must_be_boolean():
    with pytest.raises(ValueError, match="kernel_extrapolation"):
        MultimodalSRM(latent_dt=0.5, kernel_extrapolation=1).fit(recording())


def test_convergence_requires_a_settled_response():
    model = fit()
    steps = [d["kernel_step"] for d in model.restart_diagnostics_[0]["diagnostics"]]
    assert model.converged_
    assert steps[-1] <= model.kernel_tol
    assert model.configuration_["kernel_tol"] == 1e-4
    assert model.configuration_["stopping_rule"].endswith("_and_kernel_step")
    # The objective-only rule accepts the first small decrement, however far
    # the response moved in that iteration, so it can never run longer.
    legacy = fit(kernel_tol=None)
    assert legacy.configuration_["stopping_rule"].endswith("kernel_optimizer_success")
    assert legacy.n_iter_ <= model.n_iter_
    assert model.objective_history_[-1] <= legacy.objective_history_[-1] + 1e-12


def test_extrapolation_is_monotone_and_reaches_the_same_response():
    plain = fit(kernel_extrapolation=False, tol=1e-9, kernel_tol=1e-6)
    fast = fit(tol=1e-9, kernel_tol=1e-6)
    for model in (plain, fast):
        assert model.converged_
        assert np.all(np.diff(model.objective_history_) <= 1e-10)
    records = fast.restart_diagnostics_[0]["diagnostics"]
    assert all(
        d["kernel_extrapolation"]["accepted_steps"] == 0
        for d in plain.restart_diagnostics_[0]["diagnostics"]
    )
    assert sum(d["kernel_extrapolation"]["accepted_steps"] for d in records) > 0
    assert fast.n_iter_ <= plain.n_iter_
    a, b = plain.group_kernels_["signal"].parameters, fast.group_kernels_["signal"].parameters
    assert a == pytest.approx(b, abs=2e-3)
    assert fast.objective_history_[-1] == pytest.approx(plain.objective_history_[-1], rel=1e-5)


def test_fixed_responses_keep_the_objective_only_behaviour():
    fixed = {m: Response(Identity(), estimate=False) for m in ("reference", "signal")}
    a = fit(responses=fixed)
    b = fit(responses=fixed, kernel_tol=None, kernel_extrapolation=False)
    assert a.n_iter_ == b.n_iter_
    np.testing.assert_array_equal(a.objective_history_, b.objective_history_)
    assert all(d["kernel_step"] == 0.0 for d in a.restart_diagnostics_[0]["diagnostics"])


def test_extrapolated_kernels_respect_bounds_and_sharing():
    config = responses()
    kernels = {s: {"signal": Gaussian(width=0.5, lag=2.4), "reference": Identity()} for s in "ab"}
    before = free_coordinates(kernels, config)
    assert set(before) == {(s, "signal", p) for s in "ab" for p in ("width", "lag")}
    direction = {key: (0.3 if key[2] == "lag" else np.log(1.5)) for key in before}
    moved = extrapolated_kernels(kernels, config, direction, 2.0)
    assert moved["a"]["signal"] == moved["b"]["signal"]
    assert moved["a"]["signal"].lag == 2.5  # Clipped at the upper bound.
    assert moved["a"]["signal"].width == pytest.approx(0.5 * 1.5**2)
    assert moved["a"]["reference"] is not None and kernels["a"]["signal"].lag == 2.4
    assert coordinate_step(before, free_coordinates(moved, config)) == pytest.approx(
        2 * np.log(1.5)
    )
    at_bounds = {s: {"signal": Gaussian(width=2.0, lag=2.5), "reference": Identity()} for s in "ab"}
    assert extrapolated_kernels(at_bounds, config, direction, 1.0) is None
    assert extrapolated_kernels(kernels, config, {key: 0.0 for key in before}, 4.0) is None


def test_earlier_pickles_restore_the_objective_only_rule():
    model = fit(max_iter=3)
    state = dict(model.__getstate__())
    for name in ("kernel_tol", "kernel_extrapolation"):
        state.pop(name)
    restored = MultimodalSRM.__new__(MultimodalSRM)
    restored.__setstate__(state)
    assert restored.kernel_tol is None and restored.kernel_extrapolation is False
    assert restored.get_params()["kernel_tol"] is None
    current = pickle.loads(pickle.dumps(model))
    assert current.kernel_tol == 1e-4 and current.kernel_extrapolation is True
