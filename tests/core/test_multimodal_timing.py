"""Regression evidence for the explicit lag-only response baseline."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import (
    BachSCR,
    Gaussian,
    Identity,
    MultimodalSRM,
    Normal,
    Response,
    SampledKernel,
    TimeSeries,
)
from multimodalsrm.kernel_optimization import (
    kernel_penalty,
    optimize_kernels,
)


def _quadratic_fit(response, targets):
    kernels = {subject: {"brain": response.initial_kernel()} for subject in targets}

    def objective(candidate):
        data_term = sum(
            (candidate[subject]["brain"].lag - target) ** 2 for subject, target in targets.items()
        )
        return data_term + kernel_penalty(candidate, {"brain": response})

    fitted, diagnostic = optimize_kernels(kernels, {"brain": response}, objective, max_iter=100)
    assert diagnostic["accepted"]
    return fitted


def _lag_dispersion(kernels):
    return np.std([mods["brain"].lag for mods in kernels.values()])


def test_lag_only_real_optimizer_keeps_every_shape_parameter_fixed():
    """Omitting a non-lag fixed value would let the real optimizer move it."""
    template = BachSCR(t0=3.2, sigma=0.8, lambda1=0.35, lambda2=0.08)
    response = Response.lag_only(template, pooling="none", bounds={"lag": (-2.0, 2.0)})
    fitted = _quadratic_fit(response, {"s1": -1.0, "s2": 0.5, "s3": 1.5})

    assert response.free_parameters == ("lag",)
    for subject, target in {"s1": -1.0, "s2": 0.5, "s3": 1.5}.items():
        assert fitted[subject]["brain"].lag == pytest.approx(target, abs=2e-5)
        for name in ("t0", "sigma", "lambda1", "lambda2"):
            assert getattr(fitted[subject]["brain"], name) == pytest.approx(
                getattr(template, name), abs=1e-15
            )
    assert template.lag == 0.0


def test_lag_only_partial_pooling_shrinks_subject_deviations():
    """Dropping the partial-pooling penalty would match the unpooled spread."""
    targets = {"s1": -1.5, "s2": 0.0, "s3": 1.5}
    none = _quadratic_fit(Response.lag_only(Gaussian(width=0.4), pooling="none"), targets)
    partial = _quadratic_fit(
        Response.lag_only(Gaussian(width=0.4), pooling="partial", pooling_strength=2.0),
        targets,
    )
    stronger = _quadratic_fit(
        Response.lag_only(Gaussian(width=0.4), pooling="partial", pooling_strength=20.0),
        targets,
    )

    assert _lag_dispersion(partial) < _lag_dispersion(none)
    assert _lag_dispersion(stronger) < _lag_dispersion(partial)


def test_lag_only_bounds_prior_metadata_and_rejections():
    """Failing to forward timing controls or validate a lag would break the API."""
    prior = Normal(0.25, 0.5)
    response = Response.lag_only(
        Gaussian(width=0.3, lag=0.2),
        bounds={"lag": (-0.5, 0.75)},
        lag_prior=prior,
    )
    assert response.parameter_bounds()["lag"] == (-0.5, 0.75)
    assert response.metadata["lag_prior"] is prior
    assert response.metadata["free_parameters"] == ("lag",)
    assert response.metadata["timing_confounds"] == []

    with pytest.raises(ValueError, match="lag parameter"):
        Response.lag_only(Identity())
    with pytest.raises(ValueError, match="lag parameter"):
        Response.lag_only(SampledKernel([0, 1], [1, -1]))


def test_general_bach_warning_and_metadata_use_effective_free_parameters():
    """Warning on family alone would incorrectly flag a fixed-t0 response."""
    with pytest.warns(UserWarning, match="t0.*lag"):
        general = Response(BachSCR())
    assert general.metadata["timing_confounds"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fixed_t0 = Response(BachSCR(), fixed={"t0": 3.0745})
        fixed_lag = Response(BachSCR(), fixed={"lag": 0.0})
        inactive = Response(BachSCR(), estimate=False)
        baseline = Response.lag_only(BachSCR())
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    for response in (fixed_t0, fixed_lag, inactive, baseline):
        assert response.metadata["timing_confounds"] == []


def test_multirun_fit_has_one_lag_per_subject_and_modality():
    """Creating run-specific response state would expose duplicate lag estimates."""
    times = np.arange(20.0)
    data = {
        subject: {
            run: {
                "brain": TimeSeries(np.sin((times - offset)[:, None]), times),
                "reference": TimeSeries(np.sin(times[:, None]), times),
            }
            for run in ("run-a", "run-b")
        }
        for subject, offset in (("s1", -0.25), ("s2", 0.25))
    }
    model = MultimodalSRM(
        features=1,
        latent_dt=1.0,
        max_iter=2,
        kernel_max_iter=8,
        random_state=4,
        responses={
            "brain": Response.lag_only(Gaussian(width=0.5), bounds={"lag": (-0.75, 0.75)}),
            "reference": Response(Identity(), estimate=False),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(data)

    assert set(model.subject_kernels_) == {"s1", "s2"}
    assert all(
        set(modalities) == {"brain", "reference"} for modalities in model.subject_kernels_.values()
    )
    assert all(
        isinstance(modalities["brain"].lag, float) for modalities in model.subject_kernels_.values()
    )
