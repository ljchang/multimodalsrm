import numpy as np
import pytest


def _fit(kernel, pooling="shared"):
    from multimodalsrm import MultimodalSRM, Response, TimeSeries

    times = np.arange(16, dtype=float)
    data = {
        subject: {"rating": TimeSeries(np.sin(times / 3)[:, None], times)} for subject in ("a", "b")
    }
    return MultimodalSRM(
        features=1,
        latent_dt=1,
        responses={"rating": Response(kernel, pooling=pooling, estimate=False)},
        max_iter=30,
        tol=1e-3,
        random_state=0,
    ).fit(data)


def test_kernel_inspection_preserves_fixed_metadata_and_group_differences():
    from multimodalsrm import Gaussian

    model = _fit(Gaussian(width=0.2), "partial")
    lags = np.linspace(-2, 2, 41)
    curve = model.kernel("rating", subject="a", times=lags)
    assert curve.family == "Gaussian"
    assert curve.parameters["width"] == 0.2
    assert curve.fixed == {"width": 0.2, "lag": 0.0}
    assert curve.learned == ()
    assert curve.normalization == "continuous_l2"
    np.testing.assert_array_equal(curve.times, lags)
    np.testing.assert_allclose(curve.difference_from_group, 0)
    assert curve.values.shape == (41, 1)
    assert curve.valid.all()


def test_identity_inspection_reports_impulse_without_fabricating_density():
    from multimodalsrm import Identity

    model = _fit(Identity())
    curve = model.kernel("rating", level="group", times=np.linspace(-1, 1, 11))
    assert curve.impulse_mass == 1.0
    assert curve.values is None
    assert curve.support == (0.0, 0.0)
    axis = model.plot_kernels("rating")
    assert "seconds" in axis.get_xlabel().lower()
    import matplotlib.pyplot as plt

    plt.close(axis.figure)


def test_independent_kernel_group_query_is_rejected():
    from multimodalsrm import Gaussian

    model = _fit(Gaussian(width=0.2), "none")
    with pytest.raises(ValueError, match="group|independent"):
        model.kernel("rating", level="group")


def test_training_latents_identify_observed_and_model_conditioned_coverage():
    from multimodalsrm import Identity

    model = _fit(Identity())
    result = model.training_latents_["a"]["run-01"]
    assert ("a", "run-01", "rating") in result.metadata["used_sources"]
    assert result.metadata["observed_at_query"].all()
    assert result.metadata["model_conditioned"]
