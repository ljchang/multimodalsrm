import numpy as np
import pytest


def _smooth(rng, samples, count):
    kernel = np.exp(-0.5 * (np.arange(-30, 31) / 4.0) ** 2)
    z = rng.normal(size=(samples + 60, count))
    z = np.apply_along_axis(lambda column: np.convolve(column, kernel, "valid"), 0, z)
    return (z - z.mean(0)) / z.std(0)


def _fit(*, specific=True, mask=False, pooling="shared"):
    from multimodalsrm import Gaussian, Identity, MultimodalSRM, Response, TimeSeries

    rng = np.random.default_rng(0)
    samples = 240
    times = np.arange(samples, dtype=float)
    latent = _smooth(rng, samples, 2)
    data = {}
    for subject in "abcde":
        # Modality "a" carries the first factor only; "b" carries both.
        wa = rng.normal(size=(8, 2)) * np.array([1.0, 0.0 if specific else 1.0])
        wb = rng.normal(size=(6, 2))
        a = latent @ wa.T + rng.normal(size=(samples, 8))
        b = latent @ wb.T + 0.5 * rng.normal(size=(samples, 6))
        observed = rng.uniform(size=a.shape) > 0.1 if mask else np.ones(a.shape, bool)
        data[subject] = {"run": {"a": TimeSeries(a, times, observed), "b": TimeSeries(b, times)}}
    model = MultimodalSRM(
        features=2,
        latent_dt=1.0,
        latent_pooling=pooling,
        responses={
            "a": Response(Identity(), estimate=False),
            "b": Response(Gaussian(1.0, 0.0), estimate=False),
        },
        max_iter=60,
        random_state=0,
    )
    return model.fit(data)


def _fitted_variance(model, modality):
    """Mean over participants of the variance of the complete fit, per standardized feature."""
    values = []
    for subject in model.subjects_:
        squares = count = 0.0
        for block in model.observation_blocks_:
            if (block.subject, block.modality) != (subject, modality):
                continue
            fit = (block.H @ model._training_latent_arrays_[subject][block.run]) @ model.loadings_[
                subject
            ][modality].T
            observed = block.mask & block.valid[:, None]
            squares += np.sum(fit[observed] ** 2)
            count += observed.sum()
        values.append(squares / count)
    return float(np.mean(values))


@pytest.mark.parametrize("mask", [False, True])
def test_total_explained_variance_is_the_variance_of_the_fitted_signal(mask):
    model = _fit(mask=mask)
    report = model.factor_report()
    assert report["modalities"] == ["a", "b"]
    for modality in "ab":
        np.testing.assert_allclose(
            report["total_explained_variance"][modality], _fitted_variance(model, modality)
        )
        assert len(report["explained_variance"][modality]) == 2
    shares = np.array([report["modality_share"][m] for m in "ab"])
    np.testing.assert_allclose(shares.sum(0), 1.0)


def test_specificity_finds_a_factor_missing_from_one_modality():
    report = _fit(specific=True).factor_report()
    assert report["specificity"]["a"]["least"] < 0.05
    assert report["specificity"]["b"]["most"] > 0.95
    balanced = _fit(specific=False).factor_report()
    assert balanced["specificity"]["a"]["least"] > 0.1


def test_specificity_does_not_depend_on_the_factor_basis():
    model = _fit()
    before = model.factor_report()
    mixing = np.array([[1.3, 0.4], [-0.7, 0.9]])
    inverse = np.linalg.inv(mixing)
    for subject in model.subjects_:
        for run, latent in model._training_latent_arrays_[subject].items():
            model._training_latent_arrays_[subject][run] = latent @ mixing
        for modality, loading in model.loadings_[subject].items():
            model.loadings_[subject][modality] = loading @ inverse.T
    after = model.factor_report()
    for modality in "ab":
        np.testing.assert_allclose(
            after["specificity"][modality]["shares"],
            before["specificity"][modality]["shares"],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            after["total_explained_variance"][modality],
            before["total_explained_variance"][modality],
        )
    # The per-factor table is in fitted coordinates and does change.
    assert not np.allclose(after["explained_variance"]["a"], before["explained_variance"]["a"])


def test_specificity_needs_one_shared_latent():
    report = _fit(pooling="population").factor_report()
    assert report["specificity"] is None
    assert set(report["participants"]) == set("abcde")


def test_factor_report_requires_a_fit():
    from sklearn.exceptions import NotFittedError

    from multimodalsrm import MultimodalSRM

    with pytest.raises(NotFittedError):
        MultimodalSRM(features=1, latent_dt=1.0).factor_report()
