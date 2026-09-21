import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries


def fixture():
    t = np.arange(12.0)
    return {
        s: {
            "a": TimeSeries(np.sin(t[:, None] / 3 + j), t),
            "b": TimeSeries(np.cos(t[::2, None] / 3 + j), t[::2]),
        }
        for j, s in enumerate(["s1", "s2"])
    }


def test_rank_shared_reproducible_and_monotone():
    kwargs = dict(features=3, latent_dt=1, latent_pooling="shared", max_iter=15, random_state=4)
    with pytest.warns(ConvergenceWarning):
        a = MultimodalSRM(**kwargs).fit(fixture())
    with pytest.warns(ConvergenceWarning):
        b = MultimodalSRM(**kwargs).fit(fixture())
    np.testing.assert_array_equal(
        a.training_latents_["s1"]["run-01"].values,
        a.training_latents_["s2"]["run-01"].values,
    )
    np.testing.assert_array_equal(a.loadings_["s1"]["a"], b.loadings_["s1"]["a"])
    assert np.max(np.diff(a.objective_history_)) <= 1e-10
    assert a.loadings_["s1"]["a"].shape == (1, 3)
    assert not a.converged_


def test_weights_and_observed_preprocessing():
    with pytest.raises(ValueError, match="sum"):
        MultimodalSRM(latent_dt=1, modality_weights={"a": 1, "b": 1}).fit(fixture())
    with pytest.warns(ConvergenceWarning):
        model = MultimodalSRM(
            features=2,
            latent_dt=1,
            max_iter=1,
            random_state=0,
            tol=1e-12,
            modality_weights={"a": 1, "b": 0},
        ).fit(fixture())
    assert set(model.loadings_["s1"]) == {"a"}
    np.testing.assert_allclose(
        model.preprocessing_["s1"]["a"]["mean"], fixture()["s1"]["a"].values.mean(0)
    )
    assert all(np.isclose(b.coefficients.sum(), 0.5) for b in model.observation_blocks_)


def test_graph_validation_and_kernel_pooling():
    with pytest.raises(ValueError, match="symmetric"):
        MultimodalSRM(latent_dt=1, latent_pooling="neighborhood").fit(
            fixture(), affinity={"s1": {"s2": 1}, "s2": {}}
        )
    for pooling in ["shared", "partial", "none"]:
        with pytest.warns(ConvergenceWarning):
            m = MultimodalSRM(
                features=2,
                latent_dt=1,
                max_iter=1,
                random_state=0,
                tol=1e-12,
                responses={x: Response(Identity(), pooling=pooling) for x in ["a", "b"]},
            ).fit(fixture())
        assert ("a" in m.group_kernels_) == (pooling != "none")


def test_mask_runs_constant_statistics_and_actual_domain():
    data = {
        "s": {
            "one": {
                "a": TimeSeries(
                    [[1.0, 2.0], [3.0, np.nan], [5.0, 2.0]],
                    [0.0, 1.0, 2.3],
                    mask=[[True, True], [True, False], [True, True]],
                )
            },
            "two": {"a": TimeSeries([[7.0, 2.0], [9.0, 2.0]], [0.0, 2.3])},
        }
    }
    with pytest.warns(ConvergenceWarning):
        m = MultimodalSRM(features=3, latent_dt=1.0, max_iter=1, random_state=0).fit(data)
    np.testing.assert_allclose(m.preprocessing_["s"]["a"]["mean"], [5, 2])
    assert m.preprocessing_["s"]["a"]["scale"][1] == 1
    assert set(m.training_latents_["s"]) == {"one", "two"}
    assert not m.training_latents_["s"]["one"].valid[-1]
    assert all(np.isclose(b.coefficients.sum(), 0.5) for b in m.observation_blocks_)


def test_native_sparse_solver_matches_objective_stationarity():
    from multimodalsrm.objective import complete_objective, solve_latents

    with pytest.warns(ConvergenceWarning):
        m = MultimodalSRM(features=2, latent_dt=1.0, max_iter=1, random_state=0).fit(fixture())
    z, diagnostics = solve_latents(
        m.observation_blocks_,
        m.run_grids_,
        m.subjects_,
        m.loadings_,
        2,
        "population",
        1.0,
        1e-4,
        1e-2,
    )

    def obj(z):
        return complete_objective(
            m.observation_blocks_,
            m.run_grids_,
            z,
            m.loadings_,
            "population",
            1.0,
            1e-4,
            1e-2,
            1e-3,
        )

    baseline = obj(z)
    for s in z:
        for direction in [-1, 1]:
            zz = {ss: {r: v.copy() for r, v in runs.items()} for ss, runs in z.items()}
            zz[s]["run-01"][4, 0] += direction * 1e-4
            assert obj(zz) >= baseline - 1e-12
    assert all(d["success"] and d["relative_residual"] < 1e-7 for d in diagnostics)


def test_nonlinear_pooled_kernels_fixed_parameters_and_priors():
    from multimodalsrm import Gaussian, KernelPrior, Normal
    from multimodalsrm.kernel_optimization import kernel_penalty

    t = np.arange(0, 10, 0.5)
    data = {
        s: {"a": TimeSeries(np.sin(t[:, None] + j * 0.2), t)} for j, s in enumerate(["s1", "s2"])
    }
    for pooling in ["shared", "partial", "none"]:
        response = Response(
            Gaussian(width=0.15),
            pooling=pooling,
            fixed={"width": 0.15},
            bounds={"lag": (-0.2, 0.2)},
            lag_prior=Normal(0.1, 0.5),
            prior=KernelPrior({"width": 0.2}, 0.1),
        )
        with pytest.warns(ConvergenceWarning):
            model = MultimodalSRM(
                features=1,
                latent_dt=0.5,
                max_iter=2,
                kernel_max_iter=4,
                random_state=1,
                responses={"a": response},
            ).fit(data)
        assert np.max(np.diff(model.objective_history_)) <= 1e-10
        assert all(k["a"].width == 0.15 for k in model.subject_kernels_.values())
        assert np.isfinite(kernel_penalty(model.subject_kernels_, model.responses_))
        if pooling == "shared":
            assert model.subject_kernels_["s1"]["a"].lag == model.subject_kernels_["s2"]["a"].lag
        if pooling == "partial":
            np.testing.assert_allclose(
                model.group_kernels_["a"].lag,
                np.mean([k["a"].lag for k in model.subject_kernels_.values()]),
            )


def test_zero_weight_cannot_extend_domain_and_unusable_pair_excluded():
    from multimodalsrm import Gaussian

    data = {
        "s": {
            "a": TimeSeries([[1.0], [2.0], [3.0]], [0, 1, 2]),
            "b": TimeSeries([[1.0], [2.0]], [0, 100]),
        }
    }
    with pytest.warns(ConvergenceWarning):
        m = MultimodalSRM(
            features=2,
            latent_dt=1,
            max_iter=1,
            random_state=0,
            tol=1e-12,
            modality_weights={"a": 1, "b": 0},
        ).fit(data)
    assert m.run_domains_["run-01"] == (0, 2)
    data["s"]["b"] = TimeSeries([[1.0], [2.0]], [0, 2])
    with pytest.warns(ConvergenceWarning):
        m = MultimodalSRM(
            features=2,
            latent_dt=1,
            max_iter=1,
            random_state=0,
            tol=1e-12,
            responses={
                "a": Response(Identity()),
                "b": Response(Gaussian(), estimate=False),
            },
        ).fit(data)
    assert "b" not in m.loadings_["s"]
    assert "b" not in m.subject_kernels_["s"]


def test_operator_cache_equivalence():
    from multimodalsrm.data import normalize_data
    from multimodalsrm.objective import make_blocks

    with pytest.warns(ConvergenceWarning):
        model = MultimodalSRM(features=1, latent_dt=1, max_iter=1, random_state=0, tol=1e-12).fit(
            fixture()
        )
    args = (
        normalize_data(fixture()),
        model.run_grids_,
        model.run_domains_,
        model.preprocessing_,
        model.subject_kernels_,
        model.responses_,
        model.modality_weights_,
    )
    baseline = make_blocks(*args)
    cache = {}
    for _ in range(2):
        cached = make_blocks(*args, operator_cache=cache)
        for a, b in zip(baseline, cached):
            np.testing.assert_array_equal(a.H.toarray(), b.H.toarray())
            np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert len(cache) == len(baseline)


def test_refit_clears_population_trajectory():
    with pytest.warns(ConvergenceWarning):
        model = MultimodalSRM(features=1, latent_dt=1, max_iter=1, random_state=0, tol=1e-12).fit(
            fixture()
        )
    assert hasattr(model, "population_latents_")
    with pytest.warns(ConvergenceWarning):
        model.set_params(latent_pooling="neighborhood").fit(
            fixture(), affinity={"s1": {"s2": 1}, "s2": {"s1": 1}}
        )
    assert not hasattr(model, "population_latents_")
