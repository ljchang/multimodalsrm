"""Exact graph-sharing regressions using real native-time numerical solves."""

import copy
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from multimodalsrm import MultimodalSRM, TimeSeries
from multimodalsrm.objective import complete_objective, solve_latents
from multimodalsrm.optimization import balance_global_scale, training_gradients


def graph():
    return {"a": {"b": 0.1}, "b": {"a": 0.1}, "c": {"d": 9}, "d": {"c": 9}}


def data_fixture():
    rng = np.random.RandomState(303)
    data = {}
    for i, s in enumerate("abcd"):
        maps = {"x": rng.normal(size=(3, 2)), "y": rng.normal(size=(4, 2))}
        data[s] = {}
        for j, r in enumerate(["r1", "r2"]):
            data[s][r] = {}
            for m, dt in [("x", 0.5), ("y", 1.0)]:
                t = np.arange(0, 12.01, dt)
                z = np.column_stack(
                    [np.sin(t / (1.5 + i // 2) + j), np.cos(t / (2.2 + i // 2) - j)]
                )
                mask = np.ones((len(t), len(maps[m])), bool)
                mask[4, 0] = False
                data[s][r][m] = TimeSeries(z @ maps[m].T, t, mask=mask)
    return data


def fit_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return MultimodalSRM(
            features=2,
            latent_dt=0.5,
            latent_pooling="components",
            latent_strength=0,
            max_iter=15,
            random_state=7,
        ).fit(data_fixture(), affinity=graph())


def args(model, z=None, pooling="components", affinity=None):
    return (
        model.observation_blocks_,
        model.run_grids_,
        z or model._training_latent_arrays_,
        model.loadings_,
        pooling,
        0,
        0.03,
        0.07,
        0.02,
        graph() if affinity is None else affinity,
    )


def test_fit_exact_pairs_native_maps_and_reduced_diagnostics():
    # Catches implementing components as a soft penalty or one global trajectory.
    model = fit_model()
    assert model.latent_groups_ == [["a", "b"], ["c", "d"]]
    assert set(model.component_latents_) == {"a", "c"}
    assert not hasattr(model, "population_latents_")
    for r in ["r1", "r2"]:
        for s, t in [("a", "b"), ("c", "d")]:
            np.testing.assert_array_equal(
                model.training_latents_[s][r].values, model.training_latents_[t][r].values
            )
        assert not np.allclose(
            model.training_latents_["a"][r].values, model.training_latents_["c"][r].values
        )
    for s in "abcd":
        assert model.loadings_[s]["x"].shape == (3, 2)
        assert model.loadings_[s]["y"].shape == (4, 2)
    assert not np.allclose(model.loadings_["a"]["x"], model.loadings_["b"]["x"])
    for b in model.observation_blocks_:
        ts = data_fixture()[b.subject][b.run][b.modality]
        np.testing.assert_array_equal(b.times, ts.times)
        np.testing.assert_array_equal(b.mask, ts.mask)
    assert model.optimization_diagnostics_["stationarity"]["latent_variable_count"] == 200


def test_component_reductions_match_shared_and_uncoupled_individual():
    # Catches wrong subject weights, observation offsets, or edge-strength effects.
    model = fit_model()
    for A, other in [
        ({s: {t: 1 for t in "abcd" if t != s} for s in "abcd"}, "shared"),
        ({s: {} for s in "abcd"}, "population"),
    ]:
        solve_args = (model.observation_blocks_, model.run_grids_, list("abcd"), model.loadings_, 2)
        z, _ = solve_latents(*solve_args, "components", 300, 0.03, 0.07, A)
        expected, _ = solve_latents(*solve_args, other, 0, 0.03, 0.07)
        for s in "abcd":
            for r in ["r1", "r2"]:
                np.testing.assert_allclose(z[s][r], expected[s][r], atol=1e-11)
        assert complete_objective(*args(model, z, affinity=A)) == pytest.approx(
            complete_objective(*args(model, expected, pooling=other, affinity=A)), abs=1e-11
        )


def test_reduced_gradient_and_scale_balance():
    # Catches counting repeated coordinates or dropping members' likelihood gradients.
    model = fit_model()
    z = copy.deepcopy(model._training_latent_arrays_)
    _, gz = training_gradients(*args(model, z))
    assert set(gz) == {"a", "c"}
    for group in [("a", "b"), ("c", "d")]:
        for r in ["r1", "r2"]:
            plus, minus = copy.deepcopy(z), copy.deepcopy(z)
            for s in group:
                plus[s][r][8, 1] += 1e-6
                minus[s][r][8, 1] -= 1e-6
            fd = (
                complete_objective(*args(model, plus)) - complete_objective(*args(model, minus))
            ) / 2e-6
            assert gz[group[0]][r][8, 1] == pytest.approx(fd, abs=2e-8)
    balanced, _, diag = balance_global_scale(*args(model, z))
    assert diag["accepted"]
    for s, t in [("a", "b"), ("c", "d")]:
        for r in ["r1", "r2"]:
            np.testing.assert_array_equal(balanced[s][r], balanced[t][r])


def test_disconnected_donors_rejected_zero_strength_same_group_and_poison_exclusion():
    # Catches allowing unrelated donors or reading excluded target observations.
    model = fit_model()
    data = data_fixture()
    with pytest.raises(ValueError, match="disconnected|support"):
        model.predict({"c": data["c"]}, targets={"a": ["y"]}, source="across")
    selected = {"a": data["a"], "b": data["b"]}
    pred = model.predict(selected, targets={"a": ["y"]}, source="across")
    selected["a"] = {"r1": {"y": object()}, "r2": {"y": object()}}
    poisoned = model.predict(selected, targets={"a": ["y"]}, source="across")
    for r in ["r1", "r2"]:
        np.testing.assert_array_equal(pred["a"][r]["y"].values, poisoned["a"][r]["y"].values)
        assert pred["a"][r]["y"].valid.any()


def test_calibration_inherits_only_connected_frozen_reference_and_refreshes_groups():
    # Catches cross-group mean initialization and stale fitted grouping metadata.
    model = fit_model()
    new = {"e": data_fixture()["a"]}
    result = model.calibrate(new, reference="training", affinity={"e": {"b": 2}})
    assert result.latent_groups_ == [["a", "b", "e"], ["c", "d"]]
    assert result.configuration_["latent_groups"] == result.latent_groups_
    for r in ["r1", "r2"]:
        np.testing.assert_array_equal(
            result.calibration_reference_latents_["e"][r], model._training_latent_arrays_["a"][r]
        )
    for s in "abcd":
        for m in ["x", "y"]:
            np.testing.assert_array_equal(result.loadings_[s][m], model.loadings_[s][m])
    assert model.latent_groups_ == [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError, match="merg|component"):
        model.calibrate(new, reference="training", affinity={"e": {"a": 1, "c": 1}})
    with pytest.raises(ValueError, match="component|support"):
        model.calibrate(new, reference="training", affinity={"e": {}})


def test_fixed_group_donors_must_agree():
    # Catches silently accepting conflicting fixed constraints in one free coordinate.
    model = fit_model()
    fixed = copy.deepcopy(model._training_latent_arrays_)
    fixed["b"]["r1"][0, 0] += 1
    with pytest.raises(ValueError, match="agree|inconsistent"):
        solve_latents(
            [],
            model.run_grids_,
            list("abcd"),
            {},
            2,
            "components",
            0,
            0.03,
            0.07,
            graph(),
            fixed_latents=fixed,
        )


def test_components_require_named_valid_affinity_and_transitive_input_order():
    for A in [None, {"a": {}}, {"a": {"b": 1}, "b": {}, "c": {}, "d": {}}]:
        with pytest.raises(ValueError, match="affinity|symmetric"):
            MultimodalSRM(features=2, latent_dt=0.5, latent_pooling="components").fit(
                data_fixture(), affinity=A
            )
    from multimodalsrm.grouping import latent_groups

    A = {"c": {"a": 1}, "a": {"c": 1, "b": 2}, "d": {}, "b": {"a": 2}}
    assert latent_groups(["c", "d", "b", "a"], "components", A) == [["c", "b", "a"], ["d"]]


def test_unequal_component_penalty_is_subject_weighted():
    # Catches replacing subject averages by equal component averages.
    from multimodalsrm.objective import _penalty

    A = {"a": {"b": 1}, "b": {"a": 1}, "c": {}}
    P = _penalty({"r": np.array([0.0, 1.0, 2.0])}, list("abc"), "components", 99, 2.0, 0.0, A)
    # Trapezoid weights [1/4, 1/2, 1/4], ridge 2, group masses 2/3 and 1/3.
    np.testing.assert_allclose(
        P["r"].toarray(), np.diag([1 / 3, 2 / 3, 1 / 3, 1 / 6, 1 / 3, 1 / 6])
    )


def span_fixture(nonidentity=False):
    from multimodalsrm import Gaussian, Identity, Response

    data = {}
    for s in "abcd":
        t = np.arange(0.0, 13.0) if s in "ab" else np.arange(0.5, 21.0)
        mask = np.ones((len(t), 1), bool)
        if s in "ab":
            mask[:2] = False
            mask[-2:] = False
        data[s] = {"r": {m: TimeSeries(np.sin(t[:, None] / 3), t, mask=mask) for m in ["x", "y"]}}
        unused_t = np.array([-10.0, 30.0])
        data[s]["r"]["unused"] = TimeSeries(unused_t[:, None], unused_t)
    response = Gaussian(width=1 / 6) if nonidentity else Identity()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MultimodalSRM(
            features=1,
            latent_dt=0.5,
            latent_pooling="components",
            latent_strength=0,
            max_iter=8,
            random_state=4,
            responses={m: Response(response, estimate=False) for m in ["x", "y", "unused"]},
            modality_weights={"x": 0.5, "y": 0.5, "unused": 0},
        ).fit(data, affinity=graph())
    return model, data


def test_component_training_support_uses_raw_masked_group_bounds_and_sources():
    # Catches global spans/sources leaking into representative training results.
    model, _ = span_fixture()
    result = model.component_latents_["a"]["r"]
    assert result.metadata["domain"] == (2.0, 10.0)
    np.testing.assert_array_equal(result.valid, (result.times >= 2) & (result.times <= 10))
    assert {s for s, _, _ in result.metadata["used_sources"]} == {"a", "b"}
    assert {row["subject"] for row in result.metadata["observation_coverage"]} == {"a", "b"}
    assert not result.metadata["observed_at_query"][result.times == 2.5].any()


def test_component_prediction_and_inference_exclude_disconnected_native_coverage():
    # Catches unrelated half-integer samples or long spans marking target output valid/observed.
    model, data = span_fixture()
    query = np.array([1.0, 2.0, 2.5, 10.0, 11.0, 15.0])
    selected = {s: data[s] for s in ["b", "c"]}
    pred = model.predict(selected, targets={"a": ["y"]}, source="across", times=query)["a"]["r"][
        "y"
    ]
    latent = model.infer_latent(selected, times={"a": {"r": query}})["a"]["r"]
    for result in [pred, latent]:
        assert result.metadata["domain"] == (2.0, 10.0)
        np.testing.assert_array_equal(result.valid, [False, True, True, True, False, False])
        np.testing.assert_array_equal(
            result.metadata["observed_at_query"], [False, True, False, True, False, False]
        )
        assert {s for s, _, _ in result.metadata["used_sources"]} == {"b"}
        assert {row["subject"] for row in result.metadata["observation_coverage"]} == {"b"}


@pytest.mark.parametrize("donors", [("b",), ("b", "c")])
def test_component_nonidentity_bounds_apply_response_envelope_only_once(donors):
    # Catches treating already support-trimmed block rows as the raw data domain.
    model, data = span_fixture(nonidentity=True)
    query = np.array([2.0, 3.0, 4.0, 8.0, 9.0, 10.0])
    selected = {s: data[s] for s in donors}
    pred = model.predict(selected, targets={"a": ["y"]}, source="across", times=query)["a"]["r"][
        "y"
    ]
    assert pred.metadata["domain"] == (2.0, 10.0)
    np.testing.assert_array_equal(pred.valid, [False, True, True, True, True, False])
    latent = model.infer_latent(selected, times={"a": {"r": query}})["a"]["r"]
    assert latent.valid.all()
    assert latent.metadata["domain"] == (2.0, 10.0)


@pytest.mark.parametrize("reference", [None, "training"])
@pytest.mark.parametrize("subject", ["e", "a"])
def test_component_calibration_rejects_unrelated_long_reference(reference, subject):
    # A long disconnected component cannot supply temporal overlap for either
    # a newcomer or a missing mapping on an established participant.
    model, data = span_fixture()
    modality = "x" if subject == "e" else "unused"
    weights = {"x": 0.25, "y": 0.25, "unused": 0.5}
    edges = {"e": {"a": 1}} if subject == "e" else None
    t = np.array([12.0, 14.0, 16.0])
    new = {subject: {"r": {modality: TimeSeries(t[:, None], t)}}}
    if reference is None:
        new.update({s: {"r": {m: data[s]["r"][m] for m in ["x", "y"]}} for s in ["b", "c"]})
    with pytest.raises(ValueError, match="usable|support|overlap"):
        model.calibrate(new, reference=reference, affinity=edges, modality_weights=weights)


@pytest.mark.parametrize("reference", [None, "training"])
@pytest.mark.parametrize("subject", ["e", "a"])
@pytest.mark.parametrize("nonidentity", [False, True])
def test_component_calibration_masks_before_preprocessing_and_trims_once(
    reference, subject, nonidentity
):
    # Unsupported finite values must affect neither standardization nor the
    # conditional loading solve; Gaussian raw bounds [2,10] allow [3,9].
    model, data = span_fixture(nonidentity=nonidentity)
    modality = "x" if subject == "e" else "unused"
    weights = {"x": 0.25, "y": 0.25, "unused": 0.5}
    edges = {"e": {"a": 1}} if subject == "e" else None
    before = copy.deepcopy(model)
    t = np.arange(1.0, 16.0)
    usable = (t >= (3 if nonidentity else 2)) & (t <= (9 if nonidentity else 10))
    x = (t**2)[:, None]
    poisoned = x.copy()
    poisoned[~usable] = 1e6

    def run(values, mask=None):
        new = {subject: {"r": {modality: TimeSeries(values, t, mask)}}}
        if reference is None:
            new.update({s: {"r": {m: data[s]["r"][m] for m in ["x", "y"]}} for s in ["b", "c"]})
        return model.calibrate(new, reference=reference, affinity=edges, modality_weights=weights)

    clean, dirty, masked = run(x), run(poisoned), run(x, usable[:, None])
    expected = x[usable]
    for result in [clean, dirty, masked]:
        np.testing.assert_allclose(
            result.preprocessing_[subject][modality]["mean"], expected.mean(0)
        )
        np.testing.assert_allclose(
            result.preprocessing_[subject][modality]["scale"], expected.std(0)
        )
        np.testing.assert_allclose(
            result.loadings_[subject][modality], masked.loadings_[subject][modality], atol=1e-12
        )
        np.testing.assert_allclose(
            result.calibration_diagnostics_["objective_history"],
            masked.calibration_diagnostics_["objective_history"],
            atol=1e-12,
        )
        for s in "abcd":
            np.testing.assert_array_equal(
                result._training_latent_arrays_[s]["r"], before._training_latent_arrays_[s]["r"]
            )
            for m in ["x", "y"]:
                np.testing.assert_array_equal(result.loadings_[s][m], before.loadings_[s][m])
                assert (
                    result.subject_kernels_[s][m].parameters
                    == before.subject_kernels_[s][m].parameters
                )
                for stat in ["mean", "scale", "constant_features"]:
                    np.testing.assert_array_equal(
                        result.preprocessing_[s][m][stat], before.preprocessing_[s][m][stat]
                    )
        for m, kernel in before.group_kernels_.items():
            assert result.group_kernels_[m].parameters == kernel.parameters
    np.testing.assert_array_equal(x[:, 0], t**2)


@pytest.mark.parametrize("reference", [None, "training"])
def test_component_calibration_keeps_interior_reference_gaps(reference):
    # Interior missing observations do not turn the component domain into
    # disconnected exterior intervals: interpolation there remains admissible.
    model, data = span_fixture()
    for s in "ab":
        for m in ["x", "y"]:
            ts = data[s]["r"][m]
            mask = ts.mask.copy()
            mask[(ts.times >= 4) & (ts.times <= 8)] = False
            data[s]["r"][m] = TimeSeries(ts.values, ts.times, mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(data, affinity=graph())
    t = np.array([5.0, 6.0, 7.0])
    new = {"e": {"r": {"x": TimeSeries((t**2)[:, None], t)}}}
    if reference is None:
        new.update({s: data[s] for s in ["b", "c"]})
    result = model.calibrate(new, reference=reference, affinity={"e": {"a": 1}})
    np.testing.assert_allclose(result.preprocessing_["e"]["x"]["mean"], [110 / 3])
    assert np.isfinite(result.loadings_["e"]["x"]).all()


@pytest.mark.parametrize("reference", [None, "training"])
def test_component_calibration_renormalizes_remaining_modalities_across_runs(reference):
    # A wholly unsupported modality/run must have no effect when that mapping
    # has support in another run, including its local modality-weight mass.
    model, data = span_fixture()
    data = {s: {"r": runs["r"], "second": runs["r"]} for s, runs in data.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(data, affinity=graph())
    t = np.array([3.0, 5.0, 8.0])
    x = TimeSeries((t**2)[:, None], t)
    outside = TimeSeries(np.array([[1e6], [2e6], [3e6]]), np.array([12.0, 14.0, 16.0]))
    new = {"e": {"r": {"x": x, "y": outside}, "second": {"x": x, "y": x}}}
    clean = copy.deepcopy(new)
    del clean["e"]["r"]["y"]
    if reference is None:
        for inputs in [new, clean]:
            inputs.update({s: data[s] for s in ["b", "c"]})
    a = model.calibrate(new, reference=reference, affinity={"e": {"a": 1}})
    b = model.calibrate(clean, reference=reference, affinity={"e": {"a": 1}})
    for m in ["x", "y"]:
        np.testing.assert_array_equal(
            a.preprocessing_["e"][m]["mean"], b.preprocessing_["e"][m]["mean"]
        )
        np.testing.assert_array_equal(
            a.preprocessing_["e"][m]["scale"], b.preprocessing_["e"][m]["scale"]
        )
        np.testing.assert_allclose(a.loadings_["e"][m], b.loadings_["e"][m], atol=1e-12)
    np.testing.assert_allclose(
        a.calibration_diagnostics_["objective_history"],
        b.calibration_diagnostics_["objective_history"],
        atol=1e-12,
    )
