import copy

import numpy as np
import pytest

from multimodalsrm import MultimodalSRM, TimeSeries


def fixture(pooling="population", strength=1.0):
    t = np.arange(10.0)
    data = {
        s: {"r": {m: TimeSeries((np.sin(t / 2) + i)[:, None], t) for i, m in enumerate(["x", "y"])}}
        for s in ["a", "b"]
    }
    model = MultimodalSRM(
        features=1,
        latent_dt=1,
        latent_pooling=pooling,
        latent_strength=strength,
        max_iter=50,
        tol=1e-3,
        random_state=1,
    )
    model.fit(
        data,
        affinity={"a": {"b": 1}, "b": {"a": 1}} if pooling == "neighborhood" else None,
    )
    return model, data, t


@pytest.mark.parametrize("source", ["within", "across", "both"])
def test_target_union_removed_before_validation_and_domains(source):
    model, data, t = fixture()
    targets = {"a": ["x"], "b": ["y"]}
    expected = model.predict(data, targets=targets, source=source, times=t)
    changed = copy.deepcopy(data)
    changed["a"]["r"]["x"] = TimeSeries(np.ones((2, 7)), [-200, 200])
    changed["a"]["extra"] = {"x": TimeSeries(np.ones((3, 8)), [0, 1, 2])}
    changed["b"]["r"].pop("y")
    actual = model.predict(changed, targets=targets, source=source, times=t)
    for s, m in [("a", "x"), ("b", "y")]:
        np.testing.assert_allclose(actual[s]["r"][m].values, expected[s]["r"][m].values)
        assert actual[s]["r"][m].metadata["domain"] == expected[s]["r"][m].metadata["domain"]
        assert (
            actual[s]["r"][m].metadata["used_sources"]
            == expected[s]["r"][m].metadata["used_sources"]
        )
        assert ("a", "r", "x") not in actual[s]["r"][m].metadata["used_sources"]
        assert ("b", "r", "y") not in actual[s]["r"][m].metadata["used_sources"]


def test_query_extension_does_not_extend_domain_or_reuse_training():
    model, data, t = fixture()
    a = model.predict(data, targets={"a": ["x"]}, times=t)["a"]["r"]["x"]
    b = model.predict(data, targets={"a": ["x"]}, times=np.r_[t, 30.0])["a"]["r"]["x"]
    np.testing.assert_allclose(a.values, b.values[:-1])
    assert not b.valid[-1] and np.isnan(b.values[-1]).all()
    new = {s: {"new": runs["r"]} for s, runs in data.items()}
    old = model.infer_latent(data)["a"]["r"]
    fresh = model.infer_latent(new)["a"]["new"]
    np.testing.assert_allclose(old.values, fresh.values)


@pytest.mark.parametrize("pooling", ["population", "neighborhood"])
def test_across_needs_positive_coupling(pooling):
    model, data, t = fixture(pooling, 0.0)
    with pytest.raises(ValueError, match="coupl|support|connect"):
        model.predict(data, targets={"a": ["x"]}, source="across", times=t)


def test_missing_run_supported_and_mapping_rejected():
    model, data, t = fixture()
    donor = {"b": data["b"]}
    assert model.predict(donor, targets={"a": ["x"]}, source="across", times=t)["a"]["r"][
        "x"
    ].valid.all()
    with pytest.raises(ValueError, match="mapping"):
        model.predict(donor, targets={"a": ["z"]}, source="across", times=t)


def test_source_filter_precedes_feature_validation_and_queries_are_explicit():
    model, data, t = fixture()
    data["b"]["other"] = {"x": TimeSeries(np.ones((2, 9)), [0, 1])}
    assert model.predict(data, targets={"a": ["x"]}, source="within", times=t)["a"]["r"][
        "x"
    ].valid.all()
    with pytest.raises(ValueError, match="requested runs"):
        model.predict(data, targets={"a": ["x"]}, times={"unknown": t})
    output = model.predict(data, targets={"a": ["x"], "b": ["y"]}, times={"a": {"r": {"x": t}}})
    assert set(output) == {"a"}


def test_frozen_arrays_and_original_units():
    model, data, t = fixture()
    before = copy.deepcopy(model)
    for source in ["within", "across", "both"]:
        prediction = model.predict(data, targets={"a": ["x"]}, source=source, times=t)["a"]["r"][
            "x"
        ]
        assert prediction.values.shape == (len(t), 1)
    for s in model.subjects_:
        for m in model.loadings_[s]:
            np.testing.assert_array_equal(model.loadings_[s][m], before.loadings_[s][m])
            np.testing.assert_array_equal(
                model.preprocessing_[s][m]["mean"], before.preprocessing_[s][m]["mean"]
            )
        np.testing.assert_array_equal(
            model._training_latent_arrays_[s]["r"],
            before._training_latent_arrays_[s]["r"],
        )
    transformed = copy.deepcopy(model)
    transformed.preprocessing_["a"]["x"]["mean"] += 7
    transformed.preprocessing_["a"]["x"]["scale"] *= 3
    p = model.predict(data, targets={"a": ["x"]}, times=t)["a"]["r"]["x"].values
    q = transformed.predict(data, targets={"a": ["x"]}, times=t)["a"]["r"]["x"].values
    mean = model.preprocessing_["a"]["x"]["mean"]
    np.testing.assert_allclose(q, (p - mean) * 3 + mean + 7)


def test_actual_endpoint_and_internal_gap_coverage():
    model, data, t = fixture()
    source_t = np.array([0.0, 1.0, 2.0, 7.0, 8.0, 8.4])
    data = {"a": {"r": {"y": TimeSeries(np.sin(source_t)[:, None], source_t)}}}
    out = model.predict(data, targets={"a": ["x"]}, times=[0.0, 4.0, 8.4, 9.0])["a"]["r"]["x"]
    np.testing.assert_array_equal(out.valid, [True, True, True, False])
    assert out.metadata["observation_coverage"][0]["intervals"] == [
        (0.0, 2.0),
        (7.0, 8.4),
    ]
    assert not out.metadata["observed_at_query"][1]


def test_disconnected_graph_and_mixed_retained_nesting_rejected():
    model, data, t = fixture("neighborhood")
    model.affinity_ = {"a": {}, "b": {}}
    with pytest.raises(ValueError, match="disconnected"):
        model.predict(data, targets={"a": ["x"]}, source="across", times=t)
    model, data, t = fixture()
    with pytest.raises(ValueError, match="mixed nesting"):
        model.infer_latent({"a": data["a"]["r"], "b": data["b"]})


@pytest.mark.parametrize("lag", [0.0, 1.5])
def test_two_sided_and_delayed_support_stays_frozen(lag):
    from multimodalsrm import Gaussian, Response

    model, data, t = fixture()
    response = Response(Gaussian(width=0.1, lag=lag), estimate=False)
    model.responses_["x"] = response
    model.subject_kernels_["a"]["x"] = response.kernel
    query = np.arange(-1.0, 13.0, 0.5)
    first = model.predict(data, targets={"a": ["x"]}, times=query)["a"]["r"]["x"]
    second = model.predict(data, targets={"a": ["x"]}, times=np.r_[query, 100.0])["a"]["r"]["x"]
    lo, hi = response.support_envelope()
    np.testing.assert_array_equal(first.valid, (query - hi >= 0) & (query - lo <= 9))
    np.testing.assert_allclose(first.values, second.values[:-1], equal_nan=True)


def test_inference_rejects_unavailable_explicit_queries():
    model, data, t = fixture()
    with pytest.raises(ValueError, match="requested runs"):
        model.infer_latent(data, times={"missing": t})


def test_nested_inference_query_returns_donor_supported_absent_run():
    model, data, t = fixture()
    supplied = {"a": {"r1": data["a"]["r"]}, "b": {"r2": data["b"]["r"]}}
    output = model.infer_latent(supplied, times={"a": {"r2": t}})
    assert set(output) == {"a"} and set(output["a"]) == {"r2"}
    np.testing.assert_array_equal(output["a"]["r2"].times, t)
    assert output["a"]["r2"].valid.all()
    assert all(s == "b" and r == "r2" for s, r, m in output["a"]["r2"].metadata["used_sources"])
    model.latent_strength = 0
    with pytest.raises(ValueError, match="coupling"):
        model.infer_latent(supplied, times={"a": {"r2": t}})
