"""Preparation contracts shared by Bayesian and legacy reference backends."""

import copy
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian.observation_adapter import (
    BayesianObservationAdapter,
)


def native_data():
    t = np.arange(0.0, 21.0, 2.0)
    mask = np.ones((len(t), 2), dtype=bool)
    mask[3, 1] = False
    mask[-1] = False
    values = np.column_stack((t, np.ones(len(t)) * 4))
    return {
        "a": {"r": {"ref": TimeSeries(values, t, mask=mask)}},
        "b": {"r": {"signal": TimeSeries((t + 2)[:, None], t)}},
    }


def assert_preparation_equal(left, right):
    assert left.domains_ == right.domains_
    assert left.subjects_ == right.subjects_
    assert left.modalities_ == right.modalities_
    assert left.configuration_ == right.configuration_
    for modality in left.responses_:
        assert left.responses_[modality].metadata == right.responses_[modality].metadata
    for run in left.grids_:
        assert_array_equal(left.grids_[run], right.grids_[run])
    for subject, mods in left.preprocessing_.items():
        for modality, stats in mods.items():
            for name, value in stats.items():
                assert_array_equal(value, right.preprocessing_[subject][modality][name])
    ls, lb = left._systems(left._training_data, left.domains_)
    rs, rb = right._systems(right._training_data, right.domains_)
    for run in ls:
        assert ls[run].keys == rs[run].keys
        assert_array_equal(ls[run].times, rs[run].times)
        assert_array_equal(ls[run].values, rs[run].values)
    for a, b in zip(lb, rb, strict=True):
        assert (a.subject, a.run, a.modality) == (b.subject, b.run, b.modality)
        for name in ("times", "values", "mask", "valid"):
            assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("standardize", [False, True])
def test_native_preparation_matches_reference_and_fixed_support(standardize):
    responses = {"signal": Response(Gaussian(0.25, 0.0), pooling="shared", estimate=False)}
    options = dict(features=2, latent_dt=0.7, responses=responses, standardize=standardize)
    adapter = BayesianObservationAdapter(**options)
    adapter._prepare(native_data())
    assert adapter.domains_ == {"r": (0.0, 20.0)}
    systems, blocks = adapter._systems(adapter._training_data, adapter.domains_)
    assert_array_equal(blocks[1].valid, [False] + [True] * 9 + [False])
    assert len(systems["r"].keys) == 28
    stats = adapter.preprocessing_["b"]["signal"]
    assert_allclose(stats["mean"], [12.0] if standardize else [0.0])
    assert_allclose(stats["scale"], [np.sqrt(80 / 3)] if standardize else [1.0])
    assert_array_equal(adapter.preprocessing_["a"]["ref"]["constant_features"], [False, True])
    # Conditioning reuses training statistics even with a new run and changed values.
    new = copy.deepcopy(adapter._training_data)
    new = {
        s: {
            "new": {
                m: TimeSeries(ts.values + 7.0, ts.times, mask=ts.mask)
                for m, ts in runs["r"].items()
            }
        }
        for s, runs in new.items()
    }
    _, domains = adapter._grids(new)
    actual, _ = adapter._systems(new, domains)
    shifts = np.array(
        [7.0 / adapter.preprocessing_[s][m]["scale"][f] for s, m, f in systems["r"].keys]
    )
    assert_allclose(actual["new"].values, systems["r"].values + shifts)


@pytest.mark.parametrize(
    "cls",
    [
        BayesianObservationAdapter,
    ],
)
def test_shared_preprocessing_masks_and_validation(cls):
    model = cls(features=2, latent_dt=2.0)
    model._prepare(native_data())
    assert_allclose(model.preprocessing_["a"]["ref"]["mean"], [9.0, 4.0])
    with pytest.raises(ValueError, match="pooling='shared'"):
        cls(latent_dt=1.0, responses={"signal": Response(Identity(), pooling="none")})._prepare(
            native_data()
        )
    with pytest.raises(ValueError, match="features must be a positive integer"):
        cls(features=True, latent_dt=1.0)._prepare(native_data())


def test_bayesian_import_and_preparation_do_not_load_legacy_estimators():
    code = """
import sys
from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter
from multimodalsrm.bayesian import diagnostics, persistence
adapter = BayesianObservationAdapter(latent_dt=1.0)
adapter._prepare({"s": {"r": {"m": TimeSeries([[1.0], [2.0]], [0.0, 1.0])}}})
assert "multimodalsrm.probabilistic" not in sys.modules
assert "multimodalsrm.continuous.model" not in sys.modules
assert all("continuous" not in c.__module__ and "probabilistic" not in c.__module__ for c in type(adapter).__mro__)
import multimodalsrm
assert "ProbabilisticMultimodalSRM" not in dir(multimodalsrm)
assert "ContinuousTimeMultimodalSRM" not in dir(multimodalsrm)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
