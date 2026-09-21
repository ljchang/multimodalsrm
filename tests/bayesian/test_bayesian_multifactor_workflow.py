"""The saved synthetic workflow retains gates, support and factor columns."""

import numpy as np
from numpy.testing import assert_array_equal


def test_synthetic_has_independent_runs_native_rates_masks_and_shared_loadings():
    from ..reference.multifactor_fixture import synthetic

    train, test, truth = synthetic(seed=731, duration=16, aux_hz=2)
    again = synthetic(seed=731, duration=16, aux_hz=2)
    assert len(train) == 3
    assert "aux" not in train["s3"]["train"]
    assert truth["latent"]["train"].shape == (16, 2)
    assert not np.array_equal(truth["latent"]["train"], truth["latent"]["test"])
    assert train["s1"]["train"]["brain"].values.shape == (16, 3)
    assert not train["s2"]["train"]["brain"].mask.all()
    assert len(train["s1"]["train"]["aux"].times) == 32
    assert_array_equal(test["s1"]["test"]["brain"].values, again[1]["s1"]["test"]["brain"].values)
