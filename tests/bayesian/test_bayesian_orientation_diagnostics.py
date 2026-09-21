"""Two-factor loading-orientation diagnostics keep symmetry failures visible."""

import json

import numpy as np
import pytest


def _loadings(theta, determinant):
    """Construct loadings whose positive-diagonal anchor QR has given O(2)."""
    theta = np.asarray(theta, dtype=float)
    determinant = np.broadcast_to(np.asarray(determinant, dtype=float), theta.shape)
    cosine, sine = np.cos(theta), np.sin(theta)
    q = np.empty(theta.shape + (2, 2))
    q[..., 0, 0] = cosine
    q[..., 1, 0] = sine
    q[..., 0, 1] = -determinant * sine
    q[..., 1, 1] = determinant * cosine
    canonical = np.broadcast_to(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.35, -0.6]]),
        theta.shape + (3, 2),
    )
    return canonical @ np.swapaxes(q, -1, -2)


def test_reflection_diagnostic_fails_chains_trapped_at_opposite_signs():
    from multimodalsrm.bayesian.diagnostics import diagnostic_summary
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    rng = np.random.default_rng(17)
    theta = rng.uniform(-np.pi, np.pi, size=(4, 1000))
    loadings = _loadings(theta, np.array([1.0, -1.0, 1.0, -1.0])[:, None])

    # Every raw loading marginal looks mixed even though no chain changes the
    # joint reflection coordinate. This is the failure the added gate catches.
    raw = diagnostic_summary(loadings.reshape(4, 1000, -1))
    assert float(raw.r_hat.max()) < 1.01
    result = orientation_diagnostics(loadings)

    determinant = result["per_quantity"][0]
    assert determinant["name"] == "det_Q"
    assert determinant["status"] == "within_chain_constants_disagree"
    assert determinant["r_hat"] is None
    assert determinant["passes"] is False
    assert result["status"] == "computed"
    assert result["passes"] is False
    assert [row["positive_determinant_fraction"] for row in result["chains"]] == [
        1.0,
        0.0,
        1.0,
        0.0,
    ]
    assert [row["reflection_transitions"] for row in result["chains"]] == [0] * 4


def test_constant_reflection_sign_fails_even_when_all_chains_agree():
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    theta = np.random.default_rng(19).uniform(-np.pi, np.pi, size=(4, 1000))
    result = orientation_diagnostics(_loadings(theta, 1.0))

    determinant = result["per_quantity"][0]
    assert determinant["status"] == "all_draws_constant"
    assert determinant["passes"] is False
    assert result["passes"] is False


def test_seeded_haar_control_passes_all_seventeen_quantities():
    from multimodalsrm.bayesian.orientation_diagnostics import (
        ORIENTATION_LIMITS,
        orientation_diagnostics,
    )

    rng = np.random.default_rng(23)
    theta = rng.uniform(-np.pi, np.pi, size=(4, 1000))
    determinant = rng.choice([-1.0, 1.0], size=theta.shape)
    result = orientation_diagnostics(_loadings(theta, determinant))

    expected_names = ["det_Q"]
    for harmonic in range(1, 5):
        expected_names.extend(
            [
                f"cos_{harmonic}_theta",
                f"det_Q_times_cos_{harmonic}_theta",
                f"sin_{harmonic}_theta",
                f"det_Q_times_sin_{harmonic}_theta",
            ]
        )
    assert [row["name"] for row in result["per_quantity"]] == expected_names
    assert len(result["per_quantity"]) == 17
    assert all(row["status"] == "computed" for row in result["per_quantity"])
    assert all(row["passes"] for row in result["per_quantity"])
    assert result["passes"] is True
    assert result["limits"] == ORIENTATION_LIMITS
    assert result["max_rank_rhat"] <= 1.01
    assert result["min_bulk_ess"] >= 400.0
    assert result["min_tail_ess"] >= 200.0
    json.dumps(result, allow_nan=False)


def test_singular_anchor_is_an_explicit_unfiltered_failure():
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    theta = np.zeros((2, 5))
    loadings = _loadings(theta, 1.0)
    loadings[1, 3, 1] = loadings[1, 3, 0] * 1e-11
    result = orientation_diagnostics(loadings)

    assert result["status"] == "singular_anchor"
    assert result["passes"] is False
    assert result["counts"] == {
        "chains": 2,
        "draws_per_chain": 5,
        "total_draws": 10,
        "retained_draws": 10,
        "valid_anchor_draws": 9,
        "singular_anchor_draws": 1,
    }
    assert all(row["status"] == "singular_anchor" for row in result["per_quantity"])
    assert all(row["r_hat"] is None for row in result["per_quantity"])
    assert result["chains"][1]["valid_anchor_draws"] == 4
    assert result["chains"][1]["positive_determinant_fraction"] is None
    json.dumps(result, allow_nan=False)


def test_trace_size_is_reported_without_calling_undefined_diagnostics():
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    result = orientation_diagnostics(_loadings(np.zeros((1, 3)), 1.0))

    assert result["status"] == "insufficient_chains_or_draws"
    assert result["passes"] is False
    assert all(row["status"] == "insufficient_chains_or_draws" for row in result["per_quantity"])
    assert result["max_rank_rhat"] is None
    assert result["min_bulk_ess"] is None
    assert result["min_tail_ess"] is None


@pytest.mark.parametrize(
    "loadings, anchor_indices, match",
    [
        ([[[[1.0, 0.0], [0.0, 1.0]]]], (0, 1), "NumPy ndarray"),
        (np.zeros((2, 4, 2)), (0, 1), "chain, draw, row, factor"),
        (np.zeros((2, 4, 2, 1)), (0, 1), "at least two factors"),
        (np.full((2, 4, 2, 2), np.nan), (0, 1), "finite"),
        (np.ones((2, 4, 2, 2), dtype=complex), (0, 1), "real numeric"),
        (np.full((2, 4, 2, 2), "x"), (0, 1), "real numeric"),
        (np.zeros((2, 4, 2, 2)), (0,), "exactly two"),
        (np.zeros((2, 4, 2, 2)), (0, 0), "distinct"),
        (np.zeros((2, 4, 2, 2)), (0, 2), "in range"),
        (np.zeros((2, 4, 2, 2)), (False, 1), "integers"),
    ],
)
def test_invalid_inputs_raise_meaningful_errors(loadings, anchor_indices, match):
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        orientation_diagnostics(loadings, anchor_indices=anchor_indices)


def test_anchor_selection_preserves_chain_order_and_does_not_mutate_input():
    from multimodalsrm.bayesian.orientation_diagnostics import (
        orientation_diagnostics,
    )

    rng = np.random.default_rng(29)
    theta = rng.uniform(-np.pi, np.pi, size=(4, 128))
    determinant = np.ones_like(theta)
    determinant[1, :32] = -1
    determinant[2, :64] = -1
    determinant[3] = -1
    core = _loadings(theta, determinant)
    loadings = np.insert(core, 0, values=7.0, axis=2)
    before = loadings.copy()

    result = orientation_diagnostics(loadings, anchor_indices=(1, 2))

    assert result["anchor_indices"] == [1, 2]
    assert result["counts"]["chains"] == 4
    assert [row["chain"] for row in result["chains"]] == [0, 1, 2, 3]
    assert [row["positive_determinant_fraction"] for row in result["chains"]] == [
        1.0,
        0.75,
        0.5,
        0.0,
    ]
    assert np.array_equal(loadings, before)


def _haar_loadings(k, chains=4, draws=1000):
    rng = np.random.default_rng(102 + k)
    q, r = np.linalg.qr(rng.normal(size=(chains, draws, k, k)))
    q *= np.where(np.diagonal(r, axis1=-2, axis2=-1) < 0, -1.0, 1.0)[..., None, :]
    return q.swapaxes(-1, -2)


@pytest.mark.parametrize("k", [3, 5])
def test_general_k_iid_haar_finite_probes_pass(k):
    from multimodalsrm.bayesian.orientation_diagnostics import orientation_diagnostics

    result = orientation_diagnostics(_haar_loadings(k))
    assert result["passes"] is True
    assert result["anchor_indices"] == list(range(k))
    assert result["probe_schema"] == "qr_entries_squares_v1"
    assert len(result["per_quantity"]) == 1 + 4 * k * k
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("failure", ["determinant", "fixed", "subspace", "singular", "short"])
def test_general_k_failures_are_retained(k, failure):
    from multimodalsrm.bayesian.orientation_diagnostics import orientation_diagnostics

    w = _haar_loadings(k)
    if failure == "determinant":
        w[..., 0, :] *= np.linalg.slogdet(w)[0][..., None]
    elif failure == "fixed":
        w[:] = np.eye(k)
    elif failure == "subspace":
        w[:] = np.eye(k)
        w[..., :2, :2] = _haar_loadings(2)
    elif failure == "singular":
        w[1, 3, -1] = w[1, 3, 0]
    else:
        w = w[:1, :3]
    result = orientation_diagnostics(w)
    assert result["passes"] is False
    assert result["counts"]["retained_draws"] == w.shape[0] * w.shape[1]
    if failure == "singular":
        assert result["status"] == "singular_anchor"
        assert result["counts"]["singular_anchor_draws"] == 1
    if failure == "short":
        assert result["status"] == "insufficient_chains_or_draws"


@pytest.mark.parametrize("anchors", [(0, 1), (0, 1, 1), (0, 1, 3)])
def test_general_k_requires_k_distinct_in_range_anchors(anchors):
    from multimodalsrm.bayesian.orientation_diagnostics import orientation_diagnostics

    with pytest.raises(ValueError):
        orientation_diagnostics(_haar_loadings(3), anchor_indices=anchors)
