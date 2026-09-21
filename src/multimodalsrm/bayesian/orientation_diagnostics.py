"""Finite mixing probes for the unidentified orientation of loading matrices."""

import numpy as np

from .diagnostics import diagnostic_summary

ORIENTATION_LIMITS = {
    "max_rank_rhat": 1.01,
    "min_bulk_ess": 400.0,
    "min_tail_ess": 200.0,
    "rank_tolerance": 1e-10,
    "minimum_chains": 2,
    "minimum_draws": 4,
}

_QUANTITY_NAMES = ["det_Q"]
for _harmonic in range(1, 5):
    _QUANTITY_NAMES.extend(
        [
            f"cos_{_harmonic}_theta",
            f"det_Q_times_cos_{_harmonic}_theta",
            f"sin_{_harmonic}_theta",
            f"det_Q_times_sin_{_harmonic}_theta",
        ]
    )


GENERAL_K_PROBE_SCHEMA = "qr_entries_squares_v1"


def orientation_quantity_names(k):
    """Stable coordinate order; retain the original O(2) harmonic schema."""
    if k == 2:
        return list(_QUANTITY_NAMES)
    names = ["det_Q"]
    for i in range(k):
        for j in range(k):
            for name in (f"Q_{i}_{j}", f"Q_{i}_{j}_squared"):
                names.extend([name, f"det_Q_times_{name}"])
    return names


def orientation_quantity_values(anchor):
    """Extract positive-diagonal QR probes shared by diagnostics and validation."""
    q, r = np.linalg.qr(np.swapaxes(anchor, -1, -2))
    signs = np.where(np.diagonal(r, axis1=-2, axis2=-1) < 0, -1.0, 1.0)
    q = q * signs[..., None, :]
    determinant = np.where(np.linalg.det(q) > 0, 1.0, -1.0)
    quantities = [determinant]
    k = anchor.shape[-1]
    if k == 2:
        theta = np.arctan2(q[..., 1, 0], q[..., 0, 0])
        for harmonic in range(1, 5):
            for periodic in (np.cos(harmonic * theta), np.sin(harmonic * theta)):
                quantities.extend([periodic, determinant * periodic])
    else:
        for i in range(k):
            for j in range(k):
                for value in (q[..., i, j], q[..., i, j] ** 2):
                    quantities.extend([value, determinant * value])
    return np.stack(quantities, axis=-1)


def _validate(loadings, anchor_indices):
    if not isinstance(loadings, np.ndarray):
        raise TypeError("loadings must be a NumPy ndarray")
    if loadings.ndim != 4:
        raise ValueError("loadings must have chain, draw, row, factor axes")
    k = loadings.shape[-1]
    if k < 2:
        raise ValueError("orientation diagnostics require at least two factors")
    if not np.issubdtype(loadings.dtype, np.number) or np.issubdtype(
        loadings.dtype, np.complexfloating
    ):
        raise TypeError("loadings must have a real numeric dtype")
    if not np.isfinite(loadings).all():
        raise ValueError("loadings must contain only finite values")
    if anchor_indices is None:
        anchor_indices = tuple(range(k))
    if not isinstance(anchor_indices, (tuple, list)) or len(anchor_indices) != k:
        count = "two" if k == 2 else str(k)
        raise ValueError(f"anchor_indices must contain exactly {count} row indices")
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in anchor_indices
    ):
        raise TypeError("anchor_indices must be integers")
    anchors = tuple(int(i) for i in anchor_indices)
    if len(set(anchors)) != k:
        raise ValueError("anchor_indices must be distinct")
    if any(i < 0 or i >= loadings.shape[2] for i in anchors):
        raise ValueError("anchor_indices must be in range for the loading rows")
    return np.asarray(loadings, dtype=np.float64), anchors


def _empty_quantities(status, k):
    return [
        {
            "name": name,
            "status": status,
            "r_hat": None,
            "ess_bulk": None,
            "ess_tail": None,
            "passes": False,
        }
        for name in orientation_quantity_names(k)
    ]


def _base_result(loadings, anchors, valid):
    chains, draws = loadings.shape[:2]
    total = chains * draws
    valid_count = int(valid.sum())
    return {
        **({"probe_schema": GENERAL_K_PROBE_SCHEMA} if loadings.shape[-1] > 2 else {}),
        "anchor_indices": list(anchors),
        "limits": dict(ORIENTATION_LIMITS),
        "counts": {
            "chains": int(chains),
            "draws_per_chain": int(draws),
            "total_draws": int(total),
            "retained_draws": int(total),
            "valid_anchor_draws": valid_count,
            "singular_anchor_draws": int(total - valid_count),
        },
    }


def _undefined_result(loadings, anchors, valid, status):
    result = _base_result(loadings, anchors, valid)
    result.update(
        {
            "status": status,
            "passes": False,
            "max_rank_rhat": None,
            "min_bulk_ess": None,
            "min_tail_ess": None,
            "per_quantity": _empty_quantities(status, loadings.shape[-1]),
            "chains": [
                {
                    "chain": int(chain),
                    "retained_draws": int(loadings.shape[1]),
                    "valid_anchor_draws": int(valid[chain].sum()),
                    "positive_determinant_fraction": None,
                    "reflection_transitions": None,
                }
                for chain in range(loadings.shape[0])
            ],
        }
    )
    return result


def _diagnostic_records(values, names):
    within_chain_variable = np.ptp(values, axis=1) != 0
    eligible = np.any(within_chain_variable, axis=0)
    table = (
        diagnostic_summary(values[..., eligible], name="orientation") if eligible.any() else None
    )
    records = []
    table_index = 0
    for quantity_index, name in enumerate(names):
        if not eligible[quantity_index]:
            status = (
                "all_draws_constant"
                if np.ptp(values[..., quantity_index]) == 0
                else "within_chain_constants_disagree"
            )
            records.append(
                {
                    "name": name,
                    "status": status,
                    "r_hat": None,
                    "ess_bulk": None,
                    "ess_tail": None,
                    "passes": False,
                }
            )
            continue
        diagnostic = table.iloc[table_index]
        table_index += 1
        numbers = {
            key: float(diagnostic[key]) if np.isfinite(diagnostic[key]) else None
            for key in ("r_hat", "ess_bulk", "ess_tail")
        }
        computed = all(value is not None for value in numbers.values())
        passes = bool(
            computed
            and numbers["r_hat"] <= ORIENTATION_LIMITS["max_rank_rhat"]
            and numbers["ess_bulk"] >= ORIENTATION_LIMITS["min_bulk_ess"]
            and numbers["ess_tail"] >= ORIENTATION_LIMITS["min_tail_ess"]
        )
        records.append(
            {
                "name": name,
                "status": "computed" if computed else "undefined_diagnostic",
                **numbers,
                "passes": passes,
            }
        )
    return records


def orientation_diagnostics(loadings, *, anchor_indices=None):
    """Summarize finite QR-chart mixing probes of posterior loadings.

    ``loadings`` has chain, draw, row and factor axes. For every draw, the
    selected K loading rows (the first K by default) define the positive-diagonal QR chart
    ``anchor.T = Q @ R``. The diagnostic coordinates are ``det(Q)`` and the
    first four sine/cosine harmonics of ``theta = atan2(Q[1, 0], Q[0, 0])``,
    both alone and multiplied by ``det(Q)`` for K=2. For K>=3, use
    each Q entry and its square, alone and determinant-weighted. These finite
    probes diagnose mixing; passing does not establish Haar or mode coverage.

    A rank-deficient anchor makes the chart undefined. Such draws are never
    removed: the entire diagnostic fails and the returned counts retain their
    presence. Exact within-chain constants are also explicit failures because
    ArviZ cannot assign them a meaningful rank-normalized R-hat.
    """
    loadings, anchors = _validate(loadings, anchor_indices)
    anchor = loadings[..., list(anchors), :]
    singular_values = np.linalg.svd(anchor, compute_uv=False)
    valid = singular_values[..., -1] > (
        ORIENTATION_LIMITS["rank_tolerance"] * singular_values[..., 0]
    )
    if not valid.all():
        return _undefined_result(loadings, anchors, valid, "singular_anchor")

    if (
        loadings.shape[0] < ORIENTATION_LIMITS["minimum_chains"]
        or loadings.shape[1] < ORIENTATION_LIMITS["minimum_draws"]
    ):
        return _undefined_result(loadings, anchors, valid, "insufficient_chains_or_draws")

    values = orientation_quantity_values(anchor)
    determinant = values[..., 0]
    records = _diagnostic_records(values, orientation_quantity_names(loadings.shape[-1]))
    computed_rhat = [r["r_hat"] for r in records if r["r_hat"] is not None]
    computed_bulk = [r["ess_bulk"] for r in records if r["ess_bulk"] is not None]
    computed_tail = [r["ess_tail"] for r in records if r["ess_tail"] is not None]

    result = _base_result(loadings, anchors, valid)
    result.update(
        {
            "status": "computed",
            "passes": bool(all(record["passes"] for record in records)),
            "max_rank_rhat": max(computed_rhat, default=None),
            "min_bulk_ess": min(computed_bulk, default=None),
            "min_tail_ess": min(computed_tail, default=None),
            "per_quantity": records,
            "chains": [
                {
                    "chain": int(chain),
                    "retained_draws": int(loadings.shape[1]),
                    "valid_anchor_draws": int(loadings.shape[1]),
                    "positive_determinant_fraction": float(np.mean(determinant[chain] > 0)),
                    "reflection_transitions": int(
                        np.sum(determinant[chain, 1:] != determinant[chain, :-1])
                    ),
                }
                for chain in range(loadings.shape[0])
            ],
        }
    )
    return result


__all__ = ["ORIENTATION_LIMITS", "orientation_diagnostics"]
