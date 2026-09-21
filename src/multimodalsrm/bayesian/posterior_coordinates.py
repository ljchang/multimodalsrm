"""Draw-specific reporting pushforward; never used in the posterior density."""

import numpy as np


def metadata(keys):
    """Describe fitted anchors without imposing a rotation on the MAP start."""
    return dict(
        method="draw_specific_anchor_QR_positive_diagonal",
        anchors=[list(k) for k in keys],
        relative_rank_tolerance=1e-10,
        parameter_coordinates="raw_physical",
        reported_loadings="W_raw @ rotation_draw",
        reported_latent="Z_raw @ rotation_draw",
        estimated_from="each_posterior_draw_using_fitted_training_anchors",
        physiological_identification=False,
    )


def rotations(problem, draws, keys, draw_indices):
    """Positive-diagonal QR for every selected draw, retaining original IDs."""
    indices = [
        [problem.indices[("loading", *key, f)] for f in range(problem.features)] for key in keys
    ]
    result = np.empty((len(draws), problem.features, problem.features))
    for i, (x, (chain, draw)) in enumerate(zip(draws, draw_indices, strict=True)):
        block = x[indices]
        location = f"chain {chain}, draw {draw}"
        if not np.isfinite(block).all():
            raise ValueError(f"factor anchor loading block is nonfinite at {location}")
        singular = np.linalg.svd(block, compute_uv=False)
        if singular[0] == 0 or singular[-1] <= 1e-10 * singular[0]:
            raise ValueError(f"factor anchor loading block is rank deficient at {location}")
        Q, R = np.linalg.qr(block.T)
        result[i] = Q * np.where(np.diag(R) >= 0, 1.0, -1.0)[None, :]
    return result
