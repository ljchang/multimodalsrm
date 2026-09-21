"""Independent finite-response conditioning oracle from frozen source.

Only numerical reference calculations are retained; no research runner or
external artifacts are imported.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from multimodalsrm import Gaussian, Identity

from .continuous_covariance import response_covariance


def reference(problem, x, query=None):
    """Independent SciPy observation conditioning, including full factor covariance.

    Extract by parameter names; use the continuous SciPy response integrals,
    not the JAX covariance, grouped solves, or production query code.
    """
    pars = dict(zip(problem.names, x, strict=True))
    system = problem.systems["train"]
    k = problem.features
    weights = np.array([[pars[("loading", *key, f)] for f in range(k)] for key in system.keys])
    offset = np.array([pars[("offset", *key)] for key in system.keys])
    kernels = {}
    for modality, response in problem.responses.items():
        initial = response.initial_kernel()
        kernels[modality] = (
            Identity()
            if type(initial) is Identity
            else Gaussian(
                pars.get(("filter", modality, "width"), initial.width),
                pars.get(("filter", modality, "lag"), initial.lag),
            )
        )
    temporal = np.empty((len(system.times), len(system.times)))
    indices = {m: np.flatnonzero([key[1] == m for key in system.keys]) for m in kernels}
    for a, ia in indices.items():
        for b, ib in indices.items():
            temporal[np.ix_(ia, ib)] = response_covariance(
                system.times[ia],
                system.times[ib],
                kernels[a],
                kernels[b],
                problem.length_scale,
            )
    covariance = temporal * (weights @ weights.T)
    covariance += np.diag([pars[("noise", *key[:2])] for key in system.keys])
    cf = cho_factor(covariance, lower=True)
    residual = system.values - offset
    alpha = cho_solve(cf, residual)
    nll = 0.5 * (
        residual @ alpha + 2 * np.log(np.diag(cf[0])).sum() + len(residual) * np.log(2 * np.pi)
    )
    if query is None:
        return float(nll), None, None
    cross = np.empty((len(query), len(system.times)))
    for modality, ix in indices.items():
        cross[:, ix] = response_covariance(
            query,
            system.times[ix],
            Identity(),
            kernels[modality],
            problem.length_scale,
        )
    operator = cross[:, None, :] * weights.T[None, :, :]
    mean = operator @ alpha
    conditional = np.array([np.eye(k) - h @ cho_solve(cf, h.T) for h in operator])
    return float(nll), mean, conditional
