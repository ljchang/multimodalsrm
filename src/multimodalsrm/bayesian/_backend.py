"""Optional float64 runtime and differentiable continuous response covariance."""

import numpy as np
from scipy.special import erf, ndtr


def runtime():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        import numpyro.distributions as dist
    except ImportError as exc:
        raise ImportError("Bayesian SRM requires multimodalsrm[bayesian]") from exc
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Bayesian SRM requires float64: set JAX_ENABLE_X64=true before starting Python"
        )
    return jax, jnp, jsp, dist


def accuracy_bound(responses, length_scale, tolerance):
    """Conservative temporal-entry bound over every allowed filter width.

    Six-sigma tail mass times maximal response mass bounds the omitted
    integral, since the unit-variance Matérn covariance lies in [0, 1]. The
    rounding term is an estimate, not interval arithmetic certification.
    Unsupported broad-filter cancellation regimes are rejected explicitly.
    """
    from ..kernels import Gaussian

    widths = [
        r.parameter_bounds()["width"][1]
        if "width" in r.free_parameters
        else r.initial_kernel().width
        for r in responses.values()
        if type(r.initial_kernel()) is Gaussian
    ]
    if not widths:
        return 0.0
    width = max(widths)
    mass = max(1.0, np.sqrt(2) * np.pi**0.25 * np.sqrt(width) / np.sqrt(erf(6.0)))
    q = np.sqrt(6.0) * width / length_scale
    tail = mass**2 * (4 * ndtr(-6.0) - 4 * ndtr(-6.0) ** 2)
    bound = tail + 64 * np.finfo(float).eps * mass**2 * (1 + q * q)
    if q > 5 or bound > tolerance:
        raise ValueError(
            "requested covariance accuracy is unsupported for these width/length_scale bounds; narrow widths or use the finite-integral continuous reference"
        )
    return float(bound)


def temporal(times_a, times_b, width_a, width_b, lag_a, lag_b, length_scale):
    """Gaussian/Identity functional covariance with finite-L2 normalization."""
    _, jnp, _, _ = runtime()
    delta = (
        jnp.asarray(times_a)[:, None]
        - jnp.asarray(times_b)[None, :]
        - lag_a[:, None]
        + lag_b[None, :]
    )
    return temporal_values(
        delta, jnp.asarray(width_a)[:, None], jnp.asarray(width_b)[None, :], length_scale
    )


def temporal_values(delta, width_a, width_b, length_scale):
    """Evaluate broadcast-compatible separations without forming a pair grid."""
    _, jnp, jsp, _ = runtime()
    wa, wb = jnp.asarray(width_a), jnp.asarray(width_b)
    identity = (wa == 0) & (wb == 0)
    sigma = jnp.sqrt(wa**2 + wb**2 + identity)
    rate = np.sqrt(3.0) / length_scale
    d = jnp.abs(delta)
    q, r = rate * sigma, d / sigma
    plus = jnp.exp(0.5 * q * q - rate * d + jsp.special.log_ndtr(r - q))
    minus = jnp.exp(0.5 * q * q + rate * d + jsp.special.log_ndtr(-r - q))
    normal = (
        (1 + rate * d - q * q) * plus
        + (1 - rate * d - q * q) * minus
        + 2 * q * jnp.exp(-0.5 * r * r) / np.sqrt(2 * np.pi)
    )

    def mass(w):
        safe = jnp.where(w == 0, 1.0, w)
        return jnp.where(w == 0, 1.0, np.sqrt(2) * np.pi**0.25 * jnp.sqrt(safe) / np.sqrt(erf(6.0)))

    return jnp.where(
        identity,
        (1 + rate * d) * jnp.exp(-rate * d),
        normal * mass(wa) * mass(wb),
    )
