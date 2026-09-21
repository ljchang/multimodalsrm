"""Differentiable finite response functionals, with shared quadrature rules.

Every covariance is a weighted Matérn Gram block. This preserves positive
semidefiniteness without discretizing the latent trajectory or observation clock.
Quadrature order controls an approximation, not a certified entrywise bound.
"""

from functools import lru_cache

import numpy as np

from ..kernels import BachSCR, DoubleGamma, Gamma, Gaussian, Identity
from ._backend import runtime

FAMILIES = (Identity, Gaussian, Gamma, DoubleGamma, BachSCR)


def validate_order(order):
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or not 8 <= order <= 1024
    ):
        raise ValueError("response_quadrature_order must be an integer from 8 to 1024")


@lru_cache(None)
def gamma_tail():
    """Unit-scale gamma 1-1e-8 quantile, with its implicit shape derivative."""
    jax, jnp, jsp, _ = runtime()

    @jax.custom_jvp
    def quantile(shape):
        def step(_, bounds):
            lo, hi = bounds
            mid = (lo + hi) / 2
            small = jsp.special.gammaincc(shape, mid) > 1e-8
            return jnp.where(small, mid, lo), jnp.where(small, hi, mid)

        lo, hi = jax.lax.fori_loop(
            0, 60, step, (jnp.asarray(0.0), shape + 50 + 20 * jnp.sqrt(shape))
        )
        return (lo + hi) / 2

    @quantile.defjvp
    def derivative(primals, tangents):
        (shape,), (tangent,) = primals, tangents
        value = quantile(shape)
        _, d_shape = jax.jvp(lambda a: jsp.special.gammaincc(a, value), (shape,), (tangent,))
        pdf = jnp.exp((shape - 1) * jnp.log(value) - value - jsp.special.gammaln(shape))
        return value, d_shape / pdf

    return quantile


def rule(order, edges):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = np.asarray(edges[:-1]), np.asarray(edges[1:])
    return ((a[:, None] + b[:, None]) / 2 + (b - a)[:, None] * nodes / 2).ravel(), (
        (b - a)[:, None] * weights / 2
    ).ravel()


def weighted_matern(y, nodes, weights, length):
    """Sum weighted Matérn values in O(Q) preprocessing and O(log Q) per query.

    Prefix and suffix exponential moments avoid a Q by Q tensor integral.
    Nodes are increasing. Clamping exponential coordinates at the node endpoints
    keeps very distant queries finite while retaining their exact decay.
    """
    _, jnp, _, _ = runtime()
    rate = np.sqrt(3.0) / length
    center = (nodes[0] + nodes[-1]) / 2
    b = nodes - center
    plus = weights * jnp.exp(rate * b)
    minus = weights * jnp.exp(-rate * b)
    zero = jnp.zeros(1)
    p0 = jnp.concatenate((zero, jnp.cumsum(plus)))
    p1 = jnp.concatenate((zero, jnp.cumsum(plus * b)))
    s0 = jnp.concatenate((jnp.cumsum(minus[::-1])[::-1], zero))
    s1 = jnp.concatenate((jnp.cumsum((minus * b)[::-1])[::-1], zero))
    index = jnp.searchsorted(nodes, y)
    z = jnp.clip(y, nodes[0], nodes[-1]) - center
    d = y - center
    outside = jnp.abs(y - jnp.clip(y, nodes[0], nodes[-1]))
    left = jnp.exp(-rate * z - rate * outside) * ((1 + rate * d) * p0[index] - rate * p1[index])
    right = jnp.exp(rate * z - rate * outside) * ((1 - rate * d) * s0[index] + rate * s1[index])
    return left + right


class ResponseQuadrature:
    def __init__(self, responses, indices, length_scale, order, *, minimum_length_scale=None):
        validate_order(order)
        self.responses = list(responses.values())
        self.names = list(responses)
        self.indices = indices
        self.length = length_scale
        self.order = int(order)
        self.positive_rule = rule(order, [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
        self.gaussian_rule = rule(order, [-6, -4, -2, -1, 0, 1, 2, 4, 6])
        self.scr_rule = rule(order, [0, 1, 2, 3, 4, 6, 10, 20, 40, 60, 90])
        for response in self.responses:
            kernel = response.initial_kernel()
            if type(kernel) not in FAMILIES:
                raise ValueError("unsupported quadrature response family")
            if type(kernel) is BachSCR and {"t0", "lag"} <= set(response.free_parameters):
                raise ValueError("fix BachSCR t0 when learning lag")
            if type(kernel) is BachSCR:
                t0_max = (
                    response.parameter_bounds()["t0"][1]
                    if "t0" in response.free_parameters
                    else kernel.t0
                )
                if t0_max >= 90:
                    raise ValueError("BachSCR t0 must remain below its 90-second support")
            if type(kernel) is DoubleGamma:
                bounds = response.parameter_bounds()

                def extreme(name, side):
                    return (
                        bounds[name][side]
                        if name in response.free_parameters
                        else kernel.parameters[name]
                    )

                if extreme("undershoot_shape", 0) * extreme("undershoot_scale", 0) <= extreme(
                    "peak_shape", 1
                ) * extreme("peak_scale", 1):
                    raise ValueError(
                        "DoubleGamma bounds must preserve peak/undershoot ordering throughout"
                    )
            lo, hi = response.support_envelope()
            shortest = length_scale if minimum_length_scale is None else minimum_length_scale
            if np.sqrt(3.0) * (hi - lo) / shortest > 300:
                raise ValueError("response support/GP timescale exceeds stable quadrature range")

    def metadata(self):
        return dict(
            method="finite_response_gauss_legendre",
            order_per_panel=self.order,
            normalization="quadrature_l2",
            latent_time_grid=False,
            error_bound=None,
            accuracy_requires_order_comparison=True,
        )

    def nodes(self, x, modality):
        _, jnp, jsp, _ = runtime()
        if modality == -1:
            return jnp.zeros(1), jnp.ones(1)
        response = self.responses[modality]
        kernel = response.initial_kernel()
        if type(kernel) is Identity:
            return jnp.zeros(1), jnp.ones(1)
        p = {
            name: x[self.indices["filter", self.names[modality], name]]
            if ("filter", self.names[modality], name) in self.indices
            else jnp.asarray(value)
            for name, value in kernel.parameters.items()
        }
        if type(kernel) is Gaussian:
            u, q = map(jnp.asarray, self.gaussian_rule)
            u, q = u * p["width"], q * p["width"]
            raw = jnp.exp(-0.5 * (u / p["width"]) ** 2)
        elif type(kernel) in (Gamma, DoubleGamma):

            def component(prefix, u):
                a, s = p[prefix + "shape"], p[prefix + "scale"]
                return jnp.exp((a - 1) * jnp.log(u / s) - u / s - jsp.special.gammaln(a)) / s

            if type(kernel) is Gamma:
                end = gamma_tail()(p["shape"]) * p["scale"]
            else:
                end = jnp.maximum(
                    gamma_tail()(p["peak_shape"]) * p["peak_scale"],
                    gamma_tail()(p["undershoot_shape"]) * p["undershoot_scale"],
                )
            u, q = map(jnp.asarray, self.positive_rule)
            u, q = u * end, q * end
            raw = (
                component("", u)
                if type(kernel) is Gamma
                else component("peak_", u) - p["undershoot_ratio"] * component("undershoot_", u)
            )
        else:
            u, q = map(jnp.asarray, self.scr_rule)
            log_components = []
            for key in ("lambda1", "lambda2"):
                rate = p[key]
                center = p["t0"] + rate * p["sigma"] ** 2
                low, high = -center / p["sigma"], (u - center) / p["sigma"]
                # log-CDF subtraction avoids overflow in broad/fast-decay shapes.
                log_hi, log_lo = jsp.special.log_ndtr(high), jsp.special.log_ndtr(low)
                difference = -jnp.expm1(log_lo - log_hi)
                log_components.append(
                    -rate * u
                    + rate * p["t0"]
                    + 0.5 * (rate * p["sigma"]) ** 2
                    + log_hi
                    + jnp.log(difference)
                )
            log_raw = jsp.special.logsumexp(jnp.stack(log_components), axis=0)
            raw = jnp.exp(log_raw - jnp.max(log_raw))
        coefficients = q * raw / jnp.sqrt(jnp.sum(q * raw**2))
        return u + p["lag"], coefficients

    def pair(self, delta, left, right, length_scale=None):
        jax, jnp, _, _ = runtime()
        u, wa = left
        v, wb = right
        flat = jnp.asarray(delta).reshape(-1)
        length = self.length if length_scale is None else length_scale

        def one(d):
            return jnp.sum(wa * weighted_matern(d - u, -v[::-1], wb[::-1], length))

        result = jax.lax.map(one, flat, batch_size=32)
        return result.reshape(delta.shape)

    def covariance(self, x, times_a, modalities_a, times_b, modalities_b, *, length_scale=None):
        _, jnp, _, _ = runtime()
        a, b = jnp.asarray(times_a), jnp.asarray(times_b)
        ma, mb = np.asarray(modalities_a), np.asarray(modalities_b)
        result = jnp.zeros((len(a), len(b)))
        nodes = {int(m): self.nodes(x, int(m)) for m in np.union1d(ma, mb)}
        for m in np.unique(ma):
            ia = np.flatnonzero(ma == m)
            for n in np.unique(mb):
                ib = np.flatnonzero(mb == n)
                delta = a[ia, None] - b[ib][None, :]
                if isinstance(times_a, np.ndarray) and isinstance(times_b, np.ndarray):
                    # Static native clocks: reuse exact differences, never bins.
                    unique, inverse = np.unique(
                        times_a[ia, None] - times_b[ib][None, :], return_inverse=True
                    )
                    values = self.pair(
                        jnp.asarray(unique), nodes[int(m)], nodes[int(n)], length_scale
                    )
                    block = values[inverse].reshape(delta.shape)
                else:
                    block = self.pair(delta, nodes[int(m)], nodes[int(n)], length_scale)
                result = result.at[np.ix_(ia, ib)].set(block)
        return result
