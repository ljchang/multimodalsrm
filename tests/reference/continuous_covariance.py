"""Continuous Matérn-3/2 covariance of finite response functionals.

Gaussian responses retain the existing six-sigma support and continuous L2
normalization. Accuracy here is absolute, per covariance entry; it has no
connection to a latent time grid. Float64 roundoff limits the requested accuracy.
"""

from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfcx, ndtr

from multimodalsrm.kernels import Gaussian, Identity

_TAIL = 2 * ndtr(-6.0)
_EPS = np.finfo(float).eps


def _matern(distance, rate):
    with np.errstate(over="ignore", invalid="ignore"):
        x = rate * np.abs(distance)
        value = (1 + x) * np.exp(-x)
    return np.where(np.isinf(x), 0.0, value)


def _full_gaussian(distance, sigma, rate):
    """E[(1+a|X|)exp(-a|X|)] for X ~ N(distance, sigma**2).

    Differentiate the two half-line Laplace integrals of a normal density.
    erfcx avoids overflow when the exponential and normal CDF compensate.
    The caller avoids the large a*sigma cancellation regime.
    """
    d = np.abs(distance)
    q, r = rate * sigma, d / sigma
    density = np.exp(-0.5 * r * r)
    plus = np.empty_like(d)
    stable = q >= r
    plus[stable] = 0.5 * density[stable] * erfcx((q - r[stable]) / np.sqrt(2))
    plus[~stable] = np.exp(0.5 * q * q - rate * d[~stable]) * ndtr(r[~stable] - q)
    minus = 0.5 * density * erfcx((q + r) / np.sqrt(2))
    return (
        (1 + rate * d - q * q) * plus
        + (1 - rate * d - q * q) * minus
        + 2 * q * density / np.sqrt(2 * np.pi)
    )


def _finite_integral(distance, widths, rate, mass, tolerance):
    """Exact finite-support reduction to one adaptive scalar integral.

    For two normals U,V, write W=U-V, s²=wa²+wb². Given W=w,
    U is normal with mean w*wa²/s² and sd wa*wb/s. The probability
    U in [-6wa,6wa] intersect [w-6wb,w+6wb] is the finite-support
    correction to the full normal density of W. This reduction also
    accommodates one impulse by directly integrating the other response.
    """
    sigma = np.hypot(*widths) if len(widths) == 2 else widths[0]
    bound = 6 * sum(widths) / sigma
    cusp = distance / sigma
    points = [-bound, bound, -4.0, -2.0, 0.0, 2.0, 4.0, cusp]
    if len(widths) == 2:
        wa, wb = sorted(widths)  # Canonical orientation preserves transpose.
        conditional_sd = wa * (wb / sigma)
        switch = 6 * (wa - wb) / sigma
        points += [switch, -switch]

        def finite_density(z):
            w = sigma * z
            mean = w * (wa / sigma) ** 2
            lo = (max(-6 * wa, w - 6 * wb) - mean) / conditional_sd
            hi = (min(6 * wa, w + 6 * wb) - mean) / conditional_sd
            # Survival form prevents cancellation in the positive tail.
            probability = ndtr(-lo) - ndtr(-hi) if lo > 0 else ndtr(hi) - ndtr(lo)
            return exp(-0.5 * z * z) / sqrt(2 * pi) * max(0.0, probability)
    else:

        def finite_density(z):
            return exp(-0.5 * z * z) / sqrt(2 * pi)

    q = rate * sigma
    if q > 10:
        # Integrate in the Matérn decay coordinate x=q*(z-cusp). Fixed
        # normal-coordinate panels can entirely miss material tails when q
        # is large, even though QUADPACK reports a tiny estimated error.
        # The finite normal density is <= 1/sqrt(2*pi); consequently the
        # two-sided omitted |x|>R integral is bounded by
        # 2*mass/(q*sqrt(2*pi)) * (R+2)*exp(-R).
        cutoff = 16.0
        scale = mass / q
        while 2 * scale / sqrt(2 * pi) * (cutoff + 2) * exp(-cutoff) > tolerance / 8:
            cutoff += 8.0
        lower = max(-cutoff, q * (-bound - cusp))
        upper = min(cutoff, q * (bound - cusp))
        if lower >= upper:
            return 0.0
        points = [q * (point - cusp) for point in points]
        points += [-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0]

        def integrand(x):
            return finite_density(cusp + x / q) * (1 + abs(x)) * exp(-abs(x))
    else:
        scale = mass
        lower, upper = -bound, bound
        points += [cusp + multiple / q for multiple in (-16, -4, -1, 1, 4, 16)]

        def integrand(z):
            x = rate * abs(distance - sigma * z)
            return finite_density(z) * (1 + x) * exp(-x)

    points = sorted(set(p for p in points if lower < p < upper))
    result, error = quad(
        integrand,
        lower,
        upper,
        points=points,
        epsabs=tolerance / (4 * scale),
        epsrel=0.0,
        limit=300,
    )
    if not np.isfinite(result) or scale * error > tolerance / 2:
        raise ArithmeticError("continuous covariance quadrature did not attain tolerance")
    return scale * result


def response_covariance(times_a, times_b, kernel_a, kernel_b, length_scale, *, tolerance=1e-7):
    """Return continuous response covariance for two finite 1D time arrays.

    ``C_ab(t,t') = integral h_a(u) h_b(v) k(t-t'-u+v) du dv``,
    with unit-variance Matérn-3/2 k. Thus the peak occurs at
    ``t-t' = lag_a-lag_b``. Times may repeat, be unsorted, or be empty.
    Only the exact built-in Identity and Gaussian families are supported.

    For a Gaussian of width w, its finite L2 energy is
    ``sqrt(w*sqrt(pi)*erf(6))``. The corresponding untruncated L1 mass
    is ``M=sqrt(2)*pi**.25*sqrt(w)/sqrt(erf(6))``. Each omitted tail
    has probability ``p=2*Phi(-6)``. Since 0 <= k <= 1, replacing n
    finite Gaussians by full Gaussians changes an entry by at most
    ``prod(M) * (1-(1-p)**n)`` (n=1 or 2), independent of lag/ell.
    We use that analytic shortcut only when this bound is below tolerance/4
    and a conservative cancellation/roundoff estimate is below tolerance/4.
    Otherwise adaptive integration uses the exact finite support; its estimated
    absolute error must be below tolerance/2. No Gaussian-tail error remains
    on that path. For broad responses integration uses the Matérn decay
    coordinate x; any omitted |x|>R tail is bounded by
    ``2*mass/(a*sigma*sqrt(2*pi))*(R+2)*exp(-R) <= tolerance/8``.
    Entries whose finite-support upper bound is below tolerance/8
    may be set to zero. Repeated absolute centered differences are evaluated once.

    ``tolerance`` is an absolute per-entry target, not a rigorous floating-point
    interval certificate or a guarantee of exact PSD after rounding. Requests
    below 64 machine eps times max(1, product(M)) are rejected. Very extreme
    finite time/width/length-scale ratios outside float64 arithmetic are rejected.
    """
    arrays = []
    for times in (times_a, times_b):
        values = np.asarray(times, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("times must be finite one-dimensional arrays")
        arrays.append(values)
    if not np.isscalar(length_scale) or not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be positive and finite")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if any(type(k) not in (Identity, Gaussian) for k in (kernel_a, kernel_b)):
        raise TypeError("only built-in Identity and Gaussian kernels are supported")
    gaussians = [k for k in (kernel_a, kernel_b) if type(k) is Gaussian]
    widths = [k.width for k in gaussians]
    mass = float(np.prod([np.sqrt(2) * np.pi**0.25 * np.sqrt(w) / np.sqrt(erf(6)) for w in widths]))
    if not np.isfinite(mass) or tolerance < 64 * _EPS * max(1.0, mass):
        raise ValueError("tolerance is below the float64 covariance accuracy floor")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = arrays[0][:, None] - arrays[1][None, :]
        delta = delta - getattr(kernel_a, "lag", 0.0) + getattr(kernel_b, "lag", 0.0)
        rate = np.sqrt(3) / length_scale
    if not np.isfinite(delta).all() or not np.isfinite(rate):
        raise ValueError("time or length_scale exceeds float64 arithmetic range")
    if not widths:
        return _matern(delta, rate)
    if not delta.size:
        return np.empty(delta.shape)
    sigma = float(np.hypot(*widths)) if len(widths) == 2 else widths[0]
    if not np.isfinite(sigma) or not np.isfinite(rate * sigma):
        raise ValueError("response width exceeds float64 arithmetic range")
    support_radius = 6 * sum(widths)
    if not np.isfinite(support_radius):
        raise ValueError("response support exceeds float64 arithmetic range")
    distances, inverse = np.unique(np.abs(delta), return_inverse=True)
    # Positivity and monotonicity give an absolute bound beyond joint support.
    negligible = mass * _matern(np.maximum(0.0, distances - support_radius), rate) <= tolerance / 8
    values = np.zeros_like(distances)
    active = distances[~negligible]
    tail_bound = mass * (_TAIL if len(widths) == 1 else 2 * _TAIL - _TAIL**2)
    q = rate * sigma
    roundoff_bound = 64 * _EPS * mass * (1 + q * q)
    if tail_bound <= tolerance / 4 and roundoff_bound <= tolerance / 4 and q <= 10:
        values[~negligible] = mass * _full_gaussian(active, sigma, rate)
    else:
        values[~negligible] = np.array(
            [_finite_integral(d, widths, rate, mass, tolerance) for d in active]
        )
    return values[inverse].reshape(delta.shape)
