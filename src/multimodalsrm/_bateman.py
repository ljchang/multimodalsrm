"""Shared NumPy/JAX Bateman algebra, including the equal-pole limit."""

import math

import numpy as np

SCR_DURATION = 90.0


def raw(t, rise, decay, xp=np):
    """Unit-mass two-stage impulse before finite truncation and L2 scaling."""
    t = (
        xp.maximum(xp.real(t), 0) + 1j * xp.imag(t)
        if xp is np and np.iscomplexobj(t)
        else xp.maximum(t, 0)
    )
    a, b = 1 / rise, 1 / decay
    delta = a - b
    x = delta * t
    near = xp.abs(x) < 0.5
    # Keep inactive branches finite, including their JAX derivatives.
    safe = xp.where(near, 1.0, delta)
    ea, eb = xp.exp(-a * t), xp.exp(-b * t)
    direct = a * b * (eb - ea) / safe
    z = xp.where(near, x, 0.0)
    # Integral_0^1 exp(-x*v) dv, accurate at and near coincident poles.
    polynomial = 0.0
    for j in range(18, -1, -1):
        polynomial = polynomial * (-z) + 1 / math.factorial(j + 1)
    return xp.where(near, a * b * eb * t * polynomial, direct)


def energy(rise, decay, xp=np):
    """Exact finite L2 norm using the two-state observability Gramian.

    For A=[[-1/r,0],[1/d,-1/d]], Q solves A.T Q + Q A=-e_2 e_2.T.
    Subtract x(T).T Q x(T) from the infinite energy. No differences of
    poles occur, so equal and nearly equal time constants stay stable.
    """
    first = xp.exp(-SCR_DURATION / rise) / rise
    last = raw(SCR_DURATION, rise, decay, xp)
    total = rise + decay
    tail = (
        rise**2 / (2 * total) * first**2 + rise * decay / total * first * last + decay / 2 * last**2
    )
    return xp.sqrt(1 / (2 * total) - tail)


def tail_mass(rise, decay):
    """Probability remaining in the two exponential stages after 90 seconds."""
    return float(np.exp(-SCR_DURATION / rise) + decay * raw(SCR_DURATION, rise, decay))


def bounds(box):
    """Uniform L1 mass and restored-tail bounds over the declared parameter box."""
    rlo, rhi = box["rise"]
    dlo, dhi = box["decay"]
    tail = tail_mass(rhi, dhi)
    # Infinite L2 energy decreases with either time constant. The convolution
    # density is <= the smaller stage-density supremum, bounding tail energy.
    lower = 1 / (2 * (rhi + dhi)) - tail / max(rlo, dlo)
    # Cauchy-Schwarz supplies another lower bound on finite energy.
    lower = max(lower, (1 - tail) ** 2 / SCR_DURATION)
    if lower <= 0 or not np.isfinite(lower):
        raise ValueError("SCR parameter bounds do not give a positive finite energy bound")
    mass = 1 / np.sqrt(lower)
    return mass, tail * mass
