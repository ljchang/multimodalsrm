"""Differentiable real rational cascades with identity stationary covariance.

Topology is static; block entries and naive input/output maps may depend on
JAX parameters. No eigendecomposition, Gramian inversion, or global Kronecker
solve is required, including at coincident poles. Callers validate stable
pole domains before tracing; runtime values are never converted to NumPy.
"""

import numpy as np
from scipy.special import gammaln, logsumexp

from ._backend import runtime


def real_pole_block(rate):
    """Dissipative one-state block with pole ``-rate`` (rate must be positive)."""
    _, jnp, _, _ = runtime()
    rate = jnp.asarray(rate)
    return -rate.reshape((1, 1)), jnp.sqrt(2 * rate).reshape((1,))


def conjugate_pole_block(decay, frequency):
    """Real dissipative block with poles ``-decay +/- 1j*frequency``.

    Positive decay is required. Zero frequency is allowed and gives a repeated
    real pole without singular eigenvector coordinates.
    """
    _, jnp, _, _ = runtime()
    radius = jnp.hypot(decay, frequency)
    A = jnp.array([[0.0, radius], [-radius, -2 * decay]])
    b = jnp.array([0.0, 2 * jnp.sqrt(decay)])
    return A, b


def orthonormal_cascade(blocks):
    """Join stable ``(A_i, b_i)`` blocks satisfying A_i+A_i.T=-b_i b_i.T.

    Each block is real and has size one or two. The returned ``(G, b)`` has
    ``G+G.T=-b b.T`` and thus stationary covariance I. The first block is
    unchanged: passing the exact two-state Matérn realization preserves its
    latent coordinates. Off-diagonal blocks are ``-b_i b_j.T`` for i>j.
    """
    _, jnp, _, _ = runtime()
    blocks = tuple((jnp.asarray(A), jnp.asarray(b)) for A, b in blocks)
    if not blocks or any(
        b.ndim != 1 or b.size not in (1, 2) or A.shape != (b.size, b.size) for A, b in blocks
    ):
        raise ValueError("cascade blocks must be nonempty real one- or two-state blocks")
    b = jnp.concatenate([drive for _, drive in blocks])
    G = jnp.zeros((b.size, b.size), dtype=jnp.result_type(*[A for A, _ in blocks], b))
    start = 0
    for A, drive in blocks:
        end = start + drive.size
        G = G.at[start:end, start:end].set(A)
        G = G.at[start:end, :start].set(-drive[:, None] * b[None, :start])
        start = end
    return G, b


def block_sylvester(F, G, rhs, block_sizes):
    """Solve F T + T G.T = rhs for real block-lower-triangular F and G.

    ``block_sizes`` is a static partition of both matrices into 1x1 or 2x2
    diagonal blocks. Stability makes every local Sylvester operator invertible,
    even at equal poles. Inputs must have zero entries above these blocks.

    Pad each scalar block with an independent pole -1 and zero forcing, then
    run a single static-length JAX loop with 4x4 local solves. Padding avoids
    quadratic Python graph unrolling; no matrix of shape (d*d, d*d) is formed.
    The recurrence subtracts previously solved block rows and block columns.
    """
    jax, jnp, _, _ = runtime()
    F, G, rhs = map(jnp.asarray, (F, G, rhs))
    sizes = tuple(block_sizes)
    if not sizes or any(s not in (1, 2) for s in sizes):
        raise ValueError("block sizes must be a nonempty tuple of ones and twos")
    d = sum(sizes)
    if any(v.shape != (d, d) for v in (F, G, rhs)):
        raise ValueError("matrix shapes must agree with the block partition")
    n, padded = len(sizes), 2 * len(sizes)
    indices = np.array([2 * i + j for i, size in enumerate(sizes) for j in range(size)])
    dummy = np.ones(padded)
    dummy[indices] = 0
    index = jnp.asarray(indices)
    dtype = jnp.result_type(F, G, rhs, 1.0)
    Fp = (-jnp.diag(jnp.asarray(dummy, dtype=dtype))).at[index[:, None], index].set(F)
    Gp = (-jnp.diag(jnp.asarray(dummy, dtype=dtype))).at[index[:, None], index].set(G)
    Rp = jnp.zeros((padded, padded), dtype=dtype).at[index[:, None], index].set(rhs)
    eye = jnp.eye(2, dtype=dtype)

    def step(k, T):
        i, j = 2 * (k // n), 2 * (k % n)
        Frow = jax.lax.dynamic_slice(Fp, (i, 0), (2, padded))
        Grow = jax.lax.dynamic_slice(Gp, (j, 0), (2, padded))
        Trow = jax.lax.dynamic_slice(T, (i, 0), (2, padded))
        Tcol = jax.lax.dynamic_slice(T, (0, j), (padded, 2))
        residual = jax.lax.dynamic_slice(Rp, (i, j), (2, 2)) - Frow @ Tcol - Trow @ Grow.T
        A = jax.lax.dynamic_slice(Fp, (i, i), (2, 2))
        D = jax.lax.dynamic_slice(Gp, (j, j), (2, 2))
        # Row-major vectorization of A X + X D.T.
        operator = jnp.kron(A, eye) + jnp.kron(eye, D)
        solution = jnp.linalg.solve(operator, residual.reshape(4)).reshape((2, 2))
        return jax.lax.dynamic_update_slice(T, solution, (i, j))

    T = jax.lax.fori_loop(0, n * n, step, jnp.zeros_like(Rp))
    return T[index[:, None], index]


def rational_realization(F_naive, B_naive, C_naive, blocks):
    """Return ``(G, b, C)`` for an equivalent single-input rational system.

    F_naive uses the same ordered block partition and poles as ``blocks``;
    its lower off-diagonal blocks can describe parallel banks or downstream
    response filters. The stable cross Gramian T solves
    ``F_naive T + T G.T = -B_naive b.T`` and outputs are ``C_naive T``.
    With matching pole multisets this preserves the input-output transfer.
    """
    _, jnp, _, _ = runtime()
    blocks = tuple(blocks)
    G, b = orthonormal_cascade(blocks)
    B, C = jnp.asarray(B_naive), jnp.asarray(C_naive)
    if B.shape != b.shape or C.ndim not in (1, 2) or C.shape[-1] != b.size:
        raise ValueError("naive input/output shapes must agree with cascade blocks")
    T = block_sylvester(F_naive, G, -B[:, None] * b[None, :], tuple(A.shape[0] for A, _ in blocks))
    return G, b, C @ T


def safe_decay_cutoff(dimension, spectral_decay_lower, generator_norm_upper, tolerance=1e-14):
    """Uniform cutoff for ||exp(G*t)||_2 <= tolerance over a pole domain.

    Inputs must bound dimension d, Re(eigenvalue(G)) <= -a < 0, and
    ||G||_2 <= M throughout the domain. A Frobenius norm upper bound is valid.
    Complex Schur decomposition G=U(D+N)U* gives ||N||_2 <= 2M. Its triangular
    Dyson expansion terminates after d terms, so

        ||exp(G*t)||_2 <= exp(-a*t) sum_{k=0}^{d-1} (2M*t)^k/k!.

    This also covers defective and coincident poles. Every term decreases for
    t >= (d-1)/a. Bisection on that tail returns its upper endpoint, not an
    unconstrained numerical root. The bound is analytic in exact arithmetic;
    this scalar floating-point evaluation is not an interval certificate.
    For a stationary identity covariance, replacing a transition by zero
    and its innovation covariance by I has covariance error <= tolerance**2.
    """
    if (
        not isinstance(dimension, (int, np.integer))
        or dimension < 1
        or not np.isfinite([spectral_decay_lower, generator_norm_upper, tolerance]).all()
        or spectral_decay_lower <= 0
        or generator_norm_upper <= 0
        or not 0 < tolerance < 1
    ):
        raise ValueError(
            "cutoff requires positive finite decay/norm, dimension, and tolerance in (0,1)"
        )
    a, M = float(spectral_decay_lower), float(generator_norm_upper)
    k = np.arange(dimension)
    log_factorial = gammaln(k + 1)
    target = np.log(tolerance)

    def log_bound(t):
        return -a * t + logsumexp(k * (np.log(2.0) + np.log(M) + np.log(t)) - log_factorial)

    lower = max((dimension - 1) / a, np.finfo(float).tiny)
    upper = max(lower, -target / a)
    while log_bound(upper) > target:
        upper *= 2
        if not np.isfinite(upper):
            raise ValueError("pole domain cannot produce a finite floating-point cutoff")
    for _ in range(80):
        middle = lower + (upper - lower) / 2
        if log_bound(middle) > target:
            lower = middle
        else:
            upper = middle
    # Small outward margin avoids equality moving above the bound on rounding.
    return float(upper * (1 + 16 * np.finfo(float).eps))
