"""Research probe of Gaussian loading/offset marginalization for one complete block.

All three functions evaluate the SAME loading-marginalized observation density.
This is an algebra/gradient check, not a sampler, a GP-prior benchmark, or a
replacement for the package's latent-marginalized MAP objective. Assumes K>1,
unconstrained isotropic normal loadings, normal offsets, and shared white noise.
Use PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=int, default=252)
    parser.add_argument("--channels", type=int, default=3600)
    parser.add_argument("--factors", type=int, default=3)
    args = parser.parse_args()
    if not jax.config.x64_enabled or args.factors < 2:
        raise ValueError("this probe requires float64 and at least two factors")
    rng = np.random.default_rng(721)
    Y = rng.normal(size=(args.times, args.channels))
    start = time.perf_counter()
    gram = Y @ Y.T
    prep = time.perf_counter() - start
    norm = float(np.sum(Y * Y))
    Y, gram = jnp.asarray(Y), jnp.asarray(gram)

    # Unit loading SD, offset SD 0.5, shared noise variance v.
    def unpack(x):
        Z = x[:-1].reshape(args.times, args.factors)
        F = jnp.concatenate((Z, 0.5 * jnp.ones((args.times, 1))), axis=1)
        return F, x[-1]

    def dense(x):
        F, v = unpack(x)
        C = v * jnp.eye(args.times) + F @ F.T
        L = jnp.linalg.cholesky(C)
        return 0.5 * (
            args.channels * (args.times * np.log(2 * np.pi) + 2 * jnp.log(jnp.diag(L)).sum())
            + jnp.sum(Y * jsp.linalg.cho_solve((L, True), Y))
        )

    def woodbury(x, use_gram):
        F, v = unpack(x)
        B = jnp.eye(args.factors + 1) + F.T @ F / v
        L = jnp.linalg.cholesky(B)
        if use_gram:
            T = F.T @ gram @ F
            correction = jnp.trace(jsp.linalg.cho_solve((L, True), T)) / v**2
        else:
            S = F.T @ Y
            correction = jnp.sum(S * jsp.linalg.cho_solve((L, True), S)) / v**2
        return 0.5 * (
            args.channels * (args.times * jnp.log(2 * np.pi * v) + 2 * jnp.log(jnp.diag(L)).sum())
            + norm / v
            - correction
        )

    x = jnp.asarray(np.r_[rng.normal(size=args.times * args.factors), 0.7])
    results = {}
    row = {
        "config": vars(args),
        "jax": jax.__version__,
        "x64": jax.config.x64_enabled,
        "devices": [str(d) for d in jax.devices()],
        "load1": os.getloadavg()[0],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "gram_prepare_s": prep,
        "gram_bytes": gram.nbytes,
        "raw_data_bytes": Y.nbytes,
        "eliminated_loading_offset_parameters": args.channels * (args.factors + 1),
        "retained_latent_noise_parameters": len(x),
    }
    for name, function in (
        ("dense", dense),
        ("woodbury_raw", lambda x: woodbury(x, False)),
        ("woodbury_gram", lambda x: woodbury(x, True)),
    ):
        start = time.perf_counter()
        compiled = jax.jit(jax.value_and_grad(function)).lower(x).compile()
        value, gradient = jax.block_until_ready(compiled(x))
        cold = time.perf_counter() - start
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            value, gradient = jax.block_until_ready(compiled(x))
            durations.append(time.perf_counter() - start)
        results[name] = float(value), np.asarray(gradient)
        row[name] = {
            "compile_and_first_s": cold,
            "median_value_gradient_s": float(np.median(durations)),
            "warm_repeats_s": durations,
        }
    value, gradient = results["dense"]
    for name, (v, g) in results.items():
        np.testing.assert_allclose(v, value, rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(g, gradient, rtol=1e-7, atol=1e-6)
        row[name].update(
            abs_objective_error=abs(v - value),
            max_abs_gradient_error=float(np.max(np.abs(g - gradient))),
        )
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
