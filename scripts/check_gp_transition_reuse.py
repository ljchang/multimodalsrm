"""Check interval reuse derivatives, clock ties, and capacity fallback."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from gp_block_filter import interval_groups

from multimodalsrm.bayesian.state_space_parameters import parameter_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("float64 required")
    model = SimpleNamespace(
        dimension=2, minimum_rate=0.1, maximum_rate=3.0, generator_norm_bound=10.0
    )
    times = np.tile(np.arange(40, dtype=float), 3)
    times[41] = np.nextafter(times[41], np.inf)
    mods = np.repeat(np.arange(3), 40)
    times, mods = jnp.asarray(times), jnp.asarray(mods)
    weights = jnp.asarray(np.random.default_rng(455).normal(size=(2, len(times), 2, 2)))

    def evaluate(x, capacity=0, diagnostic=False):
        rate = jnp.exp(x[0])
        F = jnp.array([[-rate, 0.0], [-2 * rate, -rate]])
        b = jnp.full(2, jnp.sqrt(2 * rate))
        lag = jnp.array([0.0, x[1], x[2] + x[3] ** 2])
        shifted = times - lag[mods]
        order = jnp.argsort(shifted, stable=True)
        shifted = shifted[order]
        delta = jnp.diff(shifted, prepend=shifted[0])
        if capacity:
            first, inverse, count = interval_groups(times, mods, order, shifted, capacity)

            def reuse(_):
                A, Q = parameter_transitions(model, F, b, delta[first], 1)
                return A[inverse], Q[inverse]

            A, Q = jax.lax.cond(
                count <= capacity,
                reuse,
                lambda _: parameter_transitions(model, F, b, delta, 1),
                None,
            )
        else:
            A, Q = parameter_transitions(model, F, b, delta, 1)
            count = len(times)
        if diagnostic:
            return count
        return jnp.sum(weights * jnp.stack((A, Q)) ** 2)

    functions = {c: jax.jit(jax.value_and_grad(lambda x, c=c: evaluate(x, c))) for c in (0, 1, 64)}
    result = dict(status="running", cases=[])
    for x in (
        np.array([-0.3, 0.13, -0.31, 0.2]),
        np.zeros(4),
        np.array([0.2, 1e-10, -1e-10, 0.0]),
        np.array([0.0, 8.0, -12.0, 0.3]),
    ):
        ref, grad = functions[0](x)
        for capacity in (1, 64):
            actual, g = functions[capacity](x)
            count = int(jax.jit(lambda y: evaluate(y, capacity, True))(x))
            np.testing.assert_allclose(actual, ref, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(g, grad, atol=1e-10, rtol=1e-10)
            result["cases"].append(
                dict(
                    parameters=x.tolist(),
                    capacity=capacity,
                    groups=count,
                    fallback=count > capacity,
                    objective_absolute_difference=float(abs(actual - ref)),
                    gradient_max_absolute_difference=float(np.max(abs(g - grad))),
                )
            )
    x = np.array([-0.3, 0.13, -0.31, 0.2])
    grad = functions[64](x)[1]
    fd = []
    for i in range(4):
        dx = np.eye(4)[i] * 1e-5
        value = (functions[64](x + dx)[0] - functions[64](x - dx)[0]) / 2e-5
        np.testing.assert_allclose(value, grad[i], atol=1e-6, rtol=1e-6)
        fd.append(float(abs(value - grad[i])))
    assert any(c["fallback"] for c in result["cases"])
    assert any(not c["fallback"] for c in result["cases"])
    result.update(status="finished", finite_difference_errors=fd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
