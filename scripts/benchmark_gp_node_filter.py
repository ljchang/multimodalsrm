"""Bounded synthetic research probe: exact node compression before Kalman filtering.

This is not a supported inference backend. It handles fixed responses and positive
white observation noise, uses covariance subtraction (not a production square-root
filter), and compares likelihoods and all physical gradients with current backends.
No optimizer or posterior speed is claimed. Run with the optional Bayesian runtime:

PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 \
    python scripts/benchmark_gp_node_filter.py --times 128 --channels 32
"""

import argparse
import json
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianPriors, Prior
from multimodalsrm.bayesian.grouped import GroupedSystem
from multimodalsrm.bayesian.observation_adapter import BayesianObservationAdapter
from multimodalsrm.bayesian.problem import BayesianProblem
from multimodalsrm.bayesian.state_space import StateSpaceSystem
from multimodalsrm.bayesian.state_space_responses import ResponseStateSpace


def node_objective(problem):
    """Integrate exact per-node Gaussian information, including normalization.

    The prototype takes a grouped problem to avoid allocating the old scalar
    backend's transitions. Nodes use exact modality/time equality, without bins.
    Singular loading information is allowed; it is never inverted.
    """
    if (
        any(response.free_parameters for response in problem.responses.values())
        or problem.length_scale_prior is not None
        or problem.noise_timescales
        or problem.run_baseline_sd
        or problem.response_quadrature_order is not None
    ):
        raise ValueError(
            "node probe requires fixed responses/timescale and white observation noise"
        )
    model = ResponseStateSpace.prepare(
        problem.responses, problem.length_scale, problem.adapter.covariance_tolerance
    )
    factors = problem.features
    dimension = factors * model.dimension
    prepared = []
    for run, system in problem.systems.items():
        ki, gi, mi = problem._packed[run]
        nodes = GroupedSystem.prepare(system.times, mi, covariance_lookup=False)
        events = StateSpaceSystem.prepare(
            nodes.times - model.lags[nodes.modalities], model, factors
        )
        outputs = np.stack(
            [np.kron(np.eye(factors), model.outputs[m][None, :]) for m in nodes.modalities]
        )
        prepared.append((system, ki, gi, nodes, events, jnp.asarray(outputs)))

    def objective(x):
        weights, offsets, noise, _, _ = problem.arrays(x)
        weights = weights.reshape(len(problem.keys), factors)
        total = -problem.log_prior(x)
        for system, ki, gi, nodes, events, outputs in prepared:
            w = weights[ki]
            v = noise[gi]
            r = jnp.asarray(system.values) - offsets[ki]
            count = len(nodes.times)
            index = jnp.asarray(nodes.observation_nodes)
            D = jax.ops.segment_sum(w[:, :, None] * w[:, None, :] / v[:, None, None], index, count)
            h = jax.ops.segment_sum(w * (r / v)[:, None], index, count)
            c = jax.ops.segment_sum(r * r / v + jnp.log(2 * np.pi * v), index, count)

            def step(carry, inputs):
                mean, covariance, score = carry
                A, Q, C, precision, information, constant = inputs
                mean = A @ mean
                covariance = A @ covariance @ A.T + Q
                cross = covariance @ C.T
                V = C @ cross
                mu = C @ mean
                u = information - precision @ mu
                B = jnp.eye(factors) + precision @ V
                solved = jnp.linalg.solve(B, jnp.concatenate((u[:, None], precision), axis=1))
                shift, reduction = solved[:, 0], solved[:, 1:]
                increment = 0.5 * (
                    constant
                    - 2 * information @ mu
                    + mu @ precision @ mu
                    + jnp.linalg.slogdet(B)[1]
                    - u @ V @ shift
                )
                mean = mean + cross @ shift
                covariance = covariance - cross @ reduction @ cross.T
                covariance = (covariance + covariance.T) * 0.5
                return (mean, covariance, score + increment), None

            order = events.order
            final, _ = jax.lax.scan(
                jax.checkpoint(step, prevent_cse=False),
                (jnp.zeros(dimension), jnp.eye(dimension), jnp.asarray(0.0)),
                (
                    jnp.asarray(events.transition),
                    jnp.asarray(events.process_covariance),
                    outputs[order],
                    D[order],
                    h[order],
                    c[order],
                ),
            )
            total = total + final[2]
        return total

    metadata = {
        "state_dimension": dimension,
        "nodes": sum(len(p[3].times) for p in prepared),
        "node_transition_bytes": sum(
            p[4].transition.nbytes + p[4].process_covariance.nbytes for p in prepared
        ),
        "response_covariance_error_bound": model.tail_bound,
    }
    return objective, metadata


def measure(fun, x, repeats):
    start = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(fun)).lower(x).compile()
    value, gradient = jax.block_until_ready(compiled(x))
    cold = time.perf_counter() - start
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(compiled(x))
        durations.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    return (
        {
            "compile_and_first_s": cold,
            "median_value_gradient_s": float(np.median(durations)),
            "warm_repeats_s": durations,
            "compiled_temp_bytes": memory.temp_size_in_bytes,
            "compiled_argument_bytes": memory.argument_size_in_bytes,
            "compiled_output_bytes": memory.output_size_in_bytes,
        },
        float(value),
        np.asarray(gradient),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--zero-loadings", action="store_true")
    parser.add_argument("--noise", type=float, default=0.7)
    parser.add_argument("--skip-scalar", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not jax.config.x64_enabled or args.factors < 2:
        raise ValueError("this probe requires float64 and at least two factors")
    rng = np.random.default_rng(721)
    # Irregular clocks; a second modality has a different clock and response.
    times = np.cumsum(rng.uniform(0.5, 1.5, args.times))
    responses = {"a": Response(Identity(), estimate=False, pooling="shared")}
    if args.gaussian:
        responses["b"] = Response(Gaussian(width=0.6, lag=0.3), estimate=False, pooling="shared")
    data = {}
    for participant in ("p0", "p1"):
        streams = {}
        for modality in responses:
            t = times if modality == "a" else times + 0.27
            values = rng.normal(size=(args.times, args.channels))
            mask = rng.uniform(size=values.shape) > 0.15 if args.mask else None
            streams[modality] = TimeSeries(values, t, mask)
        data[participant] = {"r": streams}
    adapter = BayesianObservationAdapter(
        features=args.factors,
        latent_dt=1.0,
        responses=responses,
        standardize=False,
        length_scale=3.0,
        max_observations=2_000_000,
    )
    adapter._prepare(data)
    priors = BayesianPriors(noise=Prior.lognormal(-0.3, 0.7))
    grouped = BayesianProblem(adapter, priors, linear_algebra="grouped")
    x = grouped.initial.copy()
    for i, name in enumerate(grouped.names):
        if name[0] == "loading":
            x[i] = 0.0 if args.zero_loadings else rng.normal(scale=0.5)
        elif name[0] == "offset":
            x[i] = rng.normal(scale=0.2)
        elif name[0] == "noise":
            x[i] = args.noise
    if args.factors == 1 and args.zero_loadings:
        raise ValueError("zero-loadings probe uses multifactor unconstrained loadings")
    x = jnp.asarray(x)
    fun, metadata = node_objective(grouped)
    row = {
        "config": vars(args),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "package_source": __import__("multimodalsrm").__file__,
        "devices": [str(d) for d in jax.devices()],
        "x64": jax.config.x64_enabled,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "load1": os.getloadavg()[0],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "parameters": len(x),
        "observations": sum(len(s.times) for s in grouped.systems.values()),
        **metadata,
    }
    results = {}
    functions = [("dense_grouped", grouped.objective), ("node_filter", fun)]
    if not args.skip_scalar:
        scalar = BayesianProblem(adapter, priors, linear_algebra="state_space")
        assert scalar.names == grouped.names
        row["scalar_transition_bytes"] = sum(
            s.transition.nbytes + s.process_covariance.nbytes
            for s in scalar.state_space_systems.values()
        )
        functions.append(("scalar_filter", scalar.objective))
    for label, function in functions:
        metrics, value, gradient = measure(function, x, args.repeats)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"nonfinite result: {label}")
        results[label] = (value, gradient)
        row[label] = metrics
    ref_value, ref_gradient = results["dense_grouped"]
    for label, (value, gradient) in results.items():
        row[label].update(
            objective=value,
            abs_objective_error=abs(value - ref_value),
            max_abs_gradient_error=float(np.max(np.abs(gradient - ref_gradient))),
            relative_gradient_l2_error=float(
                np.linalg.norm(gradient - ref_gradient) / max(1.0, np.linalg.norm(ref_gradient))
            ),
        )
        if not args.gaussian:
            np.testing.assert_allclose(value, ref_value, atol=1e-7, rtol=1e-10)
            np.testing.assert_allclose(gradient, ref_gradient, atol=1e-6, rtol=1e-7)
    if "scalar_filter" in results:
        a, ga = results["scalar_filter"]
        b, gb = results["node_filter"]
        row["node_vs_scalar"] = {
            "abs_objective_error": abs(a - b),
            "max_abs_gradient_error": float(np.max(np.abs(ga - gb))),
        }
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-10)
        np.testing.assert_allclose(ga, gb, atol=1e-6, rtol=1e-7)
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
