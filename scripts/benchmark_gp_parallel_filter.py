"""Research-only parallel grouped Kalman likelihood; no public backend changes.

Compare current dense grouped and sequential state-space objectives with a
parallel prefix implementation of Sarkka & Garcia-Fernandez (2021),
https://arxiv.org/abs/1905.13002. Messages use information-form observations,
so neither loading information nor process noise needs to be invertible.
The final score reuses the production stable residual/Joseph calculation.
This is a performance probe, not a qualified production backend or sampler.
The optional functional initialization reduces observation solves to K x K
and uses a Joseph covariance update. It improves speed/memory, but low-noise
gradient failures remain in the higher-factor qualification experiments.

Example (select an available GPU before launching):
  PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_gp_parallel_filter.py
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from multimodalsrm import BatemanSCR, Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.state_space import StateSpaceSystem
from multimodalsrm.bayesian.state_space_grouped import event_system, statistics, update


def sym(x):
    return (x + jnp.swapaxes(x, -1, -2)) * 0.5


def parallel_moments(A, Q, rows, D, h, *, observation_solve="full"):
    """Return filtering moments, including deterministic transitions at ties."""
    if observation_solve not in ("full", "functional"):
        raise ValueError("observation_solve must be full or functional")
    factors = D.shape[-1]
    dimension = A.shape[-1]
    eye = jnp.eye(dimension)

    def initialize(a, q, row, precision, information):
        output = jnp.kron(jnp.eye(factors), row[None])
        if observation_solve == "functional":
            # Woodbury in factor coordinates. No inverse of Q or precision,
            # so exact ties and rank-deficient loadings remain supported.
            cross = q @ output.T
            B = jnp.eye(factors) + precision @ output @ cross
            reduced_precision = sym(jnp.linalg.solve(B, precision))
            reduced_information = jnp.linalg.solve(B, information)
            gain = jnp.linalg.solve(B.T, cross.T).T
            residual = eye - gain @ precision @ output
            conditional_A = residual @ a
            conditional_C = sym(residual @ q @ residual.T + gain @ precision @ gain.T)
            conditional_b = cross @ reduced_information
            projected_A = output @ a
            J = sym(projected_A.T @ reduced_precision @ projected_A)
            eta = projected_A.T @ reduced_information
            return conditional_A, conditional_b, conditional_C, J, eta
        H = output.T @ precision @ output
        score = output.T @ information
        # All solves are I + products of PSD matrices, also when q == 0.
        B = eye + q @ H
        conditional_A = jnp.linalg.solve(B, a)
        conditional_C = sym(jnp.linalg.solve(B, q))
        conditional_b = conditional_C @ score
        J = sym(a.T @ jnp.linalg.solve(eye + H @ q, H @ a))
        eta = a.T @ jnp.linalg.solve(eye + H @ q, score)
        return conditional_A, conditional_b, conditional_C, J, eta

    # Stationary whitened initial state; first transition has zero elapsed time.
    A = A.at[0].set(jnp.zeros_like(eye))
    Q = Q.at[0].set(eye)
    messages = jax.vmap(initialize)(A, Q, rows, D, h)

    @jax.vmap
    def combine(left, right):
        a, b, c, J, eta = left
        aa, bb, cc, JJ, ee = right
        T = jnp.linalg.solve((eye + c @ JJ).T, aa.T).T
        U = jnp.linalg.solve(eye + c @ JJ, a).T
        return (
            T @ a,
            T @ (b + c @ ee) + bb,
            sym(T @ c @ aa.T + cc),
            sym(U @ JJ @ a + J),
            U @ (ee - JJ @ b) + eta,
        )

    _, mean, covariance, _, _ = jax.lax.associative_scan(combine, messages)
    return mean, covariance


def parallel_nll(problem, x, run, *, observation_solve="full"):
    nodes = problem.grouped_systems[run]
    model = problem.response_state_space
    if problem.dynamic_state_space:
        events, outputs = event_system(problem, x, run)
    else:
        events = problem.state_space_systems[run]
        outputs = jnp.asarray(model.outputs)
    order = events.order
    A, Q = jnp.asarray(events.transition), jnp.asarray(events.process_covariance)
    D, h, weights, residual, variance = statistics(problem, x, run)
    rows = outputs[nodes.modalities][order]
    D, h = D[order], h[order]
    means, covariances = parallel_moments(A, Q, rows, D, h, observation_solve=observation_solve)
    dimension = A.shape[-1]
    previous_m = jnp.concatenate((jnp.zeros((1, dimension)), means[:-1]))
    previous_P = jnp.concatenate((jnp.eye(dimension)[None], covariances[:-1]))
    # Re-evaluate local updates in parallel to retain the production score's
    # resistance to high-SNR cancellation. This duplicates some arithmetic.
    _, _, scores, projected, _, _ = jax.vmap(update)(previous_m, previous_P, A, Q, rows, D, h)
    projected = projected[jnp.argsort(order)]
    error = residual - jnp.sum(weights * projected[nodes.observation_nodes], axis=1)
    return 0.5 * (
        scores.sum() + jnp.sum(error**2 / variance + jnp.log(variance) + np.log(2 * np.pi))
    )


def dataset(args):
    rng = np.random.default_rng(args.seed)
    responses = {"ref": Response(Identity(), estimate=False, pooling="shared")}
    for modality, width, lag in (("aux", 0.7, 0.3), ("third", 1.1, 0.8)):
        kernel = {
            "identity": Identity(),
            "bateman": BatemanSCR(0.7, 2.0, lag),
            "gaussian": Gaussian(width, lag),
        }[args.response]
        responses[modality] = (
            Response.lag_only(kernel, pooling="shared", bounds={"lag": (-0.5, 1.5)})
            if args.learned and args.response != "identity"
            else Response(kernel, estimate=False, pooling="shared")
        )
    data = {}
    for subject in ("a", "b"):
        streams = {}
        for modality, dt in (("ref", 1.8), ("aux", 1.2), ("third", 1.5)):
            t = np.arange(0, args.duration, dt)
            if args.response != "identity":
                # Leave room for the declared response support before the
                # timed observation window (the initial marker is excluded).
                t = np.r_[0.0, 94.0 + t]
            # Shared clocks include exact/near ties; masks differ by feature.
            z = np.sin(t[:, None] / np.arange(2.0, 2.0 + args.factors))
            values = z @ rng.normal(size=(args.factors, args.channels))
            values += rng.normal(scale=0.4, size=values.shape)
            mask = rng.uniform(size=values.shape) > 0.05
            streams[modality] = TimeSeries(values, t, mask)
        data[subject] = {"run": streams}
    model = BayesianMultimodalSRM(
        features=args.factors,
        responses=responses,
        priors=BayesianPriors(
            noise=Prior.lognormal(np.log(0.3), 0.7),
            filters={m: {"lag": Prior.normal(0.5, 0.5)} for m in ("aux", "third")}
            if args.learned and args.response != "identity"
            else {},
        ),
        anchor=("a", "ref", 0),
        inference="map",
        linear_algebra="state_space",
        length_scale=3.0,
        max_observations=10_000_000,
    )
    _, state = _prepare(model, data)
    dense = None
    if not args.skip_dense:
        dense_model = copy.copy(model).set_params(linear_algebra="grouped")
        if args.response == "bateman":
            dense_model.set_params(response_quadrature_order=96)
        _, dense = _prepare(dense_model, data)
        assert state.names == dense.names
    point = state.initial.copy()
    for i, name in enumerate(state.names):
        if name[0] == "loading":
            point[i] = rng.normal(scale=0.7) if args.rank == "full" else 0.3
            if args.rank == "zero":
                point[i] = 0.0
        elif name[0] == "noise":
            point[i] = args.noise
    return state, dense, point


def measure(function, point, repeats):
    f = jax.jit(jax.value_and_grad(function))
    start = time.perf_counter()
    out = jax.block_until_ready(f(point))
    first = time.perf_counter() - start
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = jax.block_until_ready(f(point))
        durations.append(time.perf_counter() - start)
    value, gradient = float(out[0]), np.asarray(out[1])
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite value or gradient")
    return dict(
        first_seconds=first,
        median_seconds=float(np.median(durations)),
        repeats_seconds=durations,
        objective=value,
    ), gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response", choices=("identity", "bateman", "gaussian"), default="identity"
    )
    parser.add_argument("--duration", type=float, default=450)
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--rank", choices=("full", "one", "zero"), default="full")
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--scalar", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true; this benchmark requires float64")
    start = time.perf_counter()
    state, dense, point = dataset(args)
    preparation_seconds = time.perf_counter() - start
    functions = {
        "sequential": state.objective,
        "parallel": lambda x: (
            sum(parallel_nll(state, x, r) for r in state.systems) - state.log_prior(x)
        ),
    }
    if not args.skip_dense:
        functions["dense_grouped"] = dense.objective
    if args.scalar:
        scalar = copy.copy(state)
        scalar.grouped_state_space = False
        if not scalar.dynamic_state_space:
            scalar.state_space_systems = {
                r: StateSpaceSystem.prepare(
                    s.times - state.response_state_space.lags[state._packed[r][2]],
                    state.response_state_space,
                    state.features,
                )
                for r, s in state.systems.items()
            }
        functions["scalar"] = scalar.objective
    row = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        load=os.getloadavg(),
        environment={
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        preparation_seconds=preparation_seconds,
        observations=sum(len(s.times) for s in state.systems.values()),
        nodes=sum(len(s.times) for s in state.grouped_systems.values()),
        parameters=len(point),
        state_dimension=state.response_state_space.dimension * state.features,
        measurements={},
    )

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(row, indent=2) + "\n")

    print(
        json.dumps({"prepared": {k: v for k, v in row.items() if k != "measurements"}}), flush=True
    )
    save()
    reference_gradient = reference_value = None
    for name, function in functions.items():
        result, gradient = measure(function, jnp.asarray(point), args.repeats)
        if reference_gradient is None:
            reference_gradient, reference_value = gradient, result["objective"]
        result.update(
            value_absolute_difference=abs(result["objective"] - reference_value),
            gradient_max_absolute_difference=float(np.max(abs(gradient - reference_gradient))),
            gradient_relative_l2=float(
                np.linalg.norm(gradient - reference_gradient)
                / max(np.linalg.norm(reference_gradient), 1)
            ),
        )
        row["measurements"][name] = result
        save()
        print(json.dumps({"method": name, **result}), flush=True)
    print(json.dumps(row), flush=True)
    save()


if __name__ == "__main__":
    main()
