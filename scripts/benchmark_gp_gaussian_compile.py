"""Research probe: separate Gaussian tracing, compilation, and warm execution."""

import argparse
import copy
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_gp_parallel_filter import dataset, parallel_nll

from multimodalsrm.bayesian.state_space import StateSpaceSystem

parser = argparse.ArgumentParser()
parser.add_argument("--closed", action="store_true")
parser.add_argument("--duration", type=float, default=450)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
result = dict(
    device=str(jax.devices()[0]),
    device_kind=jax.devices()[0].device_kind,
    jax=jax.__version__,
    mode="closed" if args.closed else "operands",
    stages={},
)


def record(stage, start, **extra):
    result["stages"][stage] = dict(seconds=time.perf_counter() - start, **extra)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({stage: result["stages"][stage]}), flush=True)


start = time.perf_counter()
state, _, point = dataset(
    argparse.Namespace(
        response="gaussian",
        seed=722,
        factors=3,
        channels=32,
        duration=args.duration,
        learned=False,
        skip_dense=True,
        rank="full",
        noise=0.3,
    )
)
runs = tuple(state.systems)
coefficients = tuple(
    (
        jnp.asarray(state.state_space_systems[r].transition),
        jnp.asarray(state.state_space_systems[r].process_covariance),
    )
    for r in runs
)
x = jnp.asarray(point)
jax.block_until_ready((x, coefficients))
record(
    "prepare",
    start,
    nodes=sum(len(s.times) for s in state.grouped_systems.values()),
    dimension=state.response_state_space.dimension * state.features,
)


def objective(x, coefficients):
    local = copy.copy(state)
    local.state_space_systems = {
        r: StateSpaceSystem(
            state.state_space_systems[r].order, a, q, state.state_space_systems[r].times
        )
        for r, (a, q) in zip(runs, coefficients)
    }
    return sum(parallel_nll(local, x, r) for r in runs) - state.log_prior(x)


if args.closed:
    function = jax.jit(jax.value_and_grad(lambda x: objective(x, coefficients)))
    operands = (x,)
else:
    function = jax.jit(jax.value_and_grad(objective, argnums=0))
    operands = (x, coefficients)
start = time.perf_counter()
lowered = function.lower(*operands)
record("lower", start)
start = time.perf_counter()
compiled = lowered.compile()
record("compile", start)
times = []
for i in range(4):
    start = time.perf_counter()
    value, gradient = jax.block_until_ready(compiled(*operands))
    times.append(time.perf_counter() - start)
    record("execute_" + str(i), start, value=float(value), finite=bool(np.isfinite(gradient).all()))
np.savez(args.output.with_suffix(".npz"), value=np.asarray(value), gradient=np.asarray(gradient))
result["warm_median_seconds"] = float(np.median(times[1:]))
args.output.write_text(json.dumps(result, indent=2) + "\n")
