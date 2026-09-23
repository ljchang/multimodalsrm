"""Qualify repeated transition blocks and compact Joseph products on empirical data.

Same data, priors, deterministic stress points, and gates as the earlier prefix
qualification. Compiler memory estimates are not measured device peaks.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_gp_response_candidates import load_data
from gp_block_filter import interval_groups
from gp_block_filter import nll as block_nll
from gp_response_candidates import split_physiology
from gp_response_holdout import CANDIDATES, prepare, split_data
from investigate_gp_response_bounds import observation_summary
from qualify_gp_parallel_scaling import memory_estimate, qualification_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="gamma3")
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--source", default="empirical", choices=("empirical",))
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int)
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--joseph", choices=("original", "dense", "compact", "both"), default="both"
    )
    parser.add_argument("--transition-cache", type=int, default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled or args.features < 1 or args.repeats < 1:
        parser.error("float64 and positive features/repeats required")
    start = time.perf_counter()
    split = split_data(split_physiology(load_data(args)))
    _, problem = prepare(split, args.candidate, features=args.features)
    points = qualification_points(problem, True)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={
            p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
            for p in (Path(__file__).name, "gp_block_filter.py", "gp_response_holdout.py")
        },
        python=platform.python_version(),
        jax=jax.__version__,
        device=jax.devices()[0].device_kind,
        affinity=sorted(os.sched_getaffinity(0)),
        state_dimension=problem.features * problem.base.response_state_space.dimension,
        parameters=len(problem.names),
        training=observation_summary(problem.base),
        preparation_seconds=time.perf_counter() - start,
        tolerance=dict(
            objective_atol=1e-6, objective_rtol=1e-9, gradient_atol=1e-5, gradient_rtol=1e-7
        ),
        measurements={},
        points=[dict(name=name) for name, _ in points],
        status="prepared",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(status=None):
        if status:
            result["status"] = status
            print(json.dumps(dict(status=status)), flush=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def block(x, method):
        native = problem.native(x)
        prior = sum(jnp.sum(d.log_prob(x[i])) for i, _, d in problem.prior_groups)
        return (
            sum(
                block_nll(
                    problem.base, native, run, joseph=method, transition_cache=args.transition_cache
                )
                for run in problem.base.systems
            )
            - prior
        )

    if args.inspect_only:
        if args.transition_cache < 1:
            parser.error("--inspect-only requires a positive --transition-cache")
        for run, nodes in problem.base.grouped_systems.items():

            def count_groups(x):
                lag = problem.base.response_state_space.realize(
                    problem.native(x), problem.base.indices
                )[3]
                shifted = jnp.asarray(nodes.times) - lag[nodes.modalities]
                order = jnp.argsort(shifted, stable=True)
                return interval_groups(
                    jnp.asarray(nodes.times),
                    jnp.asarray(nodes.modalities),
                    order,
                    shifted[order],
                    args.transition_cache,
                )[2]

            evaluate = jax.jit(count_groups)
            for row, (_, point) in zip(result["points"], points):
                count = int(evaluate(jnp.asarray(point)))
                row.setdefault("interval_groups", {})[run] = dict(
                    groups=count,
                    capacity=args.transition_cache,
                    reuse=count <= args.transition_cache,
                )
        save("inspected")
        return

    functions = {"reference": problem.objective}
    for method in ("dense", "compact") if args.joseph == "both" else (args.joseph,):
        functions[method] = lambda x, method=method: block(x, method)
    references = []
    for name, function in functions.items():
        save(name + "_lowering")
        begun = time.perf_counter()
        lowered = jax.jit(jax.value_and_grad(function)).lower(jnp.asarray(points[0][1]))
        measurement = dict(lowering_seconds=time.perf_counter() - begun)
        result["measurements"][name] = measurement
        save(name + "_compiling")
        begun = time.perf_counter()
        compiled = lowered.compile()
        measurement.update(
            compile_seconds=time.perf_counter() - begun, compiler_memory=memory_estimate(compiled)
        )
        save(name + "_evaluating")
        for i, (row, (_, point)) in enumerate(zip(result["points"], points)):
            durations = []
            for _ in range(1 + args.repeats):
                begun = time.perf_counter()
                value, gradient = jax.block_until_ready(compiled(jnp.asarray(point)))
                durations.append(time.perf_counter() - begun)
            value, gradient = float(value), np.asarray(gradient)
            finite = bool(np.isfinite(value) and np.isfinite(gradient).all())
            detail = dict(
                finite=finite,
                first_seconds=durations[0],
                warm_seconds=durations[1:],
                warm_median_seconds=float(np.median(durations[1:])),
            )
            row[name] = detail
            if finite:
                detail.update(objective=value, gradient_max=float(np.max(abs(gradient))))
            if name == "reference":
                references.append((value, gradient))
            else:
                ref_value, ref_gradient = references[i]
                if finite and np.isfinite(ref_gradient).all() and np.isfinite(ref_value):
                    error, tol = abs(gradient - ref_gradient), result["tolerance"]
                    scaled = error / (
                        tol["gradient_atol"] + tol["gradient_rtol"] * abs(ref_gradient)
                    )
                    detail.update(
                        objective_absolute_difference=abs(value - ref_value),
                        gradient_max_absolute_difference=float(np.max(error)),
                        gradient_max_scaled_error=float(np.max(scaled)),
                        worst_scaled_parameter=problem.names[int(np.argmax(scaled))],
                        passed=bool(
                            abs(value - ref_value)
                            <= tol["objective_atol"] + tol["objective_rtol"] * abs(ref_value)
                            and np.max(scaled) <= 1
                        ),
                    )
                else:
                    detail["passed"] = False
                detail["speedup"] = (
                    row["reference"]["warm_median_seconds"] / detail["warm_median_seconds"]
                )
            save()
            print(json.dumps(dict(backend=name, point=row["name"], **detail)), flush=True)
    result["all_passed"] = {
        name: all(row[name]["passed"] for row in result["points"])
        for name in functions
        if name != "reference"
    }
    save("finished")


if __name__ == "__main__":
    main()
