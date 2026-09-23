"""Research qualification of prefix filtering on the common held-out training set.

No held-out predictions are evaluated. Accuracy gates and stress points are
declared before execution. Compiler memory is an estimate, not measured peak
GPU usage. Use one process per GPU, float64, and a bounded external timeout.
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
from benchmark_gp_parallel_filter import parallel_nll
from benchmark_gp_response_candidates import load_data
from gp_response_candidates import split_physiology
from gp_response_holdout import CANDIDATES, prepare, split_data
from investigate_gp_response_bounds import observation_summary

from multimodalsrm.bayesian.fitting import initial_points


def qualification_points(problem, stress):
    points = [(f"initial_{i}", x) for i, x in enumerate(initial_points(problem, 2, 722))]
    if not stress:
        return points
    origin = points[0][1]
    for side in (0, 1):
        x = origin.copy()
        for i, name in enumerate(problem.names):
            if name[0] == "filter":
                lo, hi = problem.bounds[i]
                x[i] = lo + (hi - lo) * (0.001 if side == 0 else 0.999)
        points.append((f"response_{'lower' if side == 0 else 'upper'}", x))
    for variance in (1e-3, 1e-6):
        x = origin.copy()
        for i, name in enumerate(problem.names):
            if name[0] == "noise":
                x[i] = variance
        points.append((f"noise_variance_{variance:g}", x))
    for rank in ("one", "zero"):
        x = origin.copy()
        for i, name in enumerate(problem.names):
            if name[0] == "loading":
                x[i] = 0.3 if rank == "one" else 0.0
        points.append((f"loading_rank_{rank}", x))
    return points


def memory_estimate(compiled):
    stats = compiled.memory_analysis()
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    result = {k: int(getattr(stats, k)) for k in fields}
    result["estimated_live_bytes"] = (
        result["argument_size_in_bytes"]
        + result["output_size_in_bytes"]
        + result["temp_size_in_bytes"]
        - result["alias_size_in_bytes"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="gamma3")
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--source", default="empirical", choices=("empirical",))
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int)
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--observation-solve", choices=("full", "functional"), default="full")
    parser.add_argument("--repeats", type=int, default=3)
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
    points = qualification_points(problem, args.stress)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={
            p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
            for p in (
                Path(__file__).name,
                "benchmark_gp_parallel_filter.py",
                "gp_response_holdout.py",
            )
        },
        python=platform.python_version(),
        jax=jax.__version__,
        device=jax.devices()[0].device_kind,
        affinity=sorted(os.sched_getaffinity(0)),
        state_dimension=problem.features * problem.base.response_state_space.dimension,
        parameters=len(problem.names),
        training=observation_summary(problem.base),
        preparation_seconds=time.perf_counter() - start,
        # Mixed tolerances are declared in advance. Absolute gradients alone
        # are also reported: low-noise stress points can have enormous scores.
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

    def parallel(x):
        native = problem.native(x)
        prior = sum(jnp.sum(d.log_prob(x[i])) for i, _, d in problem.prior_groups)
        return (
            sum(
                parallel_nll(problem.base, native, run, observation_solve=args.observation_solve)
                for run in problem.base.systems
            )
            - prior
        )

    references = []
    for name, function in (("sequential", problem.objective), ("parallel", parallel)):
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
        for row, (_, point) in zip(result["points"], points):
            durations = []
            for _ in range(1 + args.repeats):
                begun = time.perf_counter()
                value, gradient = jax.block_until_ready(compiled(jnp.asarray(point)))
                durations.append(time.perf_counter() - begun)
            value, gradient = float(value), np.asarray(gradient)
            finite = bool(np.isfinite(value) and np.isfinite(gradient).all())
            row[name] = dict(
                finite=finite,
                first_seconds=durations[0],
                warm_seconds=durations[1:],
                warm_median_seconds=float(np.median(durations[1:])),
            )
            if finite:
                row[name].update(objective=value, gradient_max=float(np.max(abs(gradient))))
            if name == "sequential":
                references.append((value, gradient))
            else:
                ref_value, ref_gradient = references.pop(0)
                if finite and np.isfinite(ref_gradient).all() and np.isfinite(ref_value):
                    error = abs(gradient - ref_gradient)
                    tol = result["tolerance"]
                    row.update(
                        objective_absolute_difference=abs(value - ref_value),
                        gradient_max_absolute_difference=float(np.max(error)),
                        gradient_max_scaled_error=float(
                            np.max(
                                error
                                / (tol["gradient_atol"] + tol["gradient_rtol"] * abs(ref_gradient))
                            )
                        ),
                        passed=bool(
                            abs(value - ref_value)
                            <= tol["objective_atol"] + tol["objective_rtol"] * abs(ref_value)
                            and np.all(
                                error
                                <= tol["gradient_atol"] + tol["gradient_rtol"] * abs(ref_gradient)
                            )
                        ),
                    )
                else:
                    row["passed"] = False
                row["speedup"] = (
                    row["sequential"]["warm_median_seconds"]
                    / row["parallel"]["warm_median_seconds"]
                )
            save()
            print(json.dumps(dict(backend=name, **row)), flush=True)
    result["all_passed"] = all(row["passed"] for row in result["points"])
    save("finished")


if __name__ == "__main__":
    main()
