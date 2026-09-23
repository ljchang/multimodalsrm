"""Qualify the research parallel filter on the held-out comparison likelihood."""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="gamma3")
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--source", default="empirical")
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int, default=5)
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    split = split_data(split_physiology(load_data(args)))
    _, problem = prepare(split, args.candidate)
    points = initial_points(problem, 2, 722)
    if args.fit:
        saved = json.loads(args.fit.read_text())
        assert saved["parameter_names"] == [list(n) for n in problem.names]
        assert saved["training"] == observation_summary(problem.base)
        points = np.vstack((points, saved["best"]["parameters"]))
    result = dict(
        candidate=args.candidate,
        device=jax.devices()[0].device_kind,
        state_dimension=problem.base.response_state_space.dimension,
        training=observation_summary(problem.base),
        points=[],
        status="prepared",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def parallel(x):
        native = problem.native(x)
        prior = sum(jnp.sum(d.log_prob(x[indices])) for indices, _, d in problem.prior_groups)
        return sum(parallel_nll(problem.base, native, run) for run in problem.base.systems) - prior

    for name, function in (("sequential", problem.objective), ("parallel", parallel)):
        start = time.perf_counter()
        evaluate = jax.jit(jax.value_and_grad(function))
        result["status"] = name + "_lowering"
        save()
        lowered = evaluate.lower(jnp.asarray(points[0]))
        result[name + "_lowering_seconds"] = time.perf_counter() - start
        result["status"] = name + "_compiling"
        save()
        print(json.dumps(dict(status=result["status"])), flush=True)
        start = time.perf_counter()
        compiled = lowered.compile()
        result[name + "_compile_seconds"] = time.perf_counter() - start
        for index, point in enumerate(points):
            duration = []
            for _ in range(4):
                start = time.perf_counter()
                value, gradient = jax.block_until_ready(compiled(jnp.asarray(point)))
                duration.append(time.perf_counter() - start)
            if name == "sequential":
                result["points"].append(
                    dict(
                        objective=float(value),
                        gradient=np.asarray(gradient).tolist(),
                        sequential_warm_seconds=duration[1:],
                    )
                )
            else:
                row = result["points"][index]
                row.update(
                    parallel_warm_seconds=duration[1:],
                    objective_absolute_difference=abs(float(value) - row["objective"]),
                    gradient_max_absolute_difference=float(
                        np.max(abs(np.asarray(gradient) - row["gradient"]))
                    ),
                )
                assert np.isfinite(value) and np.isfinite(gradient).all()
            save()
    result["status"] = "finished"
    save()
    print(json.dumps({k: v for k, v in result.items() if k != "points"}), flush=True)
    for row in result["points"]:
        print(json.dumps({k: v for k, v in row.items() if k != "gradient"}), flush=True)


if __name__ == "__main__":
    main()
