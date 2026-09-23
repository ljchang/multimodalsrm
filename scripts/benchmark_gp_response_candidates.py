"""Prepare, evaluate, or fit the proposed modality-specific GP responses.

Example (no participant data required):
    PYTHONPATH=src JAX_ENABLE_X64=true python scripts/benchmark_gp_response_candidates.py \
        --action evaluate --output /tmp/response-probe.json

Use --source empirical to reuse the ignored local EmotionPictures loader.
The Gaussian comparator changes only face/rating families; brain remains
double-gamma and EDA remains Bateman. Different supports can select different
edge observations, so separate training objectives are not a model ranking.
"""

import argparse
import importlib.util
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gp_response_candidates import candidate_model, split_physiology

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.fitting import initial_points, search
from multimodalsrm.bayesian.persistence import _prepare


def synthetic_data():
    """Small masked asynchronous fixture for numerical checks, not recovery."""
    rng = np.random.default_rng(722)
    data = {}
    for subject in ("a", "b"):
        streams = {}
        for index, (name, channels) in enumerate(
            (("brain", 3), ("face", 2), ("rating", 1), ("physio", 3))
        ):
            times = np.r_[0.0, [104.0, 108.0, 113.0, 118.0]] + 0.17 * index
            values = rng.normal(size=(len(times), channels))
            mask = np.ones(values.shape, dtype=bool)
            mask[2, -1] = False
            streams[name] = TimeSeries(values, times, mask)
        data[subject] = {"run0": streams}
    return data


def load_data(args):
    if args.source == "synthetic":
        return synthetic_data()
    path = args.loader.resolve()
    spec = importlib.util.spec_from_file_location("response_candidate_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This research adapter depends on the loader's documented column order.
    if tuple(module.PHYSIO_COLUMNS) != ("EDA - EDA100C-MRI", "Pulse Rate", "Respiration Rate"):
        raise ValueError("unexpected physiology column order in local loader")
    return module.load(
        subjects=tuple(args.subjects.split(",")) if args.subjects else module.SUBJECTS,
        parcels=args.parcels,
        window=args.window,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "empirical"), default="synthetic")
    parser.add_argument("--action", choices=("prepare", "evaluate", "fit"), default="prepare")
    parser.add_argument("--family", choices=("gamma", "gaussian"), default="gamma")
    parser.add_argument("--algebra", choices=("state_space", "grouped"), default="state_space")
    parser.add_argument("--face-shape", type=int, default=3)
    parser.add_argument("--rating-shape", type=int, default=3)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument(
        "--other-physio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also include pulse and respiration (default: brain, face, rating, EDA only)",
    )
    parser.add_argument("--subjects")
    parser.add_argument("--parcels", type=int)
    parser.add_argument("--window", type=float)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("JAX_ENABLE_X64=true is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = split_physiology(load_data(args), other_physio=args.other_physio)
    model = candidate_model(
        data,
        features=args.features,
        family=args.family,
        face_shape=args.face_shape,
        rating_shape=args.rating_shape,
        other_physio=args.other_physio,
        algebra=args.algebra,
        maxiter=args.maxiter,
        starts=args.starts,
    )
    _, problem = _prepare(model, data)
    row = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        preparation_seconds=time.perf_counter() - started,
        parameters=len(problem.names),
        parameter_names=[list(name) for name in problem.names],
        observations=sum(len(s.times) for s in problem.systems.values()),
        nodes=sum(len(s.times) for s in problem.grouped_systems.values()),
        reference=problem.reference_convention,
        responses={
            m: dict(
                **r.initial_kernel().metadata,
                free=list(r.free_parameters),
                bounds=r.parameter_bounds(),
            )
            for m, r in model.responses.items()
        },
        priors=asdict(model.priors),
        covariance_tolerance=model.covariance_tolerance,
        status="prepared",
    )
    if args.algebra == "state_space":
        row.update(
            state_dimension=problem.features * problem.response_state_space.dimension,
            covariance_error_bound=problem.response_state_space.tail_bound,
        )

    def save():
        args.output.write_text(json.dumps(row, indent=2) + "\n")

    save()
    print(
        json.dumps({k: row[k] for k in ("status", "parameters", "observations", "nodes")}),
        flush=True,
    )
    if args.action == "evaluate":
        point = initial_points(problem, 1, 722)[0]
        evaluate = jax.jit(jax.value_and_grad(problem.objective))
        row["status"] = "compiling_and_evaluating"
        save()
        durations = []
        for _ in range(1 + args.repeats):
            start = time.perf_counter()
            value, gradient = jax.block_until_ready(evaluate(jnp.asarray(point)))
            durations.append(time.perf_counter() - start)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("nonfinite candidate objective or gradient")
        row["evaluation"] = dict(
            first_seconds=durations[0],
            warm_seconds=durations[1:],
            warm_median_seconds=float(np.median(durations[1:])),
            objective=float(value),
            gradient_norm=float(np.linalg.norm(gradient)),
        )
        np.savez(args.output.with_suffix(".npz"), parameters=point, gradient=np.asarray(gradient))
        row["status"] = "evaluated"
    elif args.action == "fit":
        start = time.perf_counter()

        def progress(event):
            row["status"] = "fitting"
            row["last_progress"] = event
            save()
            print(
                json.dumps(
                    {
                        "phase": event["phase"],
                        "record": {
                            k: v
                            for k, v in event["record"].items()
                            if k not in ("parameters", "initial_parameters")
                        },
                    }
                ),
                flush=True,
            )

        best, records = search(problem, model.search, model.random_state, progress=progress)
        row["fit_seconds"] = time.perf_counter() - start
        row["fit"] = {k: v for k, v in best.items() if k != "parameters"}
        row["filter_parameters"] = {
            "/".join(name[1:]): float(best["parameters"][i])
            for i, name in enumerate(problem.names)
            if name[0] == "filter"
        }
        row["status"] = "fit_finished"
        row["restart_count"] = len(records)
        np.savez(args.output.with_suffix(".npz"), parameters=best["parameters"])
    save()
    print(json.dumps({"status": row["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
