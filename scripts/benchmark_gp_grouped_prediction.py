"""Compare grouped prediction before/after feature and covariance reuse.

Reports whole calls, including any compilation, factorization, device transfer,
and conversion to NumPy. The baseline comes from a recorded repository commit.
Only aggregate errors/timings are written; empirical fits stay in local_data.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import types
from pathlib import Path

import jax
import numpy as np
from gp_response_holdout import make_model
from investigate_gp_response_bounds import observation_summary
from probe_gp_correlated_noise import TIMESCALES
from refit_gp_temporal_calibration import prepare

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.grouped_prediction import project
from multimodalsrm.bayesian.persistence import _prepare


def synthetic(args):
    rng = np.random.default_rng(837)
    data = {}
    for subject in ("a", "b"):
        streams = {}
        for i, (modality, channels) in enumerate(
            (("brain", args.channels), ("face", 6), ("rating", 2), ("eda", 1))
        ):
            times = np.r_[0, np.linspace(100, 400, 40), 500] + i * 0.13
            values = rng.normal(size=(len(times), channels))
            streams[modality] = TimeSeries(values, times, rng.uniform(size=values.shape) > 0.1)
        data[subject] = {"run0": streams}
    model = make_model(data, args.candidate, "grouped", args.order, features=args.factors)
    model.set_params(noise_timescales=TIMESCALES if args.noise == "ou" else {})
    _, problem = _prepare(model, data)
    x = problem.initial.copy()
    for i, name in enumerate(problem.names):
        if name[0] == "loading":
            x[i] = rng.normal(scale=0.5)
        elif name[0] == "noise":
            x[i] = 0.4
    return problem, x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit", type=Path, help="Private empirical refit JSON, otherwise synthetic"
    )
    parser.add_argument("--candidate", choices=("gamma3", "gaussian"), default="gaussian")
    parser.add_argument("--noise", choices=("independent", "ou"), default="ou")
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--order", type=int, default=384)
    parser.add_argument("--modality", choices=("brain", "face", "rating", "eda"), default="brain")
    parser.add_argument("--query-features", type=int, default=8)
    parser.add_argument("--query-times", type=int, default=16)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--include-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-commit", default="16e6574007018bf140b4ddad25b2b8e2d7b02880")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    if any(
        getattr(args, k) < 1
        for k in ("factors", "channels", "query_features", "query_times", "draws", "repeats")
    ):
        parser.error("Counts must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.reference_commit):
        parser.error("reference-commit must be a complete commit SHA")
    if args.output.exists():
        parser.error("Refusing to overwrite benchmark results")
    source = subprocess.check_output(
        ["git", "show", f"{args.reference_commit}:src/multimodalsrm/bayesian/prediction.py"],
        text=True,
    )
    reference = types.ModuleType("multimodalsrm.bayesian._prediction_benchmark_reference")
    reference.__package__ = "multimodalsrm.bayesian"
    exec(compile(source, f"prediction@{args.reference_commit}", "exec"), reference.__dict__)
    if args.fit:
        fit = json.loads(args.fit.read_text())
        settings = argparse.Namespace(**fit["arguments"])
        settings.loader = Path(settings.loader)
        _, _, canonical = prepare(settings, order=args.order)
        problem = canonical.base
        assert observation_summary(problem) == fit["training"]
        x = np.asarray(canonical.native(np.asarray(fit["best"]["parameters"])))
    else:
        problem, x = synthetic(args)
    keys = [k for k in problem.keys if k[1] == args.modality][: args.query_features]
    if not keys:
        parser.error("No features of the requested modality")
    draws = np.tile(x, (args.draws, 1))
    query = np.linspace(260, 280, args.query_times, endpoint=False)

    def before():
        components = [
            reference.project(
                problem, draws, "run0", query, key=key, include_noise=args.include_noise
            )
            for key in keys
        ]
        return tuple(np.stack([c[i] for c in components], axis=-1) for i in (0, 1))

    def after():
        return project(problem, draws, "run0", query, keys=keys, include_noise=args.include_noise)

    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            name: hashlib.sha256(Path("src/multimodalsrm/bayesian", name).read_bytes()).hexdigest()
            for name in ("prediction.py", "grouped_prediction.py")
        },
        reference_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        absolute_moment_tolerance=1e-7,
        python=platform.python_version(),
        jax=jax.__version__,
        device=jax.devices()[0].device_kind,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        training=observation_summary(problem),
        factors=problem.features,
        output_features=len(keys),
        parameter_sha256=hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest(),
        timing_scope="Whole marginal prediction calls including any compilation and host/device transfers; repeats create fresh per-call caches; repeated MAP points, not posterior draws",
        timings={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    outputs = {}
    for name, evaluate in (("before", before), ("after", after)):
        jax.clear_caches()
        seconds = []
        repeat_errors = []
        result["status"] = f"running_{name}"
        save()
        for _ in range(1 + args.repeats):
            started = time.perf_counter()
            output = evaluate()
            seconds.append(time.perf_counter() - started)
            assert all(np.isfinite(a).all() for a in output)
            if name in outputs:
                error = max(float(np.max(abs(a - b))) for a, b in zip(output, outputs[name]))
                repeat_errors.append(error)
                # GPU scatter/reduction order need not be bitwise deterministic.
                # Apply the same absolute moment gate as before/after agreement.
                assert error <= result["absolute_moment_tolerance"], error
            outputs[name] = output
            print(json.dumps(dict(implementation=name, seconds=seconds[-1])), flush=True)
        result["timings"][name] = dict(
            first_seconds=seconds[0],
            repeated_call_seconds=seconds[1:],
            repeat_max_absolute_errors=repeat_errors,
        )
        save()
    errors = [float(np.max(abs(a - b))) for a, b in zip(outputs["before"], outputs["after"])]
    assert max(errors) <= result["absolute_moment_tolerance"], errors
    result.update(
        status="finished",
        mean_max_absolute_error=errors[0],
        variance_max_absolute_error=errors[1],
        first_call_speedup=result["timings"]["before"]["first_seconds"]
        / result["timings"]["after"]["first_seconds"],
        repeated_call_speedup=float(
            np.median(result["timings"]["before"]["repeated_call_seconds"])
            / np.median(result["timings"]["after"]["repeated_call_seconds"])
        ),
    )
    save()
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "first_call_speedup",
                    "repeated_call_speedup",
                    "mean_max_absolute_error",
                    "variance_max_absolute_error",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
