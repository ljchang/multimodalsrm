"""Research MAP baseline using physical boxes and log noise variances.

This changes optimizer coordinates only, with no transform Jacobian in MAP.
Production inference is unchanged. Reuses the four-modality candidate model,
its physical-gradient qualification, and independent prior-quantile starts.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from benchmark_gp_response_candidates import load_data
from gp_response_candidates import candidate_model, split_physiology
from scipy.optimize import minimize

from multimodalsrm.bayesian.fitting import _map_diagnostics, initial_points
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.polishing import polish


def coordinate_map(problem):
    """Identity coordinates except exp(log variance); preserve all prior bounds."""
    noise = np.array([i for i, n in enumerate(problem.names) if n[0] == "noise"])
    bounds = np.asarray(problem.bounds, float).copy()
    with np.errstate(divide="ignore"):
        bounds[noise] = np.log(bounds[noise])

    def encode(x):
        z = np.asarray(x).copy()
        z[noise] = np.log(z[noise])
        return z

    def decode(z):
        x = np.asarray(z).copy()
        x[noise] = np.exp(x[noise])
        return x

    def evaluate(z):
        x = decode(z)
        if not np.isfinite(x).all() or np.any(x[noise] <= 0):
            return np.inf, np.zeros_like(z)
        value, gradient = problem.value_gradient(x)
        gradient = np.asarray(gradient).copy()
        gradient[noise] *= x[noise]
        return float(value), gradient

    return encode, decode, evaluate, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("empirical", "synthetic"), default="empirical")
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int, default=5)
    parser.add_argument("--window", type=float, default=180)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--polish-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    started = time.perf_counter()
    data = split_physiology(load_data(args))
    model = candidate_model(data, features=args.features, maxiter=args.maxiter, starts=args.starts)
    _, problem = _prepare(model, data)
    points = initial_points(problem, args.starts, args.seed)
    encode, decode, evaluate, bounds = coordinate_map(problem)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        parameters=len(problem.names),
        parameter_names=[list(n) for n in problem.names],
        observations=sum(len(s.times) for s in problem.systems.values()),
        nodes=sum(len(s.times) for s in problem.grouped_systems.values()),
        state_dimension=problem.features * problem.response_state_space.dimension,
        reference=problem.reference_convention,
        physical_gradient_tolerance=model.search.physical_gradient_tolerance,
        coordinates="physical loadings/offsets/filters; log noise variance; no Jacobian",
        preparation_seconds=time.perf_counter() - started,
        records=[],
        status="prepared",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def summary(record):
        return {
            k: record[k]
            for k in (
                "start",
                "objective",
                "iterations",
                "physical_projected_gradient",
                "meets_gradient_tolerance",
                "elapsed_seconds",
                "boundary_parameters",
            )
            if k in record
        }

    save()
    print(
        json.dumps(
            {k: result[k] for k in ("parameters", "observations", "nodes", "state_dimension")}
        ),
        flush=True,
    )
    # Check the new coordinate chain rule before optimization, using several
    # directions spanning response, loading, offset, and noise coordinates.
    z = encode(points[0])
    value, g = evaluate(z)
    physical_value, physical_gradient = problem.value_gradient(points[0])
    np.testing.assert_allclose(value, physical_value, atol=1e-9, rtol=0)
    rng = np.random.default_rng(42)
    derivative_checks = []
    for _ in range(3):
        v = rng.normal(size=len(z))
        v /= np.linalg.norm(v)
        h = 1e-5
        fd = (evaluate(z + h * v)[0] - evaluate(z - h * v)[0]) / (2 * h)
        expected = float(g @ v)
        np.testing.assert_allclose(fd, expected, atol=2e-4, rtol=2e-5)
        derivative_checks.append(dict(finite_difference=float(fd), gradient_projection=expected))
    result["coordinate_derivative_checks"] = derivative_checks
    for index, point in enumerate(points):
        start = time.perf_counter()
        count = 0

        def callback(intermediate_result):
            nonlocal count
            count += 1
            if count == 1 or count % 50 == 0:
                result["status"] = "fitting"
                result["checkpoint"] = dict(
                    start=index,
                    iteration=count,
                    objective=float(intermediate_result.fun),
                    parameters=decode(intermediate_result.x).tolist(),
                    elapsed_seconds=time.perf_counter() - start,
                )
                save()
                print(
                    json.dumps(
                        {k: v for k, v in result["checkpoint"].items() if k != "parameters"}
                    ),
                    flush=True,
                )

        fitted = minimize(
            evaluate,
            encode(point),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=dict(maxiter=args.maxiter, ftol=1e-15, gtol=1e-8, maxcor=50, maxls=40),
        )
        record = _map_diagnostics(problem, decode(fitted.x), model.search, fitted)
        record.update(
            start=index,
            elapsed_seconds=time.perf_counter() - start,
            function_evaluations=int(fitted.nfev),
        )
        result["records"].append(record)
        save()
        print(json.dumps(summary(record)), flush=True)
    best = min(result["records"], key=lambda r: r["objective"])
    if args.polish_steps and not best["meets_gradient_tolerance"]:
        result["status"] = "polishing"
        result["pre_polish_best"] = {**best}
        save()
        best["polishing"] = polish(problem, best, model.search, args.polish_steps)
    result.update(
        status="finished",
        total_seconds=time.perf_counter() - started,
        best_start=best["start"],
        best_summary=summary(best),
    )
    result["filter_parameters"] = {
        "/".join(n[1:]): float(best["parameters"][i])
        for i, n in enumerate(problem.names)
        if n[0] == "filter"
    }
    value, gradient = problem.value_gradient(np.array(best["parameters"]))
    result["fresh_objective"] = float(value)
    result["gradient_by_parameter"] = [
        dict(parameter=list(n), value=float(v), gradient=float(g))
        for n, v, g in zip(problem.names, best["parameters"], gradient)
    ]
    np.savez(
        args.output.with_suffix(".npz"),
        parameters=np.asarray(best["parameters"]),
        starts=points,
        solutions=np.asarray([r["parameters"] for r in result["records"]]),
    )
    save()
    print(json.dumps(result["best_summary"]), flush=True)


if __name__ == "__main__":
    main()
