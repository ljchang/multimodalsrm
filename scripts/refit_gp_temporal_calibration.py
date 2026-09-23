"""Training-only timescale/noise refits with every validation schedule reserved.

Research artifacts contain fitted parameters: write them under local_data.
Fixed OU timescales are an explicit model specification, not learned parameters.
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
import numpy as np
from benchmark_gp_response_candidates import load_data
from compare_gp_response_families import test_summary
from fit_gp_response_baseline import coordinate_map
from gp_response_candidates import split_physiology
from gp_response_holdout import BLOCK_FOLDS, CanonicalProblem, make_model, split_data
from investigate_gp_response_bounds import observation_summary
from probe_gp_correlated_noise import TIMESCALES
from scipy.optimize import minimize
from threadpoolctl import threadpool_info

from multimodalsrm.bayesian.fitting import _map_diagnostics, initial_points
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.problem import BayesianProblem
from multimodalsrm.bayesian.quality import structured_shape


def prepare(args, *, order=None):
    split = split_data(
        split_physiology(load_data(args)), fold=args.fold, excluded_folds=tuple(BLOCK_FOLDS)
    )
    algebra = "state_space" if args.noise == "independent" else "grouped"
    if getattr(args, "algebra", "auto") != "auto":
        algebra = args.algebra
    if order is not None:
        algebra = "grouped"
    order = args.order if order is None else order
    noise = TIMESCALES if args.noise == "ou" else {}
    model = make_model(split["training"], args.candidate, algebra, order, features=args.features)
    model.set_params(length_scale=args.length_scale, noise_timescales=noise)
    adapter, full = _prepare(model, split["training"])
    systems, _ = adapter._systems(split["common"], adapter.domains_)
    base = BayesianProblem(
        adapter,
        model.priors,
        systems=systems,
        anchor=full.anchor,
        reference_modality="brain",
        linear_algebra=algebra,
        response_quadrature_order=None if algebra == "state_space" else order,
        noise_timescales=noise,
    )
    return split, model, CanonicalProblem(base, args.candidate)


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_start(problem, model, point, index, result, args):
    encode, decode, evaluate, bounds = coordinate_map(problem)
    started = time.perf_counter()
    iterations = 0

    def callback(intermediate_result):
        nonlocal iterations
        iterations += 1
        if iterations == 1 or iterations % 25 == 0:
            result["checkpoint"] = dict(
                start=index,
                iteration=iterations,
                objective=float(intermediate_result.fun),
                parameters=decode(intermediate_result.x).tolist(),
                seconds=time.perf_counter() - started,
            )
            result["status"] = "fitting"
            save(args.output, result)
            print(
                json.dumps({k: v for k, v in result["checkpoint"].items() if k != "parameters"}),
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
    if not record["meets_gradient_tolerance"]:
        previous = {k: v for k, v in record.items() if k != "parameters"}
        fitted = minimize(
            evaluate,
            fitted.x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=dict(maxiter=400, ftol=0, gtol=1e-9, maxcor=20, maxls=50),
        )
        record = _map_diagnostics(problem, decode(fitted.x), model.search, fitted)
        record["before_retry"] = previous
    record.update(start=index, seconds=time.perf_counter() - started, total_iterations=iterations)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("gamma3", "gaussian"), required=True)
    parser.add_argument("--noise", choices=("independent", "ou"), required=True)
    parser.add_argument("--length-scale", type=float, choices=(3.0, 10.0, 30.0), required=True)
    parser.add_argument("--action", choices=("probe", "fit", "validate"), default="probe")
    parser.add_argument("--algebra", choices=("auto", "grouped"), default="auto")
    parser.add_argument("--source", choices=("empirical",), default="empirical")
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument("--parcels", type=int, default=100)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--fold", choices=BLOCK_FOLDS, default="original")
    parser.add_argument(
        "--loader", type=Path, default=Path(__file__).with_name("clock_preserving_emo_data.py")
    )
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--start-index", type=int, help="Otherwise finish all seeded starts")
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--order", type=int, default=384)
    parser.add_argument("--validation-order", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    if args.starts < 1 or (
        args.start_index is not None and not 0 <= args.start_index < args.starts
    ):
        parser.error("Invalid indexed restart")
    if args.output.exists() and args.action != "validate":
        parser.error("Refusing to overwrite an existing probe or fit")
    begun = time.perf_counter()
    split, model, problem = prepare(args)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        jax=jax.__version__,
        device_kind=jax.devices()[0].device_kind,
        execution=dict(
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        ),
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("gp_response_holdout.py", Path(__file__).name)
        },
        data_loader_sha256=hashlib.sha256(args.loader.read_bytes()).hexdigest(),
        training=observation_summary(problem.base),
        testing=test_summary(split["testing"]),
        reserved_folds=BLOCK_FOLDS,
        selection_scope="All three validation schedules excluded from fitting, standardization, and timescale selection",
        parameter_names=[list(n) for n in problem.names],
        noise_timescales=TIMESCALES if args.noise == "ou" else {},
        algebra=problem.base.linear_algebra,
        preparation_seconds=time.perf_counter() - begun,
        status="prepared",
    )
    if args.action == "validate":
        saved = json.loads(args.output.read_text())
        assert saved["algebra"] == result["algebra"]
        for key in ("training", "testing", "parameter_names", "noise_timescales", "reserved_folds"):
            assert saved[key] == json.loads(json.dumps(result[key])), key
        for key in ("candidate", "length_scale", "noise", "features", "order"):
            assert saved["arguments"][key] == result["arguments"][key], key
        point = np.asarray(saved["best"]["parameters"])
        _, _, reference = prepare(args, order=args.validation_order)
        value, gradient = problem.value_gradient(point)
        np.testing.assert_allclose(value, saved["best"]["objective"], atol=1e-6, rtol=0)
        other, independent = reference.value_gradient(point)
        check = dict(
            order=args.validation_order,
            device_kind=result["device_kind"],
            validator_sha256=result["source_hashes"][Path(__file__).name],
            objective_absolute_difference=float(abs(value - other)),
            gradient_max_absolute_difference=float(max(abs(gradient - independent))),
        )
        saved.setdefault("quadrature_checks", []).append(check)
        save(args.output, saved)
        print(json.dumps(check), flush=True)
        return
    save(args.output, result)
    points = initial_points(problem, args.starts, args.seed)
    timings = []
    for _ in range(4):
        start = time.perf_counter()
        value, gradient = problem.value_gradient(points[0])
        assert np.isfinite(value) and np.isfinite(gradient).all()
        timings.append(time.perf_counter() - start)
    result["timings"] = dict(
        first_value_gradient_seconds=timings[0], warm_value_gradient_seconds=timings[1:]
    )
    result["initial_objective"] = float(value)
    result["execution"]["threadpools"] = [
        {k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads")}
        for p in threadpool_info()
    ]
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=points[0], gradient=gradient)
    print(json.dumps(result["timings"]), flush=True)
    save(args.output, result)
    encode, _, evaluate, _ = coordinate_map(problem)
    z = encode(points[0])
    _, gradient = evaluate(z)
    rng = np.random.default_rng(42)
    checks = []
    for _ in range(3):
        direction = rng.normal(size=len(z))
        direction /= np.linalg.norm(direction)
        step = 1e-5
        finite = (evaluate(z + step * direction)[0] - evaluate(z - step * direction)[0]) / (
            2 * step
        )
        expected = float(gradient @ direction)
        np.testing.assert_allclose(finite, expected, atol=5e-4, rtol=3e-5)
        checks.append(dict(finite_difference=float(finite), gradient_projection=expected))
    result["derivative_checks"] = checks
    if args.action == "probe":
        result["status"] = "probed"
    else:
        result["records"] = []
        indices = range(args.starts) if args.start_index is None else [args.start_index]
        for index in indices:
            record = fit_start(problem, model, points[index], index, result, args)
            result["records"].append(record)
            save(args.output, result)
        record = min(result["records"], key=lambda r: r["objective"])
        result["best"] = record
        native = np.asarray(problem.native(np.asarray(record["parameters"])))
        result["responses"] = {}
        for modality, response in model.responses.items():
            updates = {
                name[2]: float(native[i])
                for i, name in enumerate(problem.base.names)
                if name[:2] == ("filter", modality)
            }
            kernel = response.initial_kernel().with_parameters(**updates)
            result["responses"][modality] = dict(
                parameters=kernel.parameters, shape=structured_shape(kernel, 0)
            )
        result["status"] = "fit_finished"
    result["elapsed_seconds"] = time.perf_counter() - begun
    save(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "timings", "derivative_checks", "elapsed_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
