"""Fit and score a predeclared missing-block response-family validation fold.

Research only: matched physical FWHM/peak priors, common eligible observations,
training-only standardization, MAP inference, and marginal predictive scores.
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
from fit_gp_response_baseline import coordinate_map
from gp_response_candidates import split_physiology
from gp_response_holdout import (
    BLOCK_FOLDS,
    CANDIDATES,
    PEAK_BOUNDS,
    WIDTH_BOUNDS,
    prepare,
    split_data,
)
from investigate_gp_response_bounds import observation_summary
from scipy.optimize import minimize

from multimodalsrm.bayesian.fitting import _map_diagnostics, initial_points
from multimodalsrm.bayesian.quality import structured_shape
from multimodalsrm.bayesian.state_space import smoother


def test_summary(testing):
    h = hashlib.sha256()
    counts = {}
    for s, runs in testing.items():
        for r, mods in runs.items():
            for m, ts in mods.items():
                h.update(json.dumps([s, r, m]).encode())
                h.update(ts.mask.tobytes())
                h.update(ts.times.tobytes())
                h.update(ts.values[ts.mask].tobytes())
                counts[f"{s}/{m}"] = int(ts.mask.sum())
    return dict(counts=counts, total=sum(counts.values()), sha256=h.hexdigest())


def score(problem, point, testing, filename):
    """One smoother per modality; retain every feature's noise and loading."""
    base = problem.base
    native = np.asarray(problem.native(point))
    weights, offsets, noise, _, _ = map(np.asarray, base.arrays(native))
    weights = weights.reshape(len(base.keys), base.features)
    records, arrays = {}, {}
    for run in base.systems:
        for m in base.modalities:
            queries = np.unique(
                np.concatenate(
                    [
                        runs[run][m].times[runs[run][m].mask.any(axis=1)]
                        for runs in testing.values()
                        if m in runs[run]
                    ]
                )
            )
            evaluate = jax.jit(smoother(base, run, queries, base.modalities.index(m)))
            mean, covariance = map(np.asarray, jax.block_until_ready(evaluate(native)))
            if not np.isfinite(covariance).all() or np.any(np.linalg.eigvalsh(covariance) < -1e-8):
                raise FloatingPointError("Invalid smoothed functional variance")
            observed, means, variances = [], [], []
            for s, runs in testing.items():
                if m not in runs[run]:
                    continue
                ts = runs[run][m]
                rows, features = np.nonzero(ts.mask)
                lookup = np.searchsorted(queries, ts.times[rows])
                keys = [base.keys.index((s, m, int(f))) for f in features]
                w, offset = weights[keys], offsets[keys]
                prediction = np.einsum("ik,ik->i", mean[lookup], w) + offset
                uncertainty = (
                    np.maximum(np.einsum("ik,ikl,il->i", w, covariance[lookup], w), 0)
                    + noise[base.groups.index((s, m))]
                )
                observed.extend(ts.values[rows, features])
                means.extend(prediction)
                variances.extend(uncertainty)
            y, mu, var = map(np.asarray, (observed, means, variances))
            assert np.isfinite(mu).all() and np.isfinite(var).all() and np.all(var > 0)
            error = y - mu
            records[m] = dict(
                observations=len(y),
                rmse=float(np.sqrt(np.mean(error**2))),
                marginal_nlpd=float(np.mean(0.5 * (np.log(2 * np.pi * var) + error**2 / var))),
                coverage_95=float(np.mean(np.abs(error) <= 1.95996398454 * np.sqrt(var))),
                zero_prediction_rmse=float(np.sqrt(np.mean(y**2))),
                unit_normal_nlpd=float(np.mean(0.5 * (np.log(2 * np.pi) + y**2))),
            )
            arrays.update({m + "_observed": y, m + "_mean": mu, m + "_variance": var})
    np.savez(filename, **arrays)
    return dict(
        modalities=records,
        macro_rmse=float(np.mean([r["rmse"] for r in records.values()])),
        macro_marginal_nlpd=float(np.mean([r["marginal_nlpd"] for r in records.values()])),
        uncertainty="conditional on MAP parameters and training standardization",
        task="missing-block smoothing with other modalities; not forecasting or new-subject prediction",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--action", choices=("probe", "fit", "score", "validate"), default="fit")
    parser.add_argument("--source", default="empirical", choices=("empirical",))
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument("--parcels", type=int, default=5)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--fold", choices=BLOCK_FOLDS, default="original")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument(
        "--start-index", type=int, help="Fit one indexed start from the unchanged seeded design"
    )
    parser.add_argument(
        "--fit-only", action="store_true", help="Save fits without scoring holdouts"
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--order", type=int, default=384)
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
    if args.starts < 1 or (
        args.start_index is not None and not 0 <= args.start_index < args.starts
    ):
        parser.error("--starts must be positive and --start-index must lie in [0, starts)")
    begun = time.perf_counter()
    split = split_data(split_physiology(load_data(args)), fold=args.fold)
    model, problem = prepare(split, args.candidate, features=args.features)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("gp_response_holdout.py", "compare_gp_response_families.py")
        },
        data_loader_sha256=hashlib.sha256(args.loader.read_bytes()).hexdigest(),
        parameters=len(problem.names),
        parameter_names=[list(n) for n in problem.names],
        training=observation_summary(problem.base),
        testing=test_summary(split["testing"]),
        holdout_blocks=split["blocks"],
        common_support=split["support"],
        width_bounds=WIDTH_BOUNDS,
        peak_bounds=PEAK_BOUNDS,
        response_priors="FWHM LogNormal(log(3.4), 0.7); peak Normal(2,3); truncated to common bounds",
        map_coordinates="Physical FWHM/peak, loadings, offsets, noise variance, EDA parameters; optimizer uses log noise only",
        preprocessing_checks="training-only statistics; held-out payload perturbation and affine-invariance assertions passed",
        state_dimension=problem.features * problem.base.response_state_space.dimension,
        covariance_tolerance=model.covariance_tolerance,
        covariance_error_bound=problem.base.covariance_error_bound,
        preparation_seconds=time.perf_counter() - begun,
        records=[],
        status="prepared",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    if args.action in ("score", "validate"):
        saved = json.loads(args.output.read_text())
        for field in (
            "training",
            "testing",
            "parameter_names",
            "width_bounds",
            "peak_bounds",
            "response_priors",
            "holdout_blocks",
        ):
            assert saved[field] == json.loads(json.dumps(result[field])), field
        result = saved
        x = np.asarray(result["best"]["parameters"])
        value, _ = problem.value_gradient(x)
        np.testing.assert_allclose(value, result["best"]["objective"], atol=1e-6, rtol=0)
    else:
        save()
        print(
            json.dumps(
                {k: result[k] for k in ("status", "state_dimension", "training", "testing")}
            ),
            flush=True,
        )
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
        save()
        print(json.dumps(dict(timings=result["timings"])), flush=True)
        if args.action == "probe":
            result["status"] = "probed"
            save()
            return
        encode, decode, evaluate, bounds = coordinate_map(problem)
        checks = []
        z = encode(points[0])
        _, gradient = evaluate(z)
        rng = np.random.default_rng(42)
        for _ in range(3):
            direction = rng.normal(size=len(z))
            direction /= np.linalg.norm(direction)
            h = 1e-5
            fd = (evaluate(z + h * direction)[0] - evaluate(z - h * direction)[0]) / (2 * h)
            expected = float(gradient @ direction)
            np.testing.assert_allclose(fd, expected, atol=5e-4, rtol=3e-5)
            checks.append(dict(finite_difference=float(fd), gradient_projection=expected))
        result["derivative_checks"] = checks
        for index, point in enumerate(points):
            if args.start_index is not None and index != args.start_index:
                continue
            start = time.perf_counter()
            iterations = 0

            def callback(intermediate_result):
                nonlocal iterations
                iterations += 1
                if iterations == 1 or iterations % 50 == 0:
                    result["checkpoint"] = dict(
                        start=index,
                        iteration=iterations,
                        objective=float(intermediate_result.fun),
                        parameters=decode(intermediate_result.x).tolist(),
                        seconds=time.perf_counter() - start,
                    )
                    result["status"] = "fitting"
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
            if not record["meets_gradient_tolerance"]:
                record["before_retry"] = {k: v for k, v in record.items() if k != "parameters"}
                retry = minimize(
                    evaluate,
                    fitted.x,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=callback,
                    options=dict(maxiter=400, ftol=0, gtol=1e-9, maxcor=20, maxls=50),
                )
                refreshed = _map_diagnostics(problem, decode(retry.x), model.search, retry)
                record.update(refreshed)
            record.update(
                start=index, seconds=time.perf_counter() - start, total_iterations=iterations
            )
            result["records"].append(record)
            save()
            print(json.dumps({k: v for k, v in record.items() if k != "parameters"}), flush=True)
        result["best"] = min(result["records"], key=lambda r: r["objective"])
        x = np.asarray(result["best"]["parameters"])
        native = np.asarray(problem.native(x))
        result["responses"] = {}
        for m, response in model.responses.items():
            updates = {
                n[2]: float(native[i])
                for i, n in enumerate(problem.base.names)
                if n[:2] == ("filter", m)
            }
            kernel = response.initial_kernel().with_parameters(**updates)
            result["responses"][m] = dict(
                parameters=kernel.parameters, shape=structured_shape(kernel, 0)
            )
        result["fit_pipeline_seconds_after_imports"] = time.perf_counter() - begun
        np.savez(
            args.output.with_suffix(".npz"),
            canonical_parameters=x,
            native_parameters=native,
            starts=points,
        )
        result["status"] = "fit_finished"
        save()
        if args.fit_only:
            return
    if args.action == "validate":
        _, grouped = prepare(split, args.candidate, "grouped", args.order, features=args.features)
        assert observation_summary(grouped.base) == observation_summary(problem.base)
        value, gradient = problem.value_gradient(x)
        other, score_gradient = grouped.value_gradient(x)
        check = dict(
            order=args.order,
            objective_absolute_difference=float(abs(value - other)),
            gradient_max_absolute_difference=float(np.max(abs(gradient - score_gradient))),
        )
        result.setdefault("quadrature_checks", []).append(check)
        save()
        print(json.dumps(check), flush=True)
        return
    start = time.perf_counter()
    result["status"] = "scoring"
    save()
    result["scores"] = score(
        problem, x, split["testing"], args.output.with_name(args.output.stem + "-predictions.npz")
    )
    result["scoring_seconds"] = time.perf_counter() - start
    result["status"] = "finished"
    save()
    print(json.dumps(dict(status="finished", scores=result["scores"])), flush=True)


if __name__ == "__main__":
    main()
