"""Compare nested response boxes on one fixed four-modality MAP objective.

The widest box defines row eligibility and prior normalization for every fit.
Narrower boxes constrain only the optimizer. Thus objective changes reflect
additional allowed parameter values, not changed data or normalization terms.
Uses the empirical loader's existing full-clip preprocessing; not a CV study.
"""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from benchmark_gp_response_candidates import load_data
from fit_gp_response_baseline import coordinate_map
from gp_response_candidates import candidate_model, split_physiology
from scipy.optimize import minimize

from multimodalsrm.bayesian.fitting import _map_diagnostics, initial_points
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.polishing import polish
from multimodalsrm.bayesian.quality import structured_shape

BOXES = {
    "original": {
        "face": {"scale": (0.3, 2.0), "lag": (-2.0, 6.0)},
        "rating": {"scale": (0.3, 2.0), "lag": (-2.0, 6.0)},
    },
    "expanded": {
        "face": {"scale": (0.1, 2.0), "lag": (-2.0, 6.0)},
        "rating": {"scale": (0.3, 3.0), "lag": (-6.0, 6.0)},
    },
    "wide": {
        "face": {"scale": (0.05, 2.0), "lag": (-2.0, 6.0)},
        "rating": {"scale": (0.3, 4.0), "lag": (-10.0, 6.0)},
    },
}


def comparison_model(data, features=1, algebra="state_space"):
    model = candidate_model(data, features=features, algebra=algebra)
    model.set_params(
        responses={
            m: replace(r, bounds={**r.bounds, **BOXES["wide"].get(m, {})})
            for m, r in model.responses.items()
        }
    )
    return model


def restricted_problem(problem, box):
    """Share the compiled density while changing optimizer/diagnostic bounds."""
    view = copy.copy(problem)
    view.bounds = list(problem.bounds)
    for modality, params in box.items():
        for parameter, bounds in params.items():
            index = problem.indices[("filter", modality, parameter)]
            lo, hi = problem.bounds[index]
            assert lo <= bounds[0] < bounds[1] <= hi
            view.bounds[index] = bounds
    return view


def observation_summary(problem):
    digest = hashlib.sha256()
    counts = {}
    for run, system in problem.systems.items():
        digest.update(str(run).encode())
        for name in ("times", "values"):
            digest.update(np.asarray(getattr(system, name), dtype="<f8").tobytes())
        digest.update(json.dumps(system.keys).encode())
        for key in system.keys:
            group = "/".join(key[:2])
            counts[group] = counts.get(group, 0) + 1
    return dict(
        scalar_observations=sum(counts.values()),
        counts=counts,
        sha256=digest.hexdigest(),
        nodes=sum(len(s.times) for s in problem.grouped_systems.values()),
    )


def brief(record):
    return {k: v for k, v in record.items() if k not in ("parameters", "polishing")}


def curves(model, problem, point):
    result = {}
    for m, response in model.responses.items():
        updates = {
            n[2]: float(point[i]) for i, n in enumerate(problem.names) if n[:2] == ("filter", m)
        }
        kernel = response.initial_kernel().with_parameters(**updates)
        result[m] = dict(parameters=kernel.parameters, shape=structured_shape(kernel, 0.0))
    return result


def validate(data, model, problem, result, save, orders):
    """Cross-check selected MAPs and physical response derivatives."""
    result.setdefault("validation", {name: {} for name in result["fits"]})
    for name, fit in result["fits"].items():
        x = np.asarray(fit["best"]["parameters"])
        value, gradient = problem.value_gradient(x)
        np.testing.assert_allclose(value, fit["best"]["objective"], atol=1e-7, rtol=0)
        checks = []
        for i, parameter in enumerate(problem.names):
            if parameter[0] != "filter":
                continue
            lo, hi = problem.bounds[i]
            h = 1e-5 * max(abs(x[i]), 0.1)
            v = np.eye(len(x))[i] * h
            if x[i] - lo >= h and hi - x[i] >= h:
                fd = (problem.value_gradient(x + v)[0] - problem.value_gradient(x - v)[0]) / (2 * h)
                scheme = "central"
            else:
                sign = 1 if hi - x[i] >= 2 * h else -1
                fd = (
                    sign
                    * (
                        -3 * value
                        + 4 * problem.value_gradient(x + sign * v)[0]
                        - problem.value_gradient(x + 2 * sign * v)[0]
                    )
                    / (2 * h)
                )
                scheme = "one_sided_second_order"
            checks.append(
                dict(
                    parameter=list(parameter),
                    gradient=float(gradient[i]),
                    finite_difference=float(fd),
                    absolute_difference=float(abs(fd - gradient[i])),
                    scheme=scheme,
                )
            )
        result["validation"][name]["response_derivatives"] = checks
    save()
    for order in orders:
        grouped_model = comparison_model(data, features=model.features, algebra="grouped")
        grouped_model.set_params(response_quadrature_order=order)
        _, grouped = _prepare(grouped_model, data)
        assert grouped.names == problem.names
        assert observation_summary(grouped) == observation_summary(problem)
        for name, fit in result["fits"].items():
            x = np.asarray(fit["best"]["parameters"])
            value, gradient = problem.value_gradient(x)
            other, score = grouped.value_gradient(x)
            check = dict(
                order=order,
                objective_absolute_difference=float(abs(value - other)),
                gradient_max_absolute_difference=float(np.max(abs(gradient - score))),
            )
            checks = result["validation"][name].setdefault("quadrature", [])
            checks[:] = sorted(
                [c for c in checks if c["order"] != order] + [check], key=lambda c: c["order"]
            )
            save()
            print(json.dumps(dict(box=name, quadrature=check)), flush=True)


def plot(model, result, filename):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for ax, m, end in zip(axes.ravel(), ("brain", "face", "rating", "eda"), (35, 12, 22, 22)):
        t = np.linspace(-10, end, 2000)
        for (name, fit), color, style in zip(
            result["fits"].items(), ("#777777", "#e69f00", "#176b91"), ("--", "-", ":")
        ):
            kernel = (
                model.responses[m]
                .initial_kernel()
                .with_parameters(**fit["responses"][m]["parameters"])
            )
            ax.plot(t, kernel.evaluate(t), color=color, ls=style, lw=2, label=name.capitalize())
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.set(
            title={
                "brain": "Brain · fixed double gamma",
                "face": "Face · gamma",
                "rating": "Ratings · gamma",
                "eda": "EDA · Bateman",
            }[m],
            xlabel="Time from model reference (s)",
            ylabel="L2-normalized response",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Response-bound sensitivity · identical observations and prior density\n"
        "Small empirical MAP; no uncertainty or held-out validation",
        fontsize=12,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    fig.savefig(filename.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("empirical", "synthetic"), default="empirical")
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int, default=5)
    parser.add_argument("--window", type=float, default=180)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--orders", type=int, nargs="+", default=[96, 192, 384])
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    if args.starts < 1 or args.maxiter < 1:
        parser.error("starts and maxiter must be positive")
    started = time.perf_counter()
    data = split_physiology(load_data(args))
    model = comparison_model(data, features=args.features)
    _, problem = _prepare(model, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        parameters=len(problem.names),
        parameter_names=[list(n) for n in problem.names],
        observations=observation_summary(problem),
        state_dimension=problem.features * problem.response_state_space.dimension,
        response_boxes=BOXES,
        response_supports={m: r.support_envelope() for m, r in model.responses.items()},
        covariance_tolerance=model.covariance_tolerance,
        covariance_error_bound=problem.covariance_error_bound,
        physical_gradient_tolerance=model.search.physical_gradient_tolerance,
        density="One widest-box prior density and fixed observations; nested optimizer boxes only",
        fits={},
        status="prepared",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.check_only:
        result = json.loads(args.output.read_text())
        assert result["observations"] == observation_summary(problem)
        assert result["parameter_names"] == [list(n) for n in problem.names]
    else:
        _, original = _prepare(candidate_model(data, features=args.features), data)
        result["original_eligibility"] = observation_summary(original)
        for run, system in problem.systems.items():
            larger = original.systems[run]
            originals = set(zip(map(tuple, larger.keys), larger.times, larger.values))
            assert set(zip(map(tuple, system.keys), system.times, system.values)) <= originals
        save()
        print(
            json.dumps(
                {k: result[k] for k in ("status", "observations", "covariance_error_bound")}
            ),
            flush=True,
        )
        # Reuse identical prior-quantile/data starts, clipped once to the inner
        # box. Later fits also start from the preceding box's selected optimum.
        points = initial_points(problem, args.starts, args.seed)
        inner = np.asarray(restricted_problem(problem, BOXES["original"]).bounds)
        points = np.clip(points, inner[:, 0], inner[:, 1])
        previous = None
        for name, box in BOXES.items():
            bounded = restricted_problem(problem, box)
            encode, decode, evaluate, bounds = coordinate_map(bounded)
            row = dict(records=[], observation_sha256=observation_summary(bounded)["sha256"])
            result["fits"][name] = row
            starts = list(points) + ([] if previous is None else [previous])
            for index, point in enumerate(starts):
                tick = time.perf_counter()
                count = 0

                def callback(intermediate_result):
                    nonlocal count
                    count += 1
                    if count == 1 or count % 50 == 0:
                        result["status"] = "fitting"
                        result["checkpoint"] = dict(
                            box=name,
                            start=index,
                            iteration=count,
                            objective=float(intermediate_result.fun),
                            seconds=time.perf_counter() - tick,
                        )
                        save()
                        print(json.dumps(result["checkpoint"]), flush=True)

                fitted = minimize(
                    evaluate,
                    encode(point),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=callback,
                    options=dict(maxiter=args.maxiter, ftol=1e-15, gtol=1e-8, maxcor=50, maxls=40),
                )
                record = _map_diagnostics(bounded, decode(fitted.x), model.search, fitted)
                record.update(
                    start=index,
                    initialization="prior_or_data" if index < len(points) else "previous_box_best",
                    seconds=time.perf_counter() - tick,
                )
                if not record["meets_gradient_tolerance"]:
                    record["polishing"] = polish(bounded, record, model.search, 3)
                record["seconds_including_polishing"] = time.perf_counter() - tick
                row["records"].append(record)
                save()
                print(json.dumps(dict(box=name, result=brief(record))), flush=True)
            best = min(row["records"], key=lambda r: r["objective"])
            row["best"] = best
            previous = np.asarray(best["parameters"])
            row["responses"] = curves(model, problem, previous)
            diagnostic = _map_diagnostics(problem, previous, model.search, fitted)
            row["wide_box_diagnostic"] = {
                key: diagnostic[key]
                for key in (
                    "objective",
                    "physical_projected_gradient",
                    "meets_gradient_tolerance",
                    "boundary_parameters",
                )
            }
            if len(result["fits"]) > 1:
                earlier = list(result["fits"].values())[-2]["best"]["objective"]
                assert best["objective"] <= earlier + 1e-7
            np.savez(
                args.output.with_name(args.output.stem + "-" + name + ".npz"),
                parameters=previous,
                starts=np.asarray(starts),
                solutions=np.asarray([r["parameters"] for r in row["records"]]),
            )
            save()
        result["fit_seconds_after_imports"] = time.perf_counter() - started
    result["status"] = "validating"
    save()
    validation_started = time.perf_counter()
    validate(data, model, problem, result, save, args.orders)
    result.setdefault("validation_runs", []).append(
        dict(
            orders=args.orders,
            seconds=time.perf_counter() - validation_started,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    )
    plot(model, result, args.figure)
    result.update(status="finished", figure=str(args.figure))
    save()
    print(json.dumps(dict(status="finished", output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
