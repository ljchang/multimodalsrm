"""Export aggregate response-family results and scientific comparison figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multimodalsrm import BatemanSCR, DoubleGamma, Gamma, Gaussian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument("--objective-tolerance", type=float, default=1e-3)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    tolerances = np.array([args.objective_tolerance, args.gradient_tolerance])
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        parser.error("Numerical validation tolerances must be finite and positive")
    fits = [json.loads(path.read_text()) for path in args.inputs]
    for fit in fits:
        assert fit["status"] == "finished"
        assert fit["best"]["meets_gradient_tolerance"]
        assert fit.get("quadrature_checks"), "Run fitted-point validation before export"
        check = fit["quadrature_checks"][-1]
        assert check["objective_absolute_difference"] <= args.objective_tolerance
        assert check["gradient_max_absolute_difference"] <= args.gradient_tolerance
        for field in (
            "training",
            "testing",
            "parameter_names",
            "holdout_blocks",
            "width_bounds",
            "peak_bounds",
            "response_priors",
        ):
            assert fit[field] == fits[0][field], field
        for field, default in (("features", 1), ("fold", "original")):
            assert fit["arguments"].get(field, default) == fits[0]["arguments"].get(field, default)
    summary = {
        key: fits[0][key]
        for key in (
            "commit",
            "training",
            "testing",
            "holdout_blocks",
            "common_support",
            "width_bounds",
            "peak_bounds",
            "response_priors",
            "map_coordinates",
            "preprocessing_checks",
        )
    }
    summary["candidates"] = {}
    summary["fitted_point_validation_gates"] = dict(
        objective_absolute_difference=args.objective_tolerance,
        gradient_max_absolute_difference=args.gradient_tolerance,
    )
    summary["scope"] = {
        key: fits[0]["arguments"][key]
        for key in ("subjects", "window", "parcels", "starts", "seed")
    }
    summary["scope"].update(
        latent_factors=fits[0]["arguments"].get("features", 1),
        fold=fits[0]["arguments"].get("fold", "original"),
        parameters=fits[0]["parameters"],
        data_loader=fits[0]["arguments"].get("loader"),
    )
    for fit, source in zip(fits, args.inputs):
        name = fit["arguments"]["candidate"]
        row = {
            key: fit[key]
            for key in (
                "device_kind",
                "python",
                "jax",
                "state_dimension",
                "timings",
                "fit_pipeline_seconds_after_imports",
                "scoring_seconds",
                "responses",
                "scores",
                "derivative_checks",
                "covariance_error_bound",
                "source_hashes",
            )
        }
        row["raw_source"] = str(source)
        if "data_loader_sha256" in fit:
            row["data_loader_sha256"] = fit["data_loader_sha256"]
        row["starts"] = [{k: v for k, v in r.items() if k != "parameters"} for r in fit["records"]]
        row["selected_start"] = fit["best"]["start"]
        row["quadrature_checks"] = fit.get("quadrature_checks", [])
        if "independent_workers" in fit:
            row["independent_workers"] = fit["independent_workers"]
            row["selection"] = fit["selection"]
        prediction_check = source.parent / (name + "-quadrature-prediction-check.json")
        if prediction_check.exists():
            checked = json.loads(prediction_check.read_text())
            assert checked["status"] == "finished" and checked["algebra"] == "grouped"
            assert checked["features"] == summary["scope"]["latent_factors"]
            assert checked["selected_start"] == fit["best"]["start"]
            for point in checked["checks"]:
                assert (
                    max(
                        point["mean_max_absolute_difference"],
                        point["variance_max_absolute_difference"],
                    )
                    <= checked["absolute_tolerance"]
                )
            row["independent_prediction_check"] = checked
        summary["candidates"][name] = row
    raw = args.inputs[0].parent
    summary["performance_checks"] = {}
    for name in (
        "probe-gamma3-cpu",
        "probe-gaussian-cpu",
        "probe-gamma3-pro6000",
        "parallel-gamma3-map-3090",
        "parallel-gaussian-map-3090",
    ):
        source = raw / (name + ".json")
        if not source.exists():
            continue
        check = json.loads(source.read_text())
        assert check["training"] == fits[0]["training"]
        if name.startswith("parallel"):
            row = {k: v for k, v in check.items() if k != "points"}
            row["points"] = [
                {k: v for k, v in p.items() if k != "gradient"} for p in check["points"]
            ]
        else:
            row = {k: check[k] for k in ("device_kind", "state_dimension", "timings", "status")}
        row["source"] = str(source)
        summary["performance_checks"][name] = row
    check_path = raw / "implementation-check.json"
    if check_path.exists():
        check = json.loads(check_path.read_text())
        summary["implementation_check"] = {k: v for k, v in check.items() if k != "fixture_metrics"}
    check_path = raw / "empirical-prediction-check.json"
    if check_path.exists():
        summary["empirical_prediction_check"] = json.loads(check_path.read_text())
    summary["limitations"] = [
        "One predeclared missing-block validation fold; no independent final test set.",
        f"Subjects {summary['scope']['subjects']}, {summary['scope']['parcels']} brain parcels, "
        f"{summary['scope']['latent_factors']} latent factor(s); EDA only available for one subject.",
        "MAP conditional predictive uncertainty omits parameter and preprocessing uncertainty.",
        "Gamma shape is the same for face and rating within each screened candidate; mixed shape combinations are not screened.",
        "Marginal NLPD treats scores separately; it is not a joint predictive density or independent sample count.",
        "CPU/GPU and overlapping independent-worker fit times are not a controlled end-to-end speedup comparison.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    colors = ("#cc9a06", "#c76121", "#187b9e", "#724aa0")
    modalities = ("brain", "face", "rating", "eda")
    labels = tuple(
        f"{label} ({fits[0]['scores']['modalities'][m]['observations']})"
        for m, label in zip(modalities, ("Brain", "Face", "Ratings", "EDA"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, metric, xlabel in zip(
        axes,
        ("rmse", "marginal_nlpd"),
        ("RMSE · training SD units", "Mean marginal negative log predictive density"),
    ):
        for index, (name, row) in enumerate(summary["candidates"].items()):
            values = [row["scores"]["modalities"][m][metric] for m in modalities]
            ax.scatter(
                values,
                np.arange(4) + (index - (len(fits) - 1) / 2) * 0.15,
                color=colors[index],
                s=45,
                label=name.replace("gamma", "Gamma ").capitalize(),
            )
        baseline = "zero_prediction_rmse" if metric == "rmse" else "unit_normal_nlpd"
        ax.scatter(
            [fits[0]["scores"]["modalities"][m][baseline] for m in modalities],
            np.arange(4),
            marker="x",
            color="#888888",
            s=35,
            label="Training mean / unit variance",
        )
        ax.set(yticks=np.arange(4), yticklabels=labels, xlabel=xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Held-out 20-second blocks · lower scores are better\nNumbers in parentheses are scalar observations, not independent samples",
        fontsize=11,
    )
    fig.savefig(str(args.figure_prefix) + "-scores.svg")
    fig.savefig(str(args.figure_prefix) + "-scores.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for ax, m, end in zip(axes.ravel(), modalities, (35, 25, 40, 25)):
        times = np.linspace(-12, end, 2000)
        for (name, row), color in zip(summary["candidates"].items(), colors):
            family = {"brain": DoubleGamma, "eda": BatemanSCR}.get(
                m, Gaussian if name == "gaussian" else Gamma
            )
            kernel = family(**row["responses"][m]["parameters"])
            ax.plot(
                times,
                kernel.evaluate(times),
                color=color,
                lw=1.7,
                label=name.replace("gamma", "Gamma ").capitalize(),
            )
        ax.set(
            title={"rating": "Ratings", "eda": "EDA"}.get(m, m.capitalize()),
            xlabel="Time from model reference (s)",
            ylabel="L2-normalized response",
        )
        ax.axhline(0, color="#dddddd", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Training MAP response curves · matched width/peak priors\nNo posterior uncertainty; fixed double-gamma brain reference",
        fontsize=11,
    )
    fig.savefig(str(args.figure_prefix) + "-responses.svg")
    fig.savefig(str(args.figure_prefix) + "-responses.png", dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
