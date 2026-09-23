"""Export aggregate results only after refit, selection, and numerical gates pass."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from score_gp_temporal_calibration import parameter_hash
from select_gp_temporal_calibration import select

LABELS = {
    ("gamma3", "independent"): "Gamma / independent",
    ("gaussian", "independent"): "Gaussian / independent",
    ("gamma3", "ou"): "Gamma / OU",
    ("gaussian", "ou"): "Gaussian / OU",
}
MODALITIES = ("brain", "face", "rating", "eda")


def read(path):
    return json.loads(path.read_text())


def build(root):
    selection = read(root / "selection.json")
    assert selection["selected"] == select(root)["selected"]
    fixture = read(root / "fixture-checks.json")
    assert len(fixture["checks"]) == 30
    assert max(fixture["max_mean_error"], fixture["max_variance_error"]) <= 1e-8
    fits = []
    for row in selection["records"]:
        fit = read(root / row["filename"])
        assert any(
            c["order"] >= 768
            and c["objective_absolute_difference"] <= 1e-3
            and c["gradient_max_absolute_difference"] <= 1e-4
            for c in fit.get("quadrature_checks", [])
        ), f"Missing/pending quadrature gate: {row['filename']}"
        fits.append(
            dict(
                **row,
                algebra=fit["algebra"],
                device_kind=fit["device_kind"],
                timings=fit["timings"],
                elapsed_seconds=fit["elapsed_seconds"],
                quadrature_checks=fit.get("quadrature_checks", []),
                starts=[
                    {
                        k: r[k]
                        for k in (
                            "start",
                            "objective",
                            "optimizer_success",
                            "physical_projected_gradient",
                            "meets_gradient_tolerance",
                            "total_iterations",
                            "seconds",
                            "message",
                        )
                    }
                    for r in fit["records"]
                ],
                source_hashes=fit["source_hashes"],
                data_loader_sha256=fit["data_loader_sha256"],
            )
        )
    selected = []
    for row in selection["selected"]:
        fit = read(root / row["filename"])
        stem = Path(row["filename"]).stem
        coarse = read(root / f"{stem}-scores-384.json")
        fine = read(root / f"{stem}-scores-768.json")
        assert coarse["status"] == fine["status"] == "finished"
        for data in (coarse, fine):
            assert data["parameter_sha256"] == parameter_hash(fit)
            assert data["training"] == selection["training"]
        errors = {}
        with (
            np.load(root / f"{stem}-scores-384.npz") as a,
            np.load(root / f"{stem}-scores-768.npz") as b,
        ):
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].shape == b[key].shape
                if key.endswith(("_observed", "_interpolation")):
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    error = float(np.max(abs(a[key] - b[key])))
                    assert error <= 1e-4, (stem, key, error)
                    errors[key] = error
        checks = [c for fold in coarse["folds"].values() for c in fold["prediction_api_checks"]]
        assert len(checks) == 5
        assert max(max(c["mean_error"], c["variance_error"]) for c in checks) <= 1e-7
        selected.append(
            dict(
                **row,
                responses=fit["responses"],
                boundary_parameters=fit["best"]["boundary_parameters"],
                folds=coarse["folds"],
                brain_metadata_sha256=coarse["brain_metadata_sha256"],
                post_splice_scope=coarse["post_splice_scope"],
                prediction_refinement=errors,
                score_seconds_384=coarse["seconds"],
                score_seconds_768=fine["seconds"],
                scorer_sha256=coarse["script_sha256"],
            )
        )
    references = {}
    for device in ("cpu", "gpu", "gpu3090"):
        fit = read(root / f"{device}-reference-gamma3-ou-3-start0.json")
        assert fit["status"] == "fit_finished" and len(fit["records"]) == 1
        assert fit["best"]["start"] == 0 and fit["best"]["meets_gradient_tolerance"]
        assert fit["training"] == selection["training"]
        references[device] = dict(
            device_kind=fit["device_kind"],
            objective=fit["best"]["objective"],
            gradient=fit["best"]["physical_projected_gradient"],
            iterations=fit["best"]["total_iterations"],
            optimizer_seconds=fit["best"]["seconds"],
            pipeline_seconds=fit["elapsed_seconds"],
            timings=fit["timings"],
            execution=fit.get(
                "execution",
                dict(
                    source="recorded launch command",
                    cpu_affinity=list(range(56, 64)),
                    openblas_num_threads="1",
                    omp_num_threads="1",
                ),
            ),
        )
    for device in ("gpu", "gpu3090"):
        with (
            np.load(root / "cpu-reference-gamma3-ou-3-start0.npz") as a,
            np.load(root / f"{device}-reference-gamma3-ou-3-start0.npz") as b,
        ):
            np.testing.assert_array_equal(a["parameters"], b["parameters"])
        assert abs(references["cpu"]["objective"] - references[device]["objective"]) <= 1e-5
    probes = {}
    for device in ("cpu", "cpu-blas8", "3090", "pro6000"):
        data = read(root / f"probe-gamma-ou-{device}.json")
        assert data["status"] == "probed" and data["training"] == selection["training"]
        probes[device] = {k: data[k] for k in ("device_kind", "timings", "derivative_checks")}
        first_core = {"cpu": 24, "cpu-blas8": 40, "3090": 0, "pro6000": 8}[device]
        probes[device]["execution"] = data.get(
            "execution",
            dict(
                source="recorded launch command",
                cpu_affinity=list(range(first_core, first_core + 8)),
                openblas_num_threads="1",
                omp_num_threads="1",
            ),
        )
    scaling = read(root / "probe-gamma-ou-3090-k6.json")
    assert scaling["status"] == "probed" and scaling["arguments"]["features"] == 6
    assert scaling["training"] == selection["training"]
    scaling = dict(
        features=6,
        parameters=len(scaling["parameter_names"]),
        functional_nodes=scaling["training"]["nodes"],
        grouped_dimension=6 * scaling["training"]["nodes"],
        device_kind=scaling["device_kind"],
        timings=scaling["timings"],
        derivative_checks=scaling["derivative_checks"],
        execution=scaling["execution"],
        scope="Cost and directional derivative probe only; no six-factor fit or predictive qualification, and no increase in observed feature count",
    )
    gamma = read(root / "probe-gamma-ou-3090.json")
    gaussian = read(root / "gaussian-ou-3.json")
    assert gamma["training"] == gaussian["training"]
    assert gamma["parameter_names"] == gaussian["parameter_names"]
    with np.load(root / "probe-gamma-ou-3090.npz") as a, np.load(root / "gaussian-ou-3.npz") as b:
        np.testing.assert_array_equal(a["parameters"], b["parameters"])
    family_timing = dict(
        device_kind=gamma["device_kind"],
        gamma3_warm_seconds=gamma["timings"]["warm_value_gradient_seconds"],
        gaussian_warm_seconds=gaussian["timings"]["warm_value_gradient_seconds"],
        scope="Identical physical initial coordinates, observed data, grouped quadrature order and fixed noise/GP timescales; different response families, three warm repetitions each",
    )
    algebra_probes = {
        algebra: read(root / f"probe-gaussian-independent-3090-{algebra}.json")
        for algebra in ("auto", "grouped")
    }
    for probe in algebra_probes.values():
        assert probe["status"] == "probed" and probe["training"] == selection["training"]
    with (
        np.load(root / "probe-gaussian-independent-3090-auto.npz") as a,
        np.load(root / "probe-gaussian-independent-3090-grouped.npz") as b,
    ):
        np.testing.assert_array_equal(a["parameters"], b["parameters"])
        gradient_error = float(np.max(abs(a["gradient"] - b["gradient"])))
    objective_error = abs(
        algebra_probes["auto"]["initial_objective"] - algebra_probes["grouped"]["initial_objective"]
    )
    assert objective_error <= 1e-3 and gradient_error <= 1e-4
    algebra_timing = dict(
        device_kind=algebra_probes["auto"]["device_kind"],
        objective_error=objective_error,
        gradient_error=gradient_error,
        state_space=algebra_probes["auto"]["timings"],
        grouped=algebra_probes["grouped"]["timings"],
        scope="Same Gaussian independent-noise model, physical point, data and GPU; production state-space versus order-384 grouped quadrature, initial-point comparison only",
    )
    return dict(
        protocol="Three starts for each of twelve family/noise/GP-timescale configurations; all three block schedules excluded before refitting and training selection",
        validation_scope="Additional blocked validation within a previously explored two-subject cohort, not an independent test cohort",
        training=selection["training"],
        environment=read(root / "environment.json"),
        selection=selection,
        fits=fits,
        selected=selected,
        gradient_benchmarks=probes,
        six_factor_probe=scaling,
        response_family_timing=family_timing,
        gaussian_algebra_timing=algebra_timing,
        device_agreement=read(root / "device-agreement.json"),
        gaussian_backend_check=read(root / "gaussian-backend-check.json"),
        complete_fit_benchmark=references,
        complete_fit_speed_ratio=references["cpu"]["pipeline_seconds"]
        / references["gpu"]["pipeline_seconds"],
        complete_fit_3090_speed_ratio=references["cpu"]["pipeline_seconds"]
        / references["gpu3090"]["pipeline_seconds"],
        fixture_checks=fixture,
    )


def plot(result, directory):
    order = sorted(
        result["selected"], key=lambda row: list(LABELS).index((row["candidate"], row["noise"]))
    )
    fig, axes = plt.subplots(3, 4, figsize=(13.8, 9), layout="constrained")
    colors = ["#32936f", "#437bb4", "#32936f", "#437bb4"]
    for i, fold in enumerate(("original", "rotated", "late")):
        for j, modality in enumerate(MODALITIES):
            ax = axes[i, j]
            rows = [model["folds"][fold]["modalities"][modality] for model in order]
            baseline = rows[0]["zero_prediction_rmse"]
            ax.bar(
                np.arange(4),
                [r["observation_rmse"] / baseline for r in rows],
                color=colors,
                alpha=0.8,
            )
            ax.scatter(
                [2, 3],
                [rows[k]["shared_only_rmse"] / baseline for k in (2, 3)],
                marker="x",
                color="black",
                zorder=5,
            )
            ax.axhline(1, color="#666666", linewidth=1)
            ax.axhline(
                rows[0]["interpolation_rmse"] / baseline,
                color="#bb6655",
                linestyle="--",
                linewidth=1,
            )
            ax.set_xticks(
                np.arange(4),
                ["Gamma\nindep.", "Gaussian\nindep.", "Gamma\nOU", "Gaussian\nOU"],
                fontsize=8,
            )
            title = {"brain": "Brain", "face": "Face", "rating": "Ratings", "eda": "EDA"}[modality]
            ax.set_title(f"{fold.title()} · {title}")
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("RMSE / training-mean RMSE")
    fig.suptitle(
        "Training-selected refits on three reserved block schedules\nBars: total observation; ×: shared signal only; dashed: interpolation; gray: training mean",
        fontsize=12,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        path = directory / f"gp-temporal-refit-scores.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def report_tables(result):
    text = [
        "<!-- RESULTS:START -->",
        "",
        "The [portable aggregate results](2026-09-23-gp-temporal-refit-results.json) contain every restart, all training profiles, selected response parameters, validation scores, and numerical gates.",
        "",
        "## Training profiles",
        "",
        "Each row reports the lowest objective among three completed starts. Timescales are selected within each family/noise cell; objectives are not independent predictive scores.",
        "",
        "| Family | Noise | GP seconds | Training objective | Qualifying starts | Best qualifies | Selected |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    chosen = {r["filename"] for r in result["selected"]}
    for row in result["fits"]:
        count = sum(r["meets_gradient_tolerance"] for r in row["starts"])
        text.append(
            f"| {row['candidate']} | {row['noise']} | {row['length_scale']} | {row['objective']:.3f} | {count}/3 | {'Yes' if row['qualified'] else 'No'} | {'Yes' if row['filename'] in chosen else ''} |"
        )
    text.extend(
        [
            "",
            "## Selected response estimates",
            "",
            "All quantities are seconds. These are constrained MAP response shapes, conditional on the noise specification and this training mask; they are not precise physiological latency estimates.",
            "",
            "| Model | Face FWHM | Face peak | Rating FWHM | Rating peak | EDA decay | EDA peak |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["selected"]:
        response = row["responses"]
        values = [
            response[m]["shape"][field]
            for m in ("face", "rating")
            for field in ("fwhm_seconds", "peak_delay_seconds")
        ]
        values += [
            response["eda"]["parameters"]["decay"],
            response["eda"]["shape"]["peak_delay_seconds"],
        ]
        text.append(
            "| "
            + LABELS[row["candidate"], row["noise"]]
            + " | "
            + " | ".join(f"{v:.3f}" for v in values)
            + " |"
        )
    for fold in ("original", "rotated", "late"):
        text.extend(
            [
                "",
                f"## {fold.title()} block validation",
                "",
                "RMSE in the common training-standard-deviation units; lower is better.",
                "",
                "| Prediction | Brain | Face | Ratings | EDA |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        first = result["selected"][0]["folds"][fold]["modalities"]
        for label, key in (
            ("Training mean", "zero_prediction_rmse"),
            ("Linear interpolation", "interpolation_rmse"),
        ):
            text.append(
                "| " + label + " | " + " | ".join(f"{first[m][key]:.3f}" for m in MODALITIES) + " |"
            )
        for row in result["selected"]:
            label = LABELS[row["candidate"], row["noise"]]
            scores = row["folds"][fold]["modalities"]
            text.append(
                "| "
                + label
                + " | "
                + " | ".join(f"{scores[m]['observation_rmse']:.3f}" for m in MODALITIES)
                + " |"
            )
            if row["noise"] == "ou":
                text.append(
                    "| "
                    + label
                    + ", shared only | "
                    + " | ".join(f"{scores[m]['shared_only_rmse']:.3f}" for m in MODALITIES)
                    + " |"
                )
    text.extend(
        [
            "",
            "![Training-selected refit validation](../assets/figures/gp-temporal-refit-scores.svg)",
            "",
            "## Matched NVIDIA benchmark",
            "",
            "All devices evaluate the same float64 gamma/OU point: 52,964 scalar observations, 841 functional nodes, three factors, and quadrature order 384. Three warm repetitions follow the first compilation-inclusive call.",
            "",
            "| Device | First gradient (s) | Median warm gradient (s) | Warm CPU ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    baseline = np.median(
        result["gradient_benchmarks"]["cpu"]["timings"]["warm_value_gradient_seconds"]
    )
    for row in result["gradient_benchmarks"].values():
        warm = np.median(row["timings"]["warm_value_gradient_seconds"])
        label = row["device_kind"]
        if label == "cpu":
            label += f" (8-core affinity, BLAS {row['execution']['openblas_num_threads']})"
        text.append(
            f"| {label} | {row['timings']['first_value_gradient_seconds']:.2f} | {warm:.3f} | {baseline / warm:.2f}× |"
        )
    scaling = result["six_factor_probe"]
    six = np.median(scaling["timings"]["warm_value_gradient_seconds"])
    three = np.median(
        result["gradient_benchmarks"]["3090"]["timings"]["warm_value_gradient_seconds"]
    )
    text.extend(
        [
            "",
            f"On the RTX 3090, increasing from three to six latent factors raises the grouped matrix dimension from 2,523 to 5,046 and the median warm gradient cost from {three:.3f} to {six:.3f} seconds ({six / three:.2f}×). The data and observed feature count are unchanged. The six-factor probe passes three directional derivative checks, but is not a converged fit or a predictive accuracy qualification.",
        ]
    )
    family = result["response_family_timing"]
    text.extend(
        [
            "",
            f"At identical initial physical coordinates on the RTX 3090, the median grouped gradient takes {np.median(family['gamma3_warm_seconds']):.3f} seconds for gamma and {np.median(family['gaussian_warm_seconds']):.3f} seconds for Gaussian responses. This fixed-order grouped calculation shows little timing benefit from changing the response family itself; it is a different cost comparison from the earlier state-space filter-bank studies.",
        ]
    )
    algebra = result["gaussian_algebra_timing"]
    state = np.median(algebra["state_space"]["warm_value_gradient_seconds"])
    grouped = np.median(algebra["grouped"]["warm_value_gradient_seconds"])
    text.extend(
        [
            "",
            f"Holding the Gaussian independent-noise model fixed on the RTX 3090, the production state-space gradient takes {state:.3f} seconds versus {grouped:.3f} seconds for grouped quadrature (**{state / grouped:.2f}×**). At this same physical point, objectives differ by {algebra['objective_error']:.2g} and gradients by at most {algebra['gradient_error']:.2g}, passing the declared gates. This qualifies the measured initial point; it is not a blanket backend ranking for longer recordings or other factor counts.",
        ]
    )
    text.extend(
        [
            "",
            "The separate single-start fit benchmark uses the same seeded initial point, data, priors, bounds, optimizer settings, and convergence gate. Pipeline time includes data/model preparation, compilation, timing and derivative checks, and complete optimization, measured after Python imports.",
            "Each fit process has eight-core CPU affinity and single-thread BLAS/OMP settings. The separate eight-thread BLAS probe changes the median warm CPU cost only slightly; this comparison does not claim the fastest possible CPU configuration.",
            "",
            "| Device | Pipeline (s) | Optimization (s) | Iterations | Objective | Projected gradient |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["complete_fit_benchmark"].values():
        text.append(
            f"| {row['device_kind']} | {row['pipeline_seconds']:.1f} | {row['optimizer_seconds']:.1f} | {row['iterations']} | {row['objective']:.6f} | {row['gradient']:.3g} |"
        )
    text.extend(
        [
            "",
            f"The measured complete-fit CPU/GPU time ratio is **{result['complete_fit_speed_ratio']:.2f}× on the RTX PRO 6000** and **{result['complete_fit_3090_speed_ratio']:.2f}× on the RTX 3090** for this model and workload. It does not establish a NUTS, Gibbs, or variational-inference speedup, or complete-fit scaling to higher factor counts.",
            "",
            "<!-- RESULTS:END -->",
        ]
    )
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/performance/2026-09-23-gp-temporal-refit-results.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/performance/2026-09-23-gp-temporal-refits.md")
    )
    args = parser.parse_args()
    result = build(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.report.parents[1] / "assets/figures")
    content = args.report.read_text()
    section = report_tables(result)
    if "<!-- RESULTS:START -->" in content:
        before, remainder = content.split("<!-- RESULTS:START -->", 1)
        _, after = remainder.split("<!-- RESULTS:END -->", 1)
        content = before + section + after
    else:
        content = content.replace(
            "## Protocol fixed before scoring", section + "\n\n## Protocol fixed before scoring"
        )
    args.report.write_text(content)
    print(
        json.dumps(
            dict(
                fits=len(result["fits"]),
                selected=len(result["selected"]),
                complete_fit_speed_ratio=result["complete_fit_speed_ratio"],
            )
        )
    )


if __name__ == "__main__":
    main()
