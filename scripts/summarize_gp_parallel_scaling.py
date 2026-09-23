"""Export completed parallel-filter qualification results and paired GPU plots."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASES = (
    "gamma3-k3",
    "gamma3-k3-functional",
    "gamma3-k6-functional",
    "gaussian-k3",
    "gaussian-k3-functional",
)


def update_report(root, result):
    timing = [
        "| Model / initialization | GPU | States | Sequential (s) | Parallel (s) | Speedup | Points passing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    costs = [
        "| Model / initialization | Compile sequential / parallel (s) | Memory sequential / parallel (GB) | Estimated break-even evaluations |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in result["cases"]:
        family = "Gamma 3" if row["candidate"] == "gamma3" else "Gaussian"
        method = "original" if row["observation_solve"] == "full" else "reduced solve"
        label = f"{family}, K={row['features']}, {method}"
        device = "RTX 3090" if "3090" in row["device"] else "RTX PRO 6000"
        p, c = row["points"][0], row["measurements"]
        timing.append(
            f"| {label} | {device} | {row['state_dimension']} | "
            f"{p['sequential']['warm_median_seconds']:.3f} | "
            f"{p['parallel']['warm_median_seconds']:.3f} | "
            f"{p['speedup']:.2f}x | {row['passed_points']}/{len(row['points'])} |"
        )
        compile_times = " / ".join(
            f"{c[k]['compile_seconds']:.1f}" for k in ("sequential", "parallel")
        )
        memory = " / ".join(
            f"{c[k]['compiler_memory']['estimated_live_bytes'] / 1e9:.2f}"
            for k in ("sequential", "parallel")
        )
        costs.append(
            f"| {label} | {compile_times} | {memory} | "
            f"{row['estimated_compile_break_even_evaluations']} |"
        )
    measurements = (
        "\n".join(timing)
        + "\n\n"
        + "\n".join(costs)
        + "\n\nBreak-even counts divide extra compilation/lowering time by the measured "
        "per-evaluation saving. They assume reuse of the same compiled function and "
        "are estimates, not measured fitting durations."
    )
    numerics = [
        "For the coherent low-noise synthetic draw, errors against the dense same-realization oracle are:",
        "",
        "| Fixture | Filter | Objective absolute error | Maximum scaled gradient error |",
        "| --- | --- | ---: | ---: |",
    ]
    for family in ("gamma", "gaussian"):
        fixture = result["checks"][family + "-small-numerics"]
        case = next(c for c in fixture["cases"] if c["coherent"])
        for method in ("sequential", "full", "functional"):
            e = case["backends"][method]["dense_same_realization"]
            label = {
                "sequential": "Sequential",
                "full": "Original parallel",
                "functional": "Reduced-solve parallel",
            }[method]
            numerics.append(
                f"| {family.title()}, D={fixture['state_dimension']} | {label} | "
                f"{e['objective_absolute_difference']:.3g} | {e['gradient_max_scaled_error']:.3g} |"
            )
    numerics.extend(
        [
            "",
            "A scaled gradient error greater than one fails the componentwise gate. "
            "The reduced solve passes this small Gaussian draw but still fails the gamma "
            "draw. These fixtures use the candidate configuration's native scale/lag "
            "bounds, so passing one does not establish full empirical-box qualification.",
        ]
    )
    path = root / "docs/performance/2026-09-22-gp-parallel-scaling.md"
    report = path.read_text()
    for key, content in (("MEASUREMENTS", measurements), ("NUMERICS", "\n".join(numerics))):
        start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        pattern = re.escape(start) + ".*?" + re.escape(end)
        report, count = re.subn(
            pattern, lambda _: start + "\n\n" + content + "\n\n" + end, report, flags=re.S
        )
        assert count == 1
    path.write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("local_data/gp-parallel-scaling-2026-09-22")
    )
    args = parser.parse_args()
    result = dict(
        scope="Research value/gradient evaluation; no new fits or held-out scores",
        protocol="500 seconds, two subjects, 100 brain parcels, common original-fold training rows",
        accuracy_gate="abs(delta objective) <= 1e-6 + 1e-9*abs(reference); each abs(delta gradient) <= 1e-5 + 1e-7*abs(reference component)",
        memory="Compiler-estimated live argument/output/temporary storage, not measured process or GPU peak",
        cases=[],
        checks={},
    )
    for name in CASES:
        raw = json.loads((args.directory / (name + ".json")).read_text())
        if raw["status"] != "finished":
            raise ValueError(f"incomplete case: {name}")
        row = {
            k: raw[k]
            for k in (
                "commit",
                "source_hashes",
                "python",
                "jax",
                "device",
                "state_dimension",
                "parameters",
                "training",
                "preparation_seconds",
                "tolerance",
                "measurements",
                "points",
                "all_passed",
            )
        }
        row.update(
            name=name,
            features=raw["arguments"]["features"],
            candidate=raw["arguments"]["candidate"],
            observation_solve=raw["arguments"].get("observation_solve", "full"),
        )
        row["passed_points"] = sum(p["passed"] for p in row["points"])
        initial = row["points"][0]
        cost = row["measurements"]
        saving = (
            initial["sequential"]["warm_median_seconds"]
            - initial["parallel"]["warm_median_seconds"]
        )
        extra_compile = (
            cost["parallel"]["compile_seconds"]
            + cost["parallel"]["lowering_seconds"]
            - cost["sequential"]["compile_seconds"]
            - cost["sequential"]["lowering_seconds"]
        )
        row["estimated_compile_break_even_evaluations"] = (
            max(0, int(np.ceil(extra_compile / saving))) if saving > 0 else None
        )
        result["cases"].append(row)
    assert len({r["training"]["sha256"] for r in result["cases"]}) == 1
    for name in ("gamma-small-numerics", "gaussian-small-numerics"):
        raw = json.loads((args.directory / (name + ".json")).read_text())
        if raw["status"] != "finished":
            raise ValueError(f"incomplete check: {name}")
        result["checks"][name] = raw
    for name in ("holdout-k1-check", "holdout-k3-check"):
        raw = json.loads((args.directory / (name + ".json")).read_text())
        result["checks"][name] = {
            k: raw[k]
            for k in (
                "features",
                "fold",
                "preprocessing_checks",
                "derivative_checks",
                "production_prediction_max_absolute_difference",
            )
        }
    result["checks"]["expanded_folds"] = json.loads(
        (args.directory / "expanded-folds.json").read_text()
    )
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/performance/2026-09-22-gp-parallel-scaling-results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    update_report(root, result)

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.0), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )
    labels = []
    colors = {"sequential": "#737b84", "parallel": "#007c91"}
    for i, row in enumerate(result["cases"]):
        family = "Gamma 3" if row["candidate"] == "gamma3" else "Gaussian"
        device = "RTX 3090" if "3090" in row["device"] else "RTX PRO 6000"
        method = "original" if row["observation_solve"] == "full" else "reduced solve"
        labels.append(
            f"{family}, K={row['features']} · {method}\n{device}, D={row['state_dimension']}"
        )
        point = row["points"][0]
        for ax, values in zip(
            axes,
            (
                [point[k]["warm_median_seconds"] for k in colors],
                [
                    row["measurements"][k]["compiler_memory"]["estimated_live_bytes"] / 1e9
                    for k in colors
                ],
            ),
        ):
            ax.plot(values, [i, i], color="#cad0d5", lw=2, zorder=1)
            for backend, value in zip(colors, values):
                ax.scatter(
                    value,
                    i,
                    color=colors[backend],
                    s=55,
                    zorder=2,
                    label=backend.title() if i == 0 else None,
                )
        axes[0].annotate(
            f"{point['speedup']:.2f}×",
            (min(point[k]["warm_median_seconds"] for k in colors), i),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Warm value + gradient (seconds)")
    axes[1].set_xlabel("Compiler-estimated live memory (GB)")
    for ax in axes:
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(y=0.18)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", bbox_to_anchor=(0.56, 0.93), ncol=2, frameon=False
    )
    fig.suptitle("Parallel filtering: paired GPU speed and memory", fontsize=14, x=0.56)
    fig.text(
        0.56,
        0.01,
        "57,671 training scalars · 942 time/modality nodes · float64\n"
        "Warm speedups do not establish numerical qualification or end-to-end fit speedups.",
        ha="center",
        fontsize=9,
        color="#525a63",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.89))
    stem = root / "docs/assets/figures/gp-parallel-scaling"
    for suffix in ("svg", "png"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180)
    plt.close(fig)
    for row in result["cases"]:
        point, cost = row["points"][0], row["measurements"]
        print(
            row["name"],
            {k: point[k]["warm_median_seconds"] for k in colors},
            "speedup",
            point["speedup"],
            "compile",
            {k: cost[k]["compile_seconds"] for k in colors},
            "passes",
            row["passed_points"],
            "break_even",
            row["estimated_compile_break_even_evaluations"],
        )


if __name__ == "__main__":
    main()
