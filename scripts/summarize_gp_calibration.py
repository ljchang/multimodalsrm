"""Export aggregate clock/calibration checks, excluding raw signal predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finished(path):
    result = json.loads(path.read_text())
    assert result["status"] == "finished", path
    return result


def noise_refinement(directory, family):
    coarse = finished(directory / f"{family}-ou-noise.json")
    fine = finished(directory / f"{family}-ou-noise-768.json")
    for key in ("candidate", "training", "testing", "noise_timescales"):
        assert coarse[key] == fine[key], key
    assert (coarse["quadrature_order"], fine["quadrature_order"]) == (384, 768)
    assert len(coarse["prediction_api_checks"]) == 4
    assert max(p["max_absolute_difference"] for p in coarse["prediction_api_checks"]) <= 1e-8
    differences = {}
    with (
        np.load(directory / f"{family}-ou-noise.npz") as a,
        np.load(directory / f"{family}-ou-noise-768.npz") as b,
    ):
        for modality in coarse["modalities"]:
            row = {}
            for kind in ("shared", "private"):
                key = f"{modality}_{kind}"
                row[kind] = float(np.max(abs(a[key] - b[key])))
            shared, private = f"{modality}_shared", f"{modality}_private"
            row["observation"] = float(np.max(abs(a[shared] + a[private] - b[shared] - b[private])))
            assert max(row.values()) <= 1e-4
            differences[modality] = row
    return dict(
        absolute_prediction_gate=1e-4,
        max_absolute_prediction_differences=differences,
        nll_absolute_difference=abs(
            coarse["nll_at_original_parameters"] - fine["nll_at_original_parameters"]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-directory", type=Path, required=True)
    parser.add_argument("--preserved-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    args = parser.parse_args()
    audit = finished(args.legacy_directory / "clock-audit.json")
    result = dict(
        scope="Clock metadata, aggregate innovations and held-out diagnostics; no individual signal arrays or loadings",
        clock_audit=audit,
        loader_preparation=finished(args.preserved_directory / "preparation-check.json"),
        processing_decision="Following the user's working interpretation, retain the first 252 original brain volumes, trim eight tail volumes, and treat splice flags as boundary markers; retain the nonzero censor mask.",
        legacy_compressed_clock={},
        preserved_row_clock={},
        preserved_clock_fixed_parameter_probes={},
        limitations=[
            "Clock evidence supports retaining original row times. The user's clarified working rule trims the final eight volumes and retains uncensored splice-marker rows.",
            "Complete acquisition timing is not independently reconstructed here; the first 252 original rows span 0 to 502 seconds.",
            "Sensitivity and OU probes hold parameters fixed, with no refits or new convergence claims, on both clocks.",
            "Innovation correlations use fitted parameters and a specified simultaneous-observation whitening order; these are descriptive diagnostics.",
            "All predictive diagnostics reuse the examined original validation fold, not an independent final test set.",
        ],
    )
    for family in ("gamma3", "gaussian"):
        legacy = {
            key: finished(args.legacy_directory / f"{family}-{suffix}.json")
            for key, suffix in (
                ("diagnostics", "diagnostics"),
                ("fixed_parameter_sensitivity", "sensitivity"),
                ("fixed_parameter_ou_noise", "ou-noise"),
            )
        }
        for key in ("training", "testing"):
            assert len({json.dumps(v[key], sort_keys=True) for v in legacy.values()}) == 1
        legacy["ou_quadrature_refinement"] = noise_refinement(args.legacy_directory, family)
        assert legacy["diagnostics"]["innovations"]["likelihood_absolute_difference"] <= 1e-6
        result["legacy_compressed_clock"][family] = legacy
        corrected = finished(args.preserved_directory / f"{family}-diagnostics.json")
        assert corrected["innovations"]["likelihood_absolute_difference"] <= 1e-6
        fit = finished(args.preserved_directory / f"{family}-k3.json")
        for key in ("training", "testing"):
            assert corrected[key] == fit[key]
        result["preserved_row_clock"][family] = corrected
        probes = {
            key: finished(args.preserved_directory / f"{family}-{suffix}.json")
            for key, suffix in (
                ("sensitivity", "sensitivity"),
                ("ou_noise", "ou-noise"),
            )
        }
        for probe in probes.values():
            for key in ("training", "testing"):
                assert probe[key] == fit[key]
        probes["ou_quadrature_refinement"] = noise_refinement(args.preserved_directory, family)
        result["preserved_clock_fixed_parameter_probes"][family] = probes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    colors = ("#126b91", "#bd6725", "#4d8646", "#865b9e")
    subjects = [(s, d) for s, d in audit["subjects"].items() if "pause_marker_alignment" in d]
    for (subject, data), color in zip(subjects, colors):
        row = data["pause_marker_alignment"]
        x = np.arange(1, row["boundaries"] + 1)
        ax.plot(x, row["original_row_clock_errors_seconds"], color=color, marker="o", label=subject)
        ax.plot(x, row["compressed_clock_errors_seconds"], color=color, ls="--", alpha=0.8)
    ax.axhline(0, color="#aaaaaa", lw=0.7)
    ax.set(
        xlabel="Movie pause / brain splice boundary",
        ylabel="Brain marker time − physiology pause time (s)",
        title="Original brain row times match independent pause markers\nSolid: original row times · dashed: renumbered after excluding prior splice rows",
        xticks=np.arange(1, 9),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(str(args.figure_prefix) + "." + suffix, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
