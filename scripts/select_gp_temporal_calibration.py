"""Lock all training-only timescale choices before any refit holdout scoring."""

import argparse
import datetime
import json
from pathlib import Path

from score_gp_temporal_calibration import parameter_hash


def select(directory, *, require_numerical_checks=True):
    records, selected, training, testing = [], [], None, None
    common_settings = None
    for candidate in ("gamma3", "gaussian"):
        for noise in ("independent", "ou"):
            choices = []
            for length_scale in (3, 10, 30):
                path = directory / f"{candidate}-{noise}-{length_scale}.json"
                fit = json.loads(path.read_text())
                args = fit["arguments"]
                settings = {
                    k: args[k]
                    for k in (
                        "seed",
                        "features",
                        "subjects",
                        "window",
                        "parcels",
                        "fold",
                        "order",
                        "loader",
                    )
                }
                common_settings = settings if common_settings is None else common_settings
                assert settings == common_settings, "Mismatched experiment settings"
                assert args["candidate"] == candidate and args["noise"] == noise
                assert args["length_scale"] == length_scale
                assert fit["status"] == "fit_finished"
                assert args["starts"] == 3 and args["start_index"] is None
                assert sorted(r["start"] for r in fit["records"]) == [0, 1, 2]
                assert fit["best"] == min(fit["records"], key=lambda r: r["objective"])
                assert set(fit["reserved_folds"]) == {"original", "rotated", "late"}
                assert "scores" not in fit and "folds" not in fit
                training = fit["training"] if training is None else training
                testing = fit["testing"] if testing is None else testing
                assert fit["training"] == training and fit["testing"] == testing
                row = dict(
                    filename=path.name,
                    candidate=candidate,
                    noise=noise,
                    length_scale=length_scale,
                    objective=fit["best"]["objective"],
                    start=fit["best"]["start"],
                    qualified=fit["best"]["meets_gradient_tolerance"],
                    parameter_sha256=parameter_hash(fit),
                    all_starts_qualified=all(r["meets_gradient_tolerance"] for r in fit["records"]),
                )
                records.append(row)
                choices.append((row, fit))
            chosen, fit = min(choices, key=lambda pair: pair[0]["objective"])
            assert chosen["qualified"], (
                "Lowest-objective fit must qualify; do not substitute another"
            )
            if require_numerical_checks:
                assert any(
                    c["order"] >= 768
                    and c["objective_absolute_difference"] <= 1e-3
                    and c["gradient_max_absolute_difference"] <= 1e-4
                    for c in fit.get("quadrature_checks", [])
                ), f"Missing/pending quadrature gate: {chosen['filename']}"
            selected.append(chosen)
    return dict(
        rule="Lowest normalized training MAP objective across all three timescales and all three seeded starts within each response-family/noise cell",
        interpretation="Discrete training profile, not marginal likelihood or independent test-set model selection",
        training=training,
        records=records,
        selected=selected,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to change a locked selection")
    result = select(args.directory, require_numerical_checks=args.output is not None)
    if args.output is not None:
        result["locked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
