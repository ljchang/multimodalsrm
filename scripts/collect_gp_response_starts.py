"""Collect independent starts, selecting a fit by training objective only.

Inputs and output contain fitted parameters and belong in ignored local_data.
Scoring and independent numerical validation run on the selected result later.
"""

import argparse
import copy
import json
from pathlib import Path


def collect(paths):
    fits = [json.loads(path.read_text()) for path in paths]
    first = fits[0]
    expected = first["arguments"]["starts"]
    assert len(fits) == expected, "Supply every predeclared start"
    for fit in fits:
        assert fit["status"] == "fit_finished", "Starts must finish before selection"
        assert len(fit["records"]) == 1
        assert fit["best"] == fit["records"][0]
        assert "scores" not in fit, "Select starts before looking at held-out scores"
        for field in (
            "training",
            "testing",
            "parameter_names",
            "holdout_blocks",
            "width_bounds",
            "peak_bounds",
            "response_priors",
            "source_hashes",
            "covariance_error_bound",
        ):
            assert fit[field] == first[field], field
        for field in ("candidate", "starts", "seed", "features", "fold", "parcels", "window"):
            assert fit["arguments"][field] == first["arguments"][field], field
        assert fit["records"][0]["start"] == fit["arguments"]["start_index"]
    assert sorted(fit["records"][0]["start"] for fit in fits) == list(range(expected))
    selected = min(fits, key=lambda fit: fit["records"][0]["objective"])
    result = copy.deepcopy(selected)
    result.pop("checkpoint", None)
    result["records"] = sorted(
        [copy.deepcopy(fit["records"][0]) for fit in fits], key=lambda row: row["start"]
    )
    result["arguments"]["start_index"] = None
    result["independent_workers"] = [
        dict(
            source=str(path),
            start=fit["records"][0]["start"],
            device=fit["device_kind"],
            timings=fit["timings"],
            fit_pipeline_seconds_after_imports=fit["fit_pipeline_seconds_after_imports"],
            derivative_checks=fit["derivative_checks"],
        )
        for path, fit in zip(paths, fits)
    ]
    result["selection"] = {
        "rule": "Lowest training objective across all predeclared starts; no held-out scoring",
        "all_starts_qualified": all(r["meets_gradient_tolerance"] for r in result["records"]),
        "timing_scope": "Top-level timing/device fields describe the selected worker only; workers overlap",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing collected or scored fit")
    result = collect(args.inputs)
    result["arguments"]["output"] = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                start=result["best"]["start"],
                objective=result["best"]["objective"],
                qualified=result["best"]["meets_gradient_tolerance"],
                **result["selection"],
            )
        )
    )


if __name__ == "__main__":
    main()
