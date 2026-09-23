"""Check selected empirical predictions against the production per-feature API.

This checks the single-run response-family experiment. Raw fit and prediction
files stay local; the output contains only numerical differences and counts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_gp_response_candidates import load_data
from gp_response_candidates import split_physiology
from gp_response_holdout import prepare, split_data

from multimodalsrm.bayesian.prediction import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algebra", choices=("state_space", "grouped"), default="state_space")
    parser.add_argument("--order", type=int, default=384)
    parser.add_argument("--first-subject-only", action="store_true")
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text())
    settings = argparse.Namespace(**fit["arguments"])
    settings.loader = Path(settings.loader)
    split = split_data(split_physiology(load_data(settings)), fold=settings.fold)
    _, problem = prepare(
        split, settings.candidate, args.algebra, args.order, features=settings.features
    )
    tolerance = 1e-4 if args.algebra == "grouped" else 1e-8
    assert list(problem.base.systems) == ["run0"], "This check covers the single-run experiment"
    assert fit["parameter_names"] == [list(name) for name in problem.names]
    native = np.asarray(problem.native(np.asarray(fit["best"]["parameters"])))
    saved = np.load(args.fit.with_name(args.fit.stem + "-predictions.npz"))
    checks = []
    for modality in problem.base.modalities:
        offset = 0
        for subject, runs in split["testing"].items():
            if modality not in runs["run0"]:
                continue
            ts = runs["run0"][modality]
            rows, features = np.nonzero(ts.mask)
            selected = [0, ts.values.shape[1] - 1] if modality == "brain" else [0]
            for feature in sorted(set(selected)):
                take = features == feature
                query = ts.times[rows[take]]
                mean, variance = project(
                    problem.base,
                    native[None],
                    "run0",
                    query,
                    key=(subject, modality, feature),
                    include_noise=True,
                )
                detail = dict(
                    subject=subject,
                    modality=modality,
                    feature=feature,
                    observations=len(query),
                )
                for name, expected in (("mean", mean[0]), ("variance", variance[0])):
                    actual = saved[modality + "_" + name][offset + np.flatnonzero(take)]
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
                    detail[name + "_max_absolute_difference"] = float(
                        np.max(abs(actual - expected))
                    )
                checks.append(detail)
            offset += len(rows)
            if args.first_subject_only:
                break
    result = dict(
        status="finished",
        features=settings.features,
        selected_start=fit["best"]["start"],
        algebra=args.algebra,
        quadrature_order=args.order if args.algebra == "grouped" else None,
        absolute_tolerance=tolerance,
        checks=checks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
