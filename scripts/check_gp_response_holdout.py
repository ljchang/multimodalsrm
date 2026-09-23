"""Check held-out preprocessing, canonical derivatives, and batched predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from compare_gp_response_families import score
from gp_response_holdout import BLOCK_FOLDS, prepare, split_data

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.fitting import initial_points
from multimodalsrm.bayesian.prediction import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--fold", choices=BLOCK_FOLDS, default="original")
    args = parser.parse_args()
    rng = np.random.default_rng(64)
    data = {}
    for s in ("a", "b"):
        streams = {}
        for i, m in enumerate(("brain", "face", "rating", "eda")):
            times = np.r_[0, np.arange(160, 401, 10), 500] + i * 0.11
            values = rng.normal(size=(len(times), 2))
            streams[m] = TimeSeries(values, times)
        data[s] = {"run0": streams}
    split = split_data(data, fold=args.fold)
    _, problem = prepare(split, "gamma3", features=args.features)
    point = initial_points(problem, 1, 722)[0]
    value, gradient = problem.value_gradient(point)
    checks = []
    for i, name in enumerate(problem.names):
        if name[0] != "filter":
            continue
        v = np.eye(len(point))[i] * 1e-5
        finite = (
            problem.value_gradient(point + v)[0] - problem.value_gradient(point - v)[0]
        ) / 2e-5
        np.testing.assert_allclose(finite, gradient[i], atol=1e-5, rtol=1e-5)
        checks.append(dict(parameter=name, absolute_difference=float(abs(finite - gradient[i]))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = args.output.with_suffix(".npz")
    metrics = score(problem, point, split["testing"], predictions)
    saved = np.load(predictions)
    native = np.asarray(problem.native(point))
    errors = []
    for m, ts in split["testing"]["a"]["run0"].items():
        rows = ts.mask[:, 0]
        times = ts.times[rows]
        mean, variance = project(
            problem.base, native[None], "run0", times, key=("a", m, 0), include_noise=True
        )
        for field, reference in (("mean", mean[0]), ("variance", variance[0])):
            actual = saved[m + "_" + field][: 2 * len(times) : 2]
            np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=1e-8)
            errors.append(float(np.max(abs(actual - reference))))
    result = dict(
        features=args.features,
        fold=args.fold,
        preprocessing_checks="held-out perturbation and affine invariance passed",
        derivative_checks=checks,
        production_prediction_max_absolute_difference=max(errors),
        fixture_metrics=metrics,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                derivative_max_absolute_difference=max(c["absolute_difference"] for c in checks),
                prediction_max_absolute_difference=max(errors),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
