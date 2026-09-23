"""Check reserved-fold isolation and cached predictions against dense conditioning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_gp_response_families import test_summary
from gp_response_holdout import BLOCK_FOLDS, CanonicalProblem, make_model, split_data
from probe_gp_correlated_noise import TIMESCALES
from score_gp_temporal_calibration import CachedPrediction

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian.fitting import initial_points
from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.problem import BayesianProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(64)
    data = {}
    for subject in ("a", "b"):
        streams = {}
        for i, modality in enumerate(("brain", "face", "rating", "eda")):
            times = np.r_[0, np.arange(160, 401, 10), 500] + i * 0.11
            values = rng.normal(size=(len(times), 3))
            mask = rng.uniform(size=values.shape) > 0.1
            streams[modality] = TimeSeries(values, times, mask)
        data[subject] = {"run0": streams}
    splits = {
        fold: split_data(data, fold=fold, excluded_folds=tuple(BLOCK_FOLDS)) for fold in BLOCK_FOLDS
    }
    original = splits["original"]
    for split in splits.values():
        assert test_summary(split["common"]) == test_summary(original["common"])
        for subject, runs in split["common"].items():
            for modality, train in runs["run0"].items():
                for other in splits.values():
                    assert not np.any(train.mask & other["testing"][subject]["run0"][modality].mask)
    assert (
        test_summary(split_data(data)["common"])["total"]
        > test_summary(original["common"])["total"]
    )
    model = make_model(original["training"], "gamma3", "grouped", 192, features=3)
    adapter, full = _prepare(model, original["training"])
    systems, _ = adapter._systems(original["common"], adapter.domains_)
    checks = []
    for noise in ({}, TIMESCALES, {"brain": 3.0}):
        kwargs = dict(
            systems=systems,
            anchor=full.anchor,
            reference_modality="brain",
            response_quadrature_order=192,
            noise_timescales=noise,
        )
        dense = BayesianProblem(adapter, model.priors, linear_algebra="dense", **kwargs)
        grouped = BayesianProblem(adapter, model.priors, linear_algebra="grouped", **kwargs)
        problem = CanonicalProblem(grouped, "gamma3")
        native = np.asarray(problem.native(initial_points(problem, 1, 722)[0])).copy()
        for i, name in enumerate(grouped.names):
            if name[:4] == ("loading", "b", "face", 1):
                native[i] = 0
        cached = CachedPrediction(grouped, native, "run0")
        for key in (("a", "brain", 0), ("b", "face", 1), ("b", "rating", 2), ("a", "eda", 0)):
            times = np.array([224.0, 245.5, 381.3])
            keys = np.full(len(times), grouped.keys.index(key))
            for noisy in (False, True):
                shared, mean, variance = cached.predict(
                    key[1], times, keys, include_noise=noisy, batch_size=2
                )
                expected_mean, expected_variance = project(
                    dense, native[None], "run0", times, key=key, include_noise=noisy
                )
                np.testing.assert_allclose(mean, expected_mean[0], atol=1e-8, rtol=0)
                np.testing.assert_allclose(variance, expected_variance[0], atol=1e-8, rtol=0)
                if not noisy or key[1] not in noise:
                    np.testing.assert_allclose(mean, shared, atol=1e-8, rtol=0)
                checks.append(
                    dict(
                        noise=noise,
                        key=key,
                        include_noise=noisy,
                        mean_error=float(max(abs(mean - expected_mean[0]))),
                        variance_error=float(max(abs(variance - expected_variance[0]))),
                    )
                )
        times = np.array([224.0, 245.5, 224.0, 381.3])
        query_keys = [("a", "brain", 0), ("b", "brain", 2)]
        keys = np.array([grouped.keys.index(query_keys[i]) for i in (0, 1, 1, 0)])
        _, mean, variance = cached.predict("brain", times, keys, batch_size=3)
        for key in query_keys:
            chosen = keys == grouped.keys.index(key)
            expected_mean, expected_variance = project(
                dense, native[None], "run0", times[chosen], key=key, include_noise=True
            )
            np.testing.assert_allclose(mean[chosen], expected_mean[0], atol=1e-8, rtol=0)
            np.testing.assert_allclose(variance[chosen], expected_variance[0], atol=1e-8, rtol=0)
            checks.append(
                dict(
                    noise=noise,
                    key=key,
                    kind="mixed_keys_and_unsorted_duplicate_times",
                    mean_error=float(max(abs(mean[chosen] - expected_mean[0]))),
                    variance_error=float(max(abs(variance[chosen] - expected_variance[0]))),
                )
            )
        if noise:
            key = ("a", "brain", 0)
            indices = [i for i, k in enumerate(grouped.systems["run0"].keys) if k == key]
            times = grouped.systems["run0"].times[indices]
            _, mean, variance = cached.predict(
                "brain", times, np.full(len(times), grouped.keys.index(key))
            )
            np.testing.assert_allclose(
                mean, grouped.systems["run0"].values[indices], atol=1e-8, rtol=0
            )
            np.testing.assert_allclose(variance, 0, atol=1e-8, rtol=0)
    result = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reserved_fold_isolation="passed including all-reserved-payload perturbation and affine invariance",
        checks=checks,
        max_mean_error=max(c["mean_error"] for c in checks),
        max_variance_error=max(c["variance_error"] for c in checks),
        noisy_training_interpolation="passed for OU, with feature masks and zero loadings exercised",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
