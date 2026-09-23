"""Cached grouped predictions for training-selected temporal calibration fits.

Keep all outputs under local_data: vectors include participant measurements.
This is conditional MAP smoothing, not forecasting or new-subject prediction.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from compare_gp_response_families import test_summary
from gp_response_holdout import BLOCK_FOLDS
from investigate_gp_response_bounds import observation_summary
from refit_gp_temporal_calibration import prepare, save

from multimodalsrm.bayesian.grouped import factor
from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.temporal_noise import (
    NoiseSystem,
    design_product,
    design_transpose,
    operands,
)


def parameter_hash(saved):
    return hashlib.sha256(
        np.asarray(saved["best"]["parameters"], dtype="<f8").tobytes()
    ).hexdigest()


class CachedPrediction:
    """Factor once and solve narrow batches, retaining all cross-factor terms."""

    def __init__(self, base, native, run):
        if base.linear_algebra != "grouped" or base.run_baseline_sd:
            raise ValueError("Expected grouped algebra without run baselines")
        self.base, self.native, self.run = base, np.asarray(native), run
        system = base.systems[run]
        noise_system = base.noise_systems.get(run) or NoiseSystem.prepare(
            system.times, system.keys, base.keys, {}
        )
        nodes = base.grouped_systems[run]
        count = len(nodes.times)

        def coefficients(x):
            _, q, _, lu = factor(base, x, run)
            K, w, residual, variance, indices = operands(base, x, run)
            alpha = noise_system.precision(
                residual - design_product(K @ q, w, indices, count), variance
            )
            return K, w, variance, q, lu, alpha

        K, w, variance, q, lu, alpha = jax.block_until_ready(jax.jit(coefficients)(native))
        weights, offsets, noise, _, _ = base.arrays(native)
        weights = weights.reshape(len(base.keys), base.features)
        groups = jnp.array([base.groups.index(key[:2]) for key in base.keys])
        indices = nodes.observation_nodes
        self.weights = np.asarray(weights)
        self.noise = np.asarray(noise)

        def batch(cross, query, keys, prior, noisy):
            loading = weights[keys]
            observed_cross = cross[:, indices] * (loading @ w.T)
            shared = jnp.sum((cross @ q.reshape(count, base.features)) * loading, axis=1)
            if base.noise_timescales:
                safe = np.where(noise_system.lengths > 0, noise_system.lengths, 1.0)
                residual_cross = (
                    jnp.exp(-jnp.abs(query[:, None] - system.times) / safe)
                    * ((keys[:, None] == noise_system.keys) & (noise_system.lengths > 0))
                    * variance
                )
                observed_cross += noisy * residual_cross
            rhs = observed_cross.T
            first = noise_system.precision(rhs, variance)
            score = design_transpose(first, w, indices, count)
            coefficient = jsp.linalg.lu_solve(lu, score, trans=1)
            inverse = first - noise_system.precision(
                design_product(K @ coefficient, w, indices, count), variance
            )
            prior_variance = prior * jnp.sum(loading**2, axis=1) + noisy * noise[groups[keys]]
            conditional = prior_variance - jnp.sum(rhs * inverse, axis=0)
            return (
                shared + offsets[keys],
                observed_cross @ alpha + offsets[keys],
                conditional,
                prior_variance,
            )

        self.batch = jax.jit(batch)
        self.temporal_cache = {}

    def predict(self, modality, query, keys, *, include_noise=True, batch_size=32):
        query, keys = np.asarray(query), np.asarray(keys, int)
        if len(query) != len(keys) or not len(query):
            raise ValueError("Expected nonempty aligned query times and feature keys")
        unique, lookup = np.unique(query, return_inverse=True)
        cache_key = (modality, tuple(unique))
        if cache_key not in self.temporal_cache:
            mi = np.full(len(unique), self.base.modalities.index(modality), dtype=int)
            nodes = self.base.grouped_systems[self.run]
            cross = self.base.temporal_covariance(
                self.native, unique, mi, nodes.times, nodes.modalities
            )
            prior = self.base.temporal_covariance(
                self.native, np.zeros(1), mi[:1], np.zeros(1), mi[:1]
            )[0, 0]
            self.temporal_cache[cache_key] = (cross, prior)
        cross, prior = self.temporal_cache[cache_key]
        outputs = [[], [], []]
        for start in range(0, len(query), batch_size):
            chosen = np.arange(start, min(start + batch_size, len(query)))
            padded = np.pad(chosen, (0, batch_size - len(chosen)), mode="edge")
            shared, mean, variance, scale = map(
                np.asarray,
                self.batch(
                    cross[lookup[padded]], query[padded], keys[padded], prior, include_noise
                ),
            )
            if not all(np.isfinite(a).all() for a in (shared, mean, variance)):
                raise FloatingPointError("Nonfinite grouped prediction")
            if np.any(variance < -1e-9 * np.maximum(1, abs(scale))):
                raise FloatingPointError("Materially negative grouped predictive variance")
            for destination, value in zip(outputs, (shared, mean, np.maximum(variance, 0))):
                destination.extend(value[: len(chosen)])
        return tuple(map(np.asarray, outputs))


def metrics(y, shared, mean, variance, interpolation):
    if not np.all(variance > 0):
        raise FloatingPointError("Held-out observation variance must be positive")
    error = y - mean
    return dict(
        observations=len(y),
        shared_only_rmse=float(np.sqrt(np.mean((y - shared) ** 2))),
        observation_rmse=float(np.sqrt(np.mean(error**2))),
        marginal_nlpd=float(np.mean(0.5 * (np.log(2 * np.pi * variance) + error**2 / variance))),
        coverage_95=float(np.mean(abs(error) <= 1.95996398454 * np.sqrt(variance))),
        private_mean_rms=float(np.sqrt(np.mean((mean - shared) ** 2))),
        zero_prediction_rmse=float(np.sqrt(np.mean(y**2))),
        interpolation_rmse=float(np.sqrt(np.mean((y - interpolation) ** 2))),
        unit_normal_nlpd=float(np.mean(0.5 * (np.log(2 * np.pi) + y**2))),
        unit_normal_coverage_95=float(np.mean(abs(y) <= 1.95996398454)),
        mean_predictive_sd=float(np.mean(np.sqrt(variance))),
    )


def score_fold(predictor, split, *, check_api=False, brain_flags=None):
    base, run = predictor.base, predictor.run
    records, arrays, checks = {}, {}, []
    for modality in base.modalities:
        observed, times, keys, interpolation, contaminated = [], [], [], [], []
        for subject, runs in split["testing"].items():
            if modality not in runs[run]:
                continue
            ts = runs[run][modality]
            train = split["common"][subject][run][modality]
            rows, features = np.nonzero(ts.mask)
            observed.extend(ts.values[rows, features])
            times.extend(ts.times[rows])
            keys.extend(base.keys.index((subject, modality, int(f))) for f in features)
            if modality == "brain" and brain_flags is not None:
                volumes = np.rint(ts.times[rows] / 2).astype(int)
                np.testing.assert_array_equal(ts.times[rows], volumes * 2)
                contaminated.extend(brain_flags[subject][volumes])
            baseline = np.empty(len(rows))
            for feature in np.unique(features):
                selected = features == feature
                valid = train.mask[:, feature]
                baseline[selected] = np.interp(
                    ts.times[rows[selected]], train.times[valid], train.values[valid, feature]
                )
            interpolation.extend(baseline)
        times, keys, observed, interpolation = map(
            np.asarray, (times, keys, observed, interpolation)
        )
        shared, mean, variance = predictor.predict(modality, times, keys)
        if check_api:
            selected_keys = [keys[0]]
            if modality == "brain":
                selected_keys.append(max(keys))
            for key in selected_keys:
                selected = keys == key
                reference = project(
                    base,
                    predictor.native[None],
                    run,
                    times[selected],
                    key=base.keys[key],
                    include_noise=True,
                )
                errors = []
                for actual, expected in zip((mean[selected], variance[selected]), reference):
                    np.testing.assert_allclose(actual, expected[0], atol=1e-7, rtol=0)
                    errors.append(float(np.max(abs(actual - expected[0]))))
                checks.append(
                    dict(key=base.keys[key], mean_error=errors[0], variance_error=errors[1])
                )
        records[modality] = metrics(observed, shared, mean, variance, interpolation)
        if contaminated:
            flagged = np.asarray(contaminated, bool)
            records[modality]["post_splice_strata"] = {}
            for name, chosen in (("flagged", flagged), ("unflagged", ~flagged)):
                records[modality]["post_splice_strata"][name] = (
                    metrics(*(a[chosen] for a in (observed, shared, mean, variance, interpolation)))
                    if chosen.any()
                    else dict(observations=0)
                )
        arrays.update(
            {
                f"{modality}_{name}": value
                for name, value in zip(
                    ("observed", "shared", "mean", "variance", "interpolation"),
                    (observed, shared, mean, variance, interpolation),
                )
            }
        )
        print(json.dumps({modality: records[modality]}), flush=True)
    return records, arrays, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=384)
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/Storage/Projects/multimodalsrm/data/EmotionPictures"),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing scores")
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    saved = json.loads(args.fit.read_text())
    selection = json.loads(args.selection.read_text())
    chosen = next(r for r in selection["selected"] if r["filename"] == args.fit.name)
    assert chosen["parameter_sha256"] == parameter_hash(saved)
    settings = argparse.Namespace(**saved["arguments"])
    settings.loader = Path(settings.loader)
    begun = time.perf_counter()
    split, _, problem = prepare(settings, order=args.order)
    assert observation_summary(problem.base) == saved["training"]
    assert test_summary(split["testing"]) == saved["testing"]
    assert [list(n) for n in problem.names] == saved["parameter_names"]
    native = np.asarray(problem.native(np.asarray(saved["best"]["parameters"])))
    assert list(problem.base.systems) == ["run0"]
    predictor = CachedPrediction(problem.base, native, "run0")
    brain_flags, metadata_hashes = {}, {}
    for subject in split["testing"]:
        path = next((args.metadata_root / subject / "brain").glob("*info.csv"))
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 260
        brain_flags[subject] = np.array([float(r["post_splice_contam"]) != 0 for r in rows[:252]])
        metadata_hashes[subject] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(
        fit=args.fit.name,
        parameter_sha256=parameter_hash(saved),
        selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quadrature_order=args.order,
        training=saved["training"],
        candidate=settings.candidate,
        noise=settings.noise,
        length_scale=settings.length_scale,
        scope="All three schedules reserved from these refits; same cohort, previously explored in earlier analyses",
        task="Three scoring schedules under one common union training mask, including overlapping multimodal gaps; conditional smoothing, not forecasting or separately refitted cross-validation folds",
        uncertainty="Conditional on MAP parameters; marginal observation variance includes private OU covariance",
        brain_metadata_sha256=metadata_hashes,
        post_splice_scope="Descriptive validation strata only; no separate masked refit or response-history reset",
        folds={},
        status="scoring",
    )
    save(args.output, result)
    raw = {}
    for fold in BLOCK_FOLDS:
        settings.fold = fold
        split, _, other = prepare(settings, order=args.order)
        assert observation_summary(other.base) == saved["training"]
        scores, arrays, checks = score_fold(
            predictor,
            split,
            check_api=fold == "rotated" and not args.skip_api_check,
            brain_flags=brain_flags,
        )
        result["folds"][fold] = dict(
            testing=test_summary(split["testing"]), modalities=scores, prediction_api_checks=checks
        )
        raw.update({f"{fold}_{key}": value for key, value in arrays.items()})
        save(args.output, result)
    np.savez_compressed(args.output.with_suffix(".npz"), **raw)
    result.update(status="finished", seconds=time.perf_counter() - begun)
    save(args.output, result)


if __name__ == "__main__":
    main()
