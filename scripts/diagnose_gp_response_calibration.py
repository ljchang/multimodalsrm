"""Training innovations and simple held-out baselines for saved research fits.

Parameters are fixed at the saved MAP. Whiten entire simultaneous observation
groups before inspecting temporal residual dependence. These are descriptive
checks using fitted parameters, not independent hypothesis tests.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_gp_response_candidates import load_data
from compare_gp_response_families import test_summary
from gp_response_candidates import split_physiology
from gp_response_holdout import prepare, split_data
from investigate_gp_response_bounds import observation_summary
from scipy.linalg import solve_triangular

from multimodalsrm.bayesian.state_space_grouped import event_system, statistics, update


def functional_predictions(problem, native, run):
    """Prior functional moments, before assimilating each observation node."""
    nodes = problem.grouped_systems[run]
    events, outputs = event_system(problem, native, run)
    precision, information, _, _, _ = statistics(problem, native, run)
    order = events.order
    dimension = problem.features * problem.response_state_space.dimension

    def step(carry, inputs):
        mean, covariance, _, _, prior_mean, prior_covariance = update(*carry, *inputs)
        response = inputs[2]
        C = jnp.kron(jnp.eye(problem.features), response[None])
        return (mean, covariance), (C @ prior_mean, C @ prior_covariance @ C.T)

    _, moments = jax.lax.scan(
        step,
        (jnp.zeros(dimension), jnp.eye(dimension)),
        (
            events.transition,
            events.process_covariance,
            outputs[nodes.modalities][order],
            precision[order],
            information[order],
        ),
    )
    undo = jnp.argsort(order)
    return moments[0][undo], moments[1][undo]


def innovations(problem, native):
    weights, offsets, noise, _, _ = map(np.asarray, problem.arrays(native))
    weights = weights.reshape(len(problem.keys), problem.features)
    collected = {key: [] for key in problem.keys}
    total = 0.0
    for run, system in problem.systems.items():
        evaluate = jax.jit(lambda x: functional_predictions(problem, x, run))
        means, covariances = map(np.asarray, jax.block_until_ready(evaluate(native)))
        nodes = problem.grouped_systems[run]
        ki, gi, _ = problem._packed[run]
        for node in range(len(nodes.times)):
            rows = np.flatnonzero(nodes.observation_nodes == node)
            W = weights[ki[rows]]
            S = W @ covariances[node] @ W.T + np.diag(noise[gi[rows]])
            L = np.linalg.cholesky((S + S.T) * 0.5)
            error = np.asarray(system.values)[rows] - offsets[ki[rows]] - W @ means[node]
            white = solve_triangular(L, error, lower=True)
            total += 0.5 * (
                2 * np.log(np.diag(L)).sum() + white @ white + len(rows) * np.log(2 * np.pi)
            )
            for row, value in zip(rows, white):
                key = problem.keys[int(ki[row])]
                collected[key].append((run, float(system.times[row]), float(value)))
    reference = float(problem.nll(native))
    np.testing.assert_allclose(total, reference, atol=1e-6, rtol=0)
    result = {}
    for group in problem.groups:
        values, correlations, pair_counts = [], [], []
        for key, records in collected.items():
            if key[:2] != group:
                continue
            for run in problem.systems:
                series = sorted((t, z) for r, t, z in records if r == run)
                if not series:
                    continue
                times, z = np.asarray(series).T
                values.extend(z)
                step = np.min(np.diff(times)) if len(times) > 1 else np.nan
                adjacent = np.isclose(np.diff(times), step, atol=1e-8, rtol=0)
                if adjacent.sum() >= 3:
                    a, b = z[:-1][adjacent], z[1:][adjacent]
                    if a.std() > 0 and b.std() > 0:
                        correlations.append(float(np.corrcoef(a, b)[0, 1]))
                        pair_counts.append(int(adjacent.sum()))
        z = np.asarray(values)
        result["/".join(group)] = dict(
            observations=len(z),
            mean=float(z.mean()),
            rms=float(np.sqrt(np.mean(z**2))),
            coverage_95=float(np.mean(abs(z) <= 1.95996398454)),
            lag_one_correlation_median=float(np.median(correlations)) if correlations else None,
            lag_one_correlation_range=[float(min(correlations)), float(max(correlations))]
            if correlations
            else None,
            features_with_lag_one_check=len(correlations),
            adjacent_pairs=sum(pair_counts),
            fitted_noise_variance=float(noise[problem.groups.index(group)]),
        )
    return dict(
        likelihood_absolute_difference=abs(total - reference),
        ordering="Full node innovation Cholesky whitening in original observation order; shifted event clock",
        scope="Training diagnostics at fitted parameters; adjacent native samples only, never across held-out or censored gaps",
        groups=result,
    )


def held_out_baselines(split, predictions):
    result = {}
    brain_times = {}
    for modality in split["blocks"]:
        observed, interpolated, carried = [], [], []
        offset = 0
        for subject, runs in split["testing"].items():
            for run, modalities in runs.items():
                if modality not in modalities:
                    continue
                ts = modalities[modality]
                train = split["common"][subject][run][modality]
                rows, features = np.nonzero(ts.mask)
                interpolation, left = np.empty(len(rows)), np.empty(len(rows))
                for feature in np.unique(features):
                    known = train.mask[:, feature]
                    t, y = train.times[known], train.values[known, feature]
                    select = features == feature
                    query = ts.times[rows[select]]
                    assert np.all(query > t.min()) and np.all(query < t.max())
                    interpolation[select] = np.interp(query, t, y)
                    left[select] = y[np.searchsorted(t, query) - 1]
                y = ts.values[rows, features]
                observed.extend(y)
                interpolated.extend(interpolation)
                carried.extend(left)
                if modality == "brain":
                    mu = predictions["brain_mean"][offset : offset + len(rows)]
                    for t in np.unique(ts.times[rows]):
                        take = ts.times[rows] == t
                        slot = brain_times.setdefault(float(t), [[], [], []])
                        slot[0].extend(y[take])
                        slot[1].extend(mu[take])
                        slot[2].extend(interpolation[take])
                offset += len(rows)
        y, linear, previous = map(np.asarray, (observed, interpolated, carried))
        result[modality] = dict(
            observations=len(y),
            training_mean_rmse=float(np.sqrt(np.mean(y**2))),
            linear_interpolation_rmse=float(np.sqrt(np.mean((y - linear) ** 2))),
            previous_observation_rmse=float(np.sqrt(np.mean((y - previous) ** 2))),
        )
    timeline = []
    for t, values in sorted(brain_times.items()):
        y, mu, linear = map(np.asarray, values)
        assert len(y) >= 100, "Export only brain aggregates over many features"
        timeline.append(
            dict(
                time=t,
                observations=len(y),
                observed_rms=float(np.sqrt(np.mean(y**2))),
                predicted_rms=float(np.sqrt(np.mean(mu**2))),
                model_rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
                linear_rmse=float(np.sqrt(np.mean((y - linear) ** 2))),
            )
        )
    return dict(modalities=result, brain_block_aggregates=timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    saved = json.loads(args.fit.read_text())
    settings = argparse.Namespace(**saved["arguments"])
    settings.loader = Path(settings.loader)
    split = split_data(split_physiology(load_data(settings)), fold=settings.fold)
    _, problem = prepare(split, settings.candidate, features=settings.features)
    assert observation_summary(problem.base) == saved["training"]
    assert test_summary(split["testing"]) == saved["testing"]
    point = np.asarray(saved["best"]["parameters"])
    np.testing.assert_allclose(
        problem.value_gradient(point)[0], saved["best"]["objective"], atol=1e-6, rtol=0
    )
    native = np.asarray(problem.native(point))
    result = dict(
        candidate=settings.candidate,
        features=settings.features,
        training=saved["training"],
        testing=saved["testing"],
    )
    result["innovations"] = innovations(problem.base, native)
    with np.load(args.fit.with_name(args.fit.stem + "-predictions.npz")) as predictions:
        result["baselines"] = held_out_baselines(split, predictions)
    result["status"] = "finished"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
