"""Fixed-parameter OU-noise prediction diagnostic for a saved response fit.

Timescales are a common, rounded training-innovation diagnostic choice. No
parameters are refitted and this examined fold is not a model-selection test.
Report shared-signal and noisy-observation means separately; private residual
interpolation must not be mistaken for improved shared representation.
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
from gp_response_holdout import CanonicalProblem, make_model, split_data
from investigate_gp_response_bounds import observation_summary

from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.prediction import project
from multimodalsrm.bayesian.problem import BayesianProblem
from multimodalsrm.bayesian.temporal_noise import factor, operands

TIMESCALES = dict(brain=3.0, face=1.5, rating=15.0, eda=5.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=384)
    parser.add_argument(
        "--skip-api-check", action="store_true", help="For a separate quadrature-refinement run"
    )
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    saved = json.loads(args.fit.read_text())
    settings = argparse.Namespace(**saved["arguments"])
    settings.loader = Path(settings.loader)
    split = split_data(split_physiology(load_data(settings)), fold=settings.fold)
    model = make_model(
        split["training"], settings.candidate, "grouped", args.order, features=settings.features
    )
    adapter, full = _prepare(model, split["training"])
    systems, _ = adapter._systems(split["common"], adapter.domains_)
    base = BayesianProblem(
        adapter,
        model.priors,
        systems=systems,
        anchor=full.anchor,
        reference_modality="brain",
        linear_algebra="grouped",
        response_quadrature_order=args.order,
        noise_timescales=TIMESCALES,
    )
    problem = CanonicalProblem(base, settings.candidate)
    assert observation_summary(base) == saved["training"]
    assert test_summary(split["testing"]) == saved["testing"]
    assert [list(n) for n in problem.names] == saved["parameter_names"]
    assert list(base.systems) == ["run0"]
    native = np.asarray(problem.native(np.asarray(saved["best"]["parameters"])))
    run = "run0"

    def coefficients(x):
        value, q, _, _ = factor(base, x, run)
        K, w, residual, variance, nodes = operands(base, x, run)
        mean = (K @ q).reshape(-1, base.features)
        error = residual - jnp.sum(w * mean[nodes], axis=1)
        alpha = base.noise_systems[run].precision(error, variance)
        return value, q, alpha

    value, q, alpha = map(np.asarray, jax.block_until_ready(jax.jit(coefficients)(native)))
    weights, offsets, noise, _, _ = map(np.asarray, base.arrays(native))
    weights = weights.reshape(len(base.keys), base.features)
    result = dict(
        candidate=settings.candidate,
        noise_timescales=TIMESCALES,
        quadrature_order=args.order,
        training=saved["training"],
        testing=saved["testing"],
        nll_at_original_parameters=float(value),
        scope="No refit, no convergence claim, diagnostic use of an already examined validation fold",
        interpretation="Observation means add private OU residual interpolation to the shared functional mean",
        modalities={},
        prediction_api_checks=[],
        status="scoring",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    nodes = base.grouped_systems[run]
    ki, _, _ = base._packed[run]
    raw = {}
    for modality in base.modalities:
        query = np.unique(
            np.concatenate(
                [
                    runs[run][modality].times[runs[run][modality].mask.any(axis=1)]
                    for runs in split["testing"].values()
                    if modality in runs[run]
                ]
            )
        )
        mi = np.full(len(query), base.modalities.index(modality), dtype=int)
        cross = np.asarray(
            base.temporal_covariance(native, query, mi, nodes.times, nodes.modalities)
        )
        functional = cross @ q.reshape(len(nodes.times), base.features)
        observed, shared, private = [], [], []
        checked = False
        for subject, runs in split["testing"].items():
            if modality not in runs[run]:
                continue
            ts = runs[run][modality]
            rows, features = np.nonzero(ts.mask)
            lookup = np.searchsorted(query, ts.times[rows])
            keys = np.array([base.keys.index((subject, modality, int(f))) for f in features])
            mu = np.einsum("ik,ik->i", functional[lookup], weights[keys]) + offsets[keys]
            residual_mean = np.zeros(len(rows))
            for key in np.unique(keys):
                selected = keys == key
                training_rows = np.flatnonzero(ki == key)
                variance = noise[base.groups.index((subject, modality))]
                covariance = variance * np.exp(
                    -abs(ts.times[rows[selected], None] - base.systems[run].times[training_rows])
                    / TIMESCALES[modality]
                )
                residual_mean[selected] = covariance @ alpha[training_rows]
            if not checked and not args.skip_api_check:
                chosen = features == 0
                reference, _ = project(
                    base,
                    native[None],
                    run,
                    ts.times[rows[chosen]],
                    key=(subject, modality, 0),
                    include_noise=True,
                )
                error = float(np.max(abs(mu[chosen] + residual_mean[chosen] - reference[0])))
                np.testing.assert_allclose(
                    mu[chosen] + residual_mean[chosen], reference[0], atol=1e-8, rtol=0
                )
                result["prediction_api_checks"].append(
                    dict(subject=subject, modality=modality, max_absolute_difference=error)
                )
                checked = True
            observed.extend(ts.values[rows, features])
            shared.extend(mu)
            private.extend(residual_mean)
        y, mu, residual = map(np.asarray, (observed, shared, private))
        result["modalities"][modality] = dict(
            observations=len(y),
            shared_only_rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
            observation_rmse=float(np.sqrt(np.mean((y - mu - residual) ** 2))),
            private_mean_rms=float(np.sqrt(np.mean(residual**2))),
        )
        raw.update({modality + "_shared": mu, modality + "_private": residual})
        save()
        print(json.dumps({modality: result["modalities"][modality]}), flush=True)
    np.savez(args.output.with_suffix(".npz"), **raw)
    result["status"] = "finished"
    save()


if __name__ == "__main__":
    main()
