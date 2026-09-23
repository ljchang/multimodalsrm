"""Fixed-parameter timescale and conditioning probes, not refitted model rankings.

Keep all preprocessing, held-out rows, loadings, responses, and observation
variances fixed. Change only the declared GP timescale or conditioning streams.
This deliberately reuses an examined validation fold for diagnosis.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from benchmark_gp_response_candidates import load_data
from compare_gp_response_families import score, test_summary
from gp_response_candidates import split_physiology
from gp_response_holdout import CanonicalProblem, make_model, prepare, split_data
from investigate_gp_response_bounds import observation_summary

from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.problem import BayesianProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    saved = json.loads(args.fit.read_text())
    settings = argparse.Namespace(**saved["arguments"])
    settings.loader = Path(settings.loader)
    split = split_data(split_physiology(load_data(settings)), fold=settings.fold)
    _, original = prepare(split, settings.candidate, features=settings.features)
    assert saved["training"] == observation_summary(original.base)
    assert saved["testing"] == test_summary(split["testing"])
    assert saved["parameter_names"] == [list(n) for n in original.names]
    point = np.asarray(saved["best"]["parameters"])
    result = dict(
        candidate=settings.candidate,
        scope="Fixed-parameter diagnostic; no refitting, no independent test set, no new convergence claim",
        training=saved["training"],
        testing=saved["testing"],
        cases=[dict(name="original", length_scale=3.0, conditioning="all", scores=saved["scores"])],
        status="prepared",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads(args.output.read_text())
        for field in ("candidate", "training", "testing", "scope"):
            assert previous[field] == result[field], field
        assert previous["cases"][0] == result["cases"][0]
        result = previous

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    for name, length, conditioning in [
        ("length_10", 10.0, "all"),
        ("length_30", 30.0, "all"),
        ("brain_only", 3.0, "brain"),
    ]:
        if any(case["name"] == name for case in result["cases"]):
            continue
        result["status"] = "preparing_" + name
        save()
        model = make_model(split["training"], settings.candidate, features=settings.features)
        model.set_params(length_scale=length)
        adapter, full = _prepare(model, split["training"])
        selected = split["common"]
        if conditioning != "all":
            selected = {
                s: {r: {conditioning: mods[conditioning]} for r, mods in runs.items()}
                for s, runs in selected.items()
            }
        systems, _ = adapter._systems(selected, adapter.domains_)
        base = BayesianProblem(
            adapter,
            model.priors,
            systems=systems,
            anchor=full.anchor,
            reference_modality="brain",
            linear_algebra="state_space",
        )
        problem = CanonicalProblem(base, settings.candidate)
        assert problem.names == original.names
        training = observation_summary(base)
        if conditioning == "all":
            assert training == saved["training"]
        row = dict(
            name=name,
            length_scale=length,
            conditioning=conditioning,
            training=training,
            covariance_error_bound=base.covariance_error_bound,
        )
        result["status"] = "scoring_" + name
        save()
        row["scores"] = score(
            problem,
            point,
            split["testing"],
            args.output.with_name(args.output.stem + "-" + name + ".npz"),
        )
        result["cases"].append(row)
        save()
        print(json.dumps(row), flush=True)
    result["status"] = "finished"
    save()


if __name__ == "__main__":
    main()
