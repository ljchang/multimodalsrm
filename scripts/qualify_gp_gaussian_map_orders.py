"""Research smaller Gaussian banks at a saved MAP and empirical stress points.

Lower orders do NOT satisfy the production covariance bound. Preparation uses
an explicitly recorded relaxed research tolerance solely to measure the error.
The default reference, observations, priors, and physical coordinates stay fixed.
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from benchmark_gp_response_candidates import load_data
from gp_response_candidates import split_physiology
from gp_response_holdout import prepare, split_data
from investigate_gp_response_bounds import observation_summary
from qualify_gp_parallel_scaling import qualification_points

from multimodalsrm.bayesian import state_space_compact
from multimodalsrm.bayesian.map_conditioning import _projected_gradient
from multimodalsrm.bayesian.state_space_gaussian_rational import (
    _TRUE_MASS,
    RationalGaussianTemplate,
    _fit,
    qualify_impulse,
)


@contextmanager
def research_template(order):
    poles, residues = _fit(order)
    details = qualify_impulse(poles, residues)
    error = details["numerical_l1_bound"]
    template = RationalGaussianTemplate(order, poles, residues, error, _TRUE_MASS + error, details)
    original = state_space_compact.rational_template
    state_space_compact.rational_template = lambda requested: template
    try:
        yield template
    finally:
        state_space_compact.rational_template = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="empirical")
    parser.add_argument("--subjects", default="s001,s002")
    parser.add_argument("--parcels", type=int, default=5)
    parser.add_argument("--window", type=float, default=500)
    parser.add_argument("--orders", type=int, nargs="+", default=[16, 12])
    parser.add_argument("--research-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--fit",
        type=Path,
        default=Path("local_data/gp-response-holdout-2026-09-22/gaussian-pro6000.json"),
    )
    parser.add_argument(
        "--loader",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "local_data/empirical-eval-2026-09-22/emo_data.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = split_data(split_physiology(load_data(args)))
    model, reference = prepare(split, "gaussian")
    saved = json.loads(args.fit.read_text())
    assert saved["parameter_names"] == [list(n) for n in reference.names]
    assert saved["training"] == observation_summary(reference.base)
    points = qualification_points(reference, True) + [
        ("saved_MAP", np.asarray(saved["best"]["parameters"]))
    ]
    result = dict(
        status="prepared",
        scope="K=1, five parcels; same empirical training fold and physical priors",
        research_covariance_tolerance=args.research_tolerance,
        production_covariance_tolerance=model.covariance_tolerance,
        training=observation_summary(reference.base),
        reference_dimension=reference.base.response_state_space.dimension,
        reference_bound=reference.base.covariance_error_bound,
        gates=dict(
            objective_atol=1e-6,
            objective_rtol=1e-9,
            gradient_atol=1e-5,
            gradient_rtol=1e-7,
            map_absolute_gradient_difference=1e-4,
            map_projected_gradient=1e-3,
        ),
        reference=[],
        orders=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    refs = []
    for name, x in points:
        start = time.perf_counter()
        value, gradient = reference.value_gradient(x)
        refs.append((value, gradient))
        result["reference"].append(
            dict(
                name=name,
                objective=value,
                seconds=time.perf_counter() - start,
                projected_gradient=_projected_gradient(x, gradient, np.asarray(reference.bounds)),
            )
        )
        save()
    for order in args.orders:
        with research_template(order):
            _, other = prepare(
                split, "gaussian", research_covariance_tolerance=args.research_tolerance
            )
        assert other.names == reference.names and other.bounds == reference.bounds
        assert observation_summary(other.base) == result["training"]
        row = dict(
            order=order,
            dimension=other.base.response_state_space.dimension,
            covariance_bound=other.base.covariance_error_bound,
            points=[],
        )
        result["orders"].append(row)
        for (name, x), (v, g) in zip(points, refs):
            start = time.perf_counter()
            value, gradient = other.value_gradient(x)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError(f"order {order}, {name}")
            error = abs(gradient - g)
            scaled = error / (1e-5 + 1e-7 * abs(g))
            detail = dict(
                name=name,
                objective_absolute_difference=abs(value - v),
                gradient_max_absolute_difference=float(max(error)),
                gradient_max_scaled_error=float(max(scaled)),
                worst_scaled_parameter=reference.names[int(np.argmax(scaled))],
                projected_gradient=_projected_gradient(x, gradient, np.asarray(other.bounds)),
                seconds=time.perf_counter() - start,
                passed=bool(abs(value - v) <= 1e-6 + 1e-9 * abs(v) and max(scaled) <= 1),
            )
            if name == "saved_MAP":
                detail["map_local_gate_passed"] = bool(
                    max(error) <= 1e-4 and detail["projected_gradient"] <= 1e-3
                )
            row["points"].append(detail)
            save()
            print(json.dumps(dict(order=order, **detail)), flush=True)
    result["status"] = "finished"
    save()


if __name__ == "__main__":
    main()
