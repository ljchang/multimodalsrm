"""Time one cold GP MAP fit, including optional R initialization.

Run each configuration in a fresh process with the same thread limits, e.g.:
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python scripts/benchmark_r_initialization.py --case multifactor --r-init

Repeat with --no-r-init and several seeds. These small synthetic comparisons
measure optimizer behavior, not empirical recovery or hardware-wide speedups.
"""

import argparse
import json
import time
import warnings

import numpy as np

from multimodalsrm import Gaussian, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig


def configuration(case, seed):
    rng = np.random.default_rng(seed)
    factors = 1 if case == "gaussian" else 2
    width, lag = 0.3, 0.4
    frequencies = np.array([2 * np.pi / 8, 2 * np.pi / 13])[:factors]
    phases = rng.uniform(-np.pi, np.pi, factors)
    responses = {"reference": Response(Identity(), estimate=False, pooling="shared")}
    filters = {}
    if case == "gaussian":
        responses["signal"] = Response(
            Gaussian(width, lag),
            fixed={"width": width},
            bounds={"lag": (-0.2, 1.0)},
            pooling="shared",
        )
        filters = {"signal": {"lag": Prior.normal(lag, 0.3)}}
    data = {}
    for subject in ("a", "b", "c"):
        data[subject] = {"train": {}}
        for modality, dt in (("reference", 0.8), ("signal", 1.0)):
            times = np.arange(0, 32.01, dt)
            delay = lag if case == "gaussian" and modality == "signal" else 0
            latent = np.sin((times[:, None] - delay) * frequencies + phases)
            if delay:
                mass = np.sqrt(2) * np.pi**0.25 * np.sqrt(width)
                latent *= mass * np.exp(-0.5 * (width * frequencies) ** 2)
            loading = rng.normal(size=(factors, factors))
            values = latent @ loading + rng.normal(0, 0.15, (len(times), factors))
            mask = rng.uniform(size=values.shape) > 0.03
            data[subject]["train"][modality] = TimeSeries(values, times, mask)
    return data, dict(
        features=factors,
        anchor=("a", "reference", 0),
        responses=responses,
        priors=BayesianPriors(
            loading_sd=1.0,
            offset_sd=0.3,
            noise=Prior.lognormal(np.log(0.03), 0.7),
            filters=filters,
        ),
        length_scale=2.0,
        inference="map",
        random_state=seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gaussian", "multifactor"), default="gaussian")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--r-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--algebra", choices=("dense", "grouped", "state_space"), default="grouped")
    args = parser.parse_args()
    data, kwargs = configuration(args.case, args.seed)
    model = BayesianMultimodalSRM(
        **kwargs,
        linear_algebra=args.algebra,
        search=SearchConfig(starts=1, maxiter=1200, refine_maxiter=200, r_init=args.r_init),
    )
    with warnings.catch_warnings(record=True) as caught:
        started = time.perf_counter()
        model.fit(data)
        elapsed = time.perf_counter() - started
    diagnostic = model.map_diagnostics_
    initial_search = diagnostic.get("pre_refinement", diagnostic)
    refinement = diagnostic.get("refinement", {})
    print(
        json.dumps(
            dict(
                **vars(args),
                total_seconds=elapsed,
                objective=diagnostic["objective"],
                search_iterations=initial_search["iterations"],
                search_function_evaluations=initial_search["function_evaluations"],
                refinement_iterations=refinement.get("iterations", 0),
                optimizer_success=diagnostic["optimizer_success"],
                physical_projected_gradient=diagnostic["physical_projected_gradient"],
                meets_gradient_tolerance=diagnostic["meets_gradient_tolerance"],
                initialization=model.restart_diagnostics_[0].get("initialization"),
                warnings=[str(w.message) for w in caught],
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
