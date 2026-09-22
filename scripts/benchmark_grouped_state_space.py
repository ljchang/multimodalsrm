"""Compare complete fixed-response MAP fits and warm filtering/smoothing on CPU.

Run each backend in a fresh process with identical thread settings, for example:
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python scripts/benchmark_grouped_state_space.py --backend grouped --response bateman

The scalar arm uses the previous recurrence through a research-only patch of
the internal eligibility check. No public backend option or target is changed.
"""

import argparse
import json
import time
import warnings
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from multimodalsrm import BatemanSCR, Identity, Response, TimeSeries
from multimodalsrm.bayesian import BayesianMultimodalSRM, BayesianPriors, Prior, SearchConfig
from multimodalsrm.bayesian._backend import runtime
from multimodalsrm.bayesian.state_space import smoother


def dataset(args):
    rng = np.random.default_rng(args.seed)
    frequencies = np.array([2 * np.pi / 9, 2 * np.pi / 17])
    phase = rng.uniform(-np.pi, np.pi, 2)
    kernel = BatemanSCR(0.7, 2.0, 0.1) if args.response == "bateman" else Identity()
    responses = {
        "ref": Response(Identity(), estimate=False, pooling="shared"),
        "signal": Response(kernel, estimate=False, pooling="shared"),
    }
    times = np.r_[0.0, 94 + np.arange(args.times) * 1.2]
    data = {}
    for subject in ("a", "b"):
        data[subject] = {"train": {}}
        for modality, response in responses.items():
            t = times + (0.3 if modality == "signal" else 0)
            transfer = np.ones(2, complex)
            if isinstance(response.kernel, BatemanSCR):
                k = response.kernel
                transfer = np.exp(-1j * frequencies * k.lag) / (
                    (1 + 1j * frequencies * k.rise) * (1 + 1j * frequencies * k.decay) * k._energy
                )
            z = np.imag(np.exp(1j * (t[:, None] * frequencies + phase)) * transfer)
            weights = rng.normal(0, 0.7, (2, args.channels))
            values = z @ weights + rng.normal(0, 0.15, args.channels)
            values += rng.normal(0, 0.25, values.shape)
            mask = rng.uniform(size=values.shape) > 0.03
            data[subject]["train"][modality] = TimeSeries(values, t, mask)
    return data, responses


def measure(function, point, repeats=5):
    jax, _, _, _ = runtime()
    compiled = jax.jit(function)
    jax.block_until_ready(compiled(point))
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(point))
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("grouped", "scalar"), default="grouped")
    parser.add_argument("--response", choices=("identity", "bateman"), default="identity")
    parser.add_argument("--times", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    data, responses = dataset(args)
    model = BayesianMultimodalSRM(
        features=2,
        responses=responses,
        priors=BayesianPriors(noise=Prior.lognormal(np.log(0.0625), 0.7), offset_sd=0.3),
        inference="map",
        linear_algebra="state_space",
        length_scale=3.0,
        random_state=args.seed,
        max_observations=100_000,
        search=SearchConfig(
            starts=1, maxiter=600, refine_maxiter=200, ftol=1e-14, polish_max_parameters=0
        ),
    )
    context = (
        patch("multimodalsrm.bayesian.state_space_grouped.eligible", return_value=False)
        if args.backend == "scalar"
        else nullcontext()
    )
    with context, warnings.catch_warnings(record=True) as caught:
        start = time.perf_counter()
        model.fit(data)
        total = time.perf_counter() - start
    jax, _, _, _ = runtime()
    p = model.problem_
    # Identical physical probes in both arms, regardless of optimizer endpoint.
    point = p.initial.copy()
    rng = np.random.default_rng(42)
    for i, name in enumerate(p.names):
        if name[0] == "loading":
            point[i] = rng.normal(0, 0.7)
    value, gradient = p.value_gradient(point)
    query = np.linspace(94, 94 + (args.times - 1) * 1.2, 25)
    smooth = smoother(p, "train", query)
    moments = jax.jit(smooth)(point)
    record = model.map_diagnostics_
    first = record.get("pre_refinement", record)
    events = p.state_space_systems["train"]
    print(
        json.dumps(
            dict(
                **vars(args),
                total_fit_seconds=total,
                objective=model.objective_,
                physical_projected_gradient=record["physical_projected_gradient"],
                meets_gradient_tolerance=record["meets_gradient_tolerance"],
                search_iterations=first["iterations"],
                refinement_iterations=record.get("refinement", {}).get("iterations", 0),
                observations=len(p.systems["train"].times),
                filtering_events=len(events.times),
                transition_bytes=events.transition.nbytes + events.process_covariance.nbytes,
                warm_value_gradient_seconds=measure(jax.value_and_grad(p.objective), point),
                warm_smoother_seconds=measure(smooth, point),
                probe_objective=value,
                probe_gradient=gradient.tolist(),
                probe_mean=np.asarray(moments[0]).tolist(),
                probe_covariance=np.asarray(moments[1]).tolist(),
                warnings=[str(w.message) for w in caught],
            )
        )
    )


if __name__ == "__main__":
    main()
