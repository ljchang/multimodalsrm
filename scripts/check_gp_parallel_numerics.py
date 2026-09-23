"""Compare prefix filters to a dense covariance oracle on a small response fixture.

The dense oracle uses the identical response realization, isolating filtering
error from response approximation error. It is not a quadrature qualification.
Both arbitrary residuals and a coherent low-noise GP draw are checked.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from benchmark_gp_parallel_filter import parallel_nll
from benchmark_gp_response_candidates import synthetic_data
from gp_response_candidates import candidate_model, split_physiology

from multimodalsrm.bayesian.fitting import initial_points
from multimodalsrm.bayesian.persistence import _prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("gamma", "gaussian"), default="gamma")
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--implementation", choices=("parallel", "block"), default="parallel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("float64 required")
    data = split_physiology(synthetic_data())
    _, p = _prepare(candidate_model(data, family=args.family, features=args.features), data)
    initial = initial_points(p, 1, 722)[0]

    def dense(x):
        _, offsets, _, _, _ = p.arrays(x)
        loss = -p.log_prior(x)
        for run, system in p.systems.items():
            y = jnp.asarray(system.values) - offsets[p._packed[run][0]]
            L = jnp.linalg.cholesky(p.covariance(x, run))
            z = jsp.linalg.solve_triangular(L, y, lower=True)
            loss += 0.5 * (z @ z + len(y) * np.log(2 * np.pi)) + jnp.log(jnp.diag(L)).sum()
        return loss

    functions = {"dense_same_realization": dense, "sequential": p.objective}
    if args.implementation == "parallel":
        for method in ("full", "functional"):
            functions[method] = lambda x, method=method: (
                sum(parallel_nll(p, x, run, observation_solve=method) for run in p.systems)
                - p.log_prior(x)
            )
    else:
        from gp_block_filter import nll as block_nll

        for method in ("dense", "compact"):
            functions[method] = lambda x, method=method: (
                sum(block_nll(p, x, run, joseph=method) for run in p.systems) - p.log_prior(x)
            )
    result = dict(
        implementation=args.implementation,
        family=args.family,
        features=args.features,
        state_dimension=p.features * p.response_state_space.dimension,
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for coherent in (False, True):
        if coherent:
            point = initial.copy()
            point[[i for i, n in enumerate(p.names) if n[0] == "noise"]] = 1e-6
            offsets = np.asarray(p.arrays(point)[1])
            rng = np.random.default_rng(824)
            for run, system in p.systems.items():
                L = np.linalg.cholesky(p.covariance(point, run))
                system.values[:] = L @ rng.normal(size=len(L)) + offsets[p._packed[run][0]]
        # Retrace after replacing observations, which are closed-over constants.
        evaluated = {
            k: jax.jit(jax.value_and_grad(lambda x, f=f: f(x))) for k, f in functions.items()
        }
        for variance in (0.3, 1e-3, 1e-6) if not coherent else (1e-6,):
            x = initial.copy()
            x[[i for i, n in enumerate(p.names) if n[0] == "noise"]] = variance
            row = dict(coherent=coherent, variance=variance, backends={})
            references = {}
            for name, fn in evaluated.items():
                value, grad = jax.block_until_ready(fn(x))
                value, grad = float(value), np.asarray(grad)
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError(name)
                references[name] = (value, grad)
                actual = dict(objective=value, gradient_max=float(max(abs(grad))))
                for ref in ("dense_same_realization", "sequential"):
                    if ref in references:
                        v, g = references[ref]
                        actual[ref] = dict(
                            objective_absolute_difference=abs(value - v),
                            gradient_max_absolute_difference=float(max(abs(grad - g))),
                            gradient_max_scaled_error=float(
                                max(abs(grad - g) / (1e-5 + 1e-7 * abs(g)))
                            ),
                        )
                row["backends"][name] = actual
            result["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
    result["status"] = "finished"
    save()


if __name__ == "__main__":
    main()
