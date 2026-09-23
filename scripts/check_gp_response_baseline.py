"""Verify a saved research MAP and plot its modality response curves.

Rebuild the same four-modality problem. Compare quadrature orders at the MAP,
check physical response derivatives including one-sided probes at bounds, and
retain initial versus fitted response curves. This is not held-out validation.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmark_gp_response_candidates import load_data
from gp_response_candidates import candidate_model, split_physiology

from multimodalsrm.bayesian.persistence import _prepare
from multimodalsrm.bayesian.quality import structured_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[96, 192])
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    settings = argparse.Namespace(**saved["arguments"])
    settings.loader = Path(settings.loader)
    data = split_physiology(load_data(settings))
    model = candidate_model(data, features=settings.features)
    _, state = _prepare(model, data)
    assert saved["parameter_names"] == [list(n) for n in state.names]
    x = np.load(args.fit.with_suffix(".npz"))["parameters"]
    value, gradient = state.value_gradient(x)

    def evaluate_value(point):
        return state.value_gradient(point)[0]

    result = dict(fit=str(args.fit), state_objective=float(value), derivatives=[], quadrature=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for i, n in enumerate(state.names):
        if n[0] != "filter":
            continue
        lo, hi = state.bounds[i]
        h = 1e-5 * max(abs(x[i]), 0.1)
        v = np.eye(len(x))[i] * h
        if x[i] - lo >= h and hi - x[i] >= h:
            fd = (evaluate_value(x + v) - evaluate_value(x - v)) / (2 * h)
            scheme = "central"
        else:
            sign = 1 if hi - x[i] >= 2 * h else -1
            fd = (
                sign
                * (-3 * value + 4 * evaluate_value(x + sign * v) - evaluate_value(x + 2 * sign * v))
                / (2 * h)
            )
            scheme = "one_sided_second_order"
        result["derivatives"].append(
            dict(
                parameter=list(n),
                value=float(x[i]),
                gradient=float(gradient[i]),
                finite_difference=float(fd),
                scheme=scheme,
                absolute_difference=float(abs(fd - gradient[i])),
            )
        )
    save()
    for order in args.orders:
        start = time.perf_counter()
        grouped_model = candidate_model(data, features=settings.features, algebra="grouped")
        grouped_model.set_params(response_quadrature_order=order)
        _, grouped = _prepare(grouped_model, data)
        assert grouped.names == state.names
        for run in state.systems:
            np.testing.assert_array_equal(grouped.systems[run].times, state.systems[run].times)
            np.testing.assert_array_equal(grouped.systems[run].values, state.systems[run].values)
            np.testing.assert_array_equal(grouped.systems[run].keys, state.systems[run].keys)
        target, score = grouped.value_gradient(x)
        row = dict(
            order=order,
            seconds=time.perf_counter() - start,
            objective=float(target),
            objective_absolute_difference=float(abs(target - value)),
            gradient_max_absolute_difference=float(np.max(abs(score - gradient))),
            gradient_relative_l2=float(
                np.linalg.norm(score - gradient) / max(np.linalg.norm(score), 1)
            ),
            filter_gradients={
                "/".join(n[1:]): float(score[i])
                for i, n in enumerate(state.names)
                if n[0] == "filter"
            },
        )
        result["quadrature"].append(row)
        save()
        print(json.dumps(row), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    result["responses"] = {}
    for ax, m, end in zip(axes.ravel(), ("brain", "face", "rating", "eda"), (35, 12, 18, 22)):
        response = model.responses[m]
        initial = response.initial_kernel()
        updates = {n[2]: float(x[i]) for i, n in enumerate(state.names) if n[:2] == ("filter", m)}
        fitted = initial.with_parameters(**updates)
        result["responses"][m] = dict(
            parameters=fitted.parameters, shape=structured_shape(fitted, 0.0)
        )
        t = np.linspace(-3, end, 1500)
        ax.plot(t, initial.evaluate(t), "--", color="#969696", label="Initial response")
        ax.plot(t, fitted.evaluate(t), color="#176b91", lw=2, label="Selected MAP")
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.set(
            title=f"{m.capitalize()} · {type(fitted).__name__}",
            xlabel="Time from model reference (s)",
            ylabel="L2-normalized response",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Four-modality response baseline · two subjects · one latent factor\nConstrained MAP curves; no uncertainty or held-out validation",
        fontsize=12,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)
    plt.close(fig)
    result["figure"] = str(args.figure)
    save()


if __name__ == "__main__":
    main()
