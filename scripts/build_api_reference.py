"""Generate the public API inventory without importing numerical dependencies.

Run with --check in CI to detect stale signatures and defaults. Explanations and
supported combinations live in the hand-written choices and kernel guides.
"""

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/multimodalsrm"
PAGES = {
    "estimators": ("Estimators", ["MultimodalSRM", "BayesianMultimodalSRM"]),
    "responses": (
        "Responses and priors",
        [
            "Response",
            "Identity",
            "Gaussian",
            "Gamma",
            "DoubleGamma",
            "BachSCR",
            "BatemanSCR",
            "SampledKernel",
            "Normal",
            "KernelPrior",
            "Prior",
            "BayesianPriors",
        ],
    ),
    "configuration": (
        "Search and sampling configuration",
        ["SearchConfig", "SamplerConfig", "SpectralConfig", "ParameterSubspace", "BayesianProblem"],
    ),
    "data-results": (
        "Data and results",
        [
            "TimeSeries",
            "SeriesResult",
            "PosteriorSeriesResult",
            "KernelEstimate",
            "GaussianMixtureSeries",
            "TrajectorySamples",
            "ParticipantCalibration",
            "CrossValidationResult",
            "ModelComparisonResult",
            "ModelSelectionResult",
            "NestedCrossValidationResult",
        ],
    ),
    "evaluation": (
        "Evaluation and model selection",
        [
            "LeaveOneRunOut",
            "cross_validate",
            "compare_models",
            "select_model",
            "nested_cross_validate",
            "temporal_isc",
            "time_segment_matching",
        ],
    ),
}
INTRO = {
    "estimators": "R-MSRM and GP-MSRM have different inference and prediction contracts. Start with the [choice guide](../api-guide.md); use [capabilities](../capabilities.md) for supported combinations. Inherited scikit-learn `get_params` and `set_params` follow the standard estimator interface.",
    "responses": "A kernel describes a curve; `Response` determines what is learned and shared. Read the [illustrated kernel guide](../temporal-kernels.md) before selecting a shape. `Normal` and `KernelPrior` are R penalties; GP uses `Prior` inside `BayesianPriors` instead.",
    "configuration": "These objects are imported from `multimodalsrm.bayesian`. `SearchConfig` controls MAP optimization; `SamplerConfig` controls NUTS sampling. `BayesianProblem` and `ParameterSubspace` are advanced, lower-level interfaces. See [configuration choices](../api-guide.md#search-and-sampling) before changing budgets or conditional targets.",
    "data-results": "Arrays use time × feature (or time × factor). A `valid` mask means a query is supported, not that it was directly observed. `GaussianMixtureSeries` contains marginal distributions; `TrajectorySamples` contains joint paths. Result dataclass fields are listed below, including inherited fields.",
    "evaluation": "The cross-validation and selection helpers use the R estimator's `predict(data, targets=..., source=...)` contract. They are not drop-in GP posterior selection helpers. Splits must protect entire runs; fitted candidates are compared on common valid support. See [tutorials](../tutorials.md) for the workflow.",
}


INTRO["responses"] += """

### Common kernel operations

Finite response families inherit these operations from the kernel base class:

| Operation | Result |
| --- | --- |
| `kernel.evaluate(lags)` or `kernel(lags)` | L2-normalized values at a supplied lag array; zero outside finite support. Identity raises because it is an analytic impulse. |
| `kernel.parameters` | Named scalar shape and lag parameters |
| `kernel.support` | Finite support endpoints |
| `kernel.metadata` | Family, parameters, support, normalization and coordinate conventions |
| `kernel.with_parameters(**updates)` | New immutable kernel with validated parameter updates |
| `kernel.to_coordinates()` / `kernel.from_coordinates(coordinates)` | Log coordinates for positive parameters and linear coordinates otherwise |

`SampledKernel` has no free scalar parameters; its supplied arrays describe a fixed curve. `Response.free_parameters`, `parameter_bounds()` and `support_envelope()` expose the resolved learning specification.
"""


def tree(path):
    return ast.parse(path.read_text())


def exports(path):
    nodes = tree(path).body
    names = next(
        ast.literal_eval(n.value)
        for n in nodes
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    imports = {
        a.asname or a.name: (n.module, a.name)
        for n in nodes
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    return {name: imports[name] for name in names if name != "__version__"}


CATALOG = {}
for init in (PACKAGE / "__init__.py", PACKAGE / "bayesian/__init__.py"):
    namespace = "multimodalsrm" + (".bayesian" if init.parent.name == "bayesian" else "")
    for name, (module, original) in exports(init).items():
        path = init.parent / (module.replace(".", "/") + ".py")
        node = next(
            n
            for n in tree(path).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == original
        )
        CATALOG[name] = (node, path, namespace)


def arguments(node):
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    parts = []
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        if arg.arg not in ("self", "cls"):
            parts.append(arg.arg + ("=" + ast.unparse(default) if default is not None else ""))
        if args.posonlyargs and i + 1 == len(args.posonlyargs):
            parts.append("/")
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(arg.arg + ("=" + ast.unparse(default) if default is not None else ""))
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return parts


def signature(name, parts):
    compact = name + "(" + ", ".join(parts) + ")"
    if len(compact) > 88:
        compact = name + "(\n" + "\n".join("    " + p + "," for p in parts) + "\n)"
    return "```python\n" + compact + "\n```\n"


def fields(node):
    result = []
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in CATALOG:
            result.extend(fields(CATALOG[base.id][0]))
    result.extend(
        n for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    )
    return result


def default(field):
    if field.value is None:
        return "required"
    value = field.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "field"
    ):
        for keyword in value.keywords:
            if keyword.arg == "default_factory":
                return ast.unparse(keyword.value) + "() (new per instance)"
    return ast.unparse(value)


def doc(node):
    return (ast.get_docstring(node) or "").replace("``", "`")


def render(key, title, names):
    out = [
        f"# {title}\n",
        INTRO[key],
        '\n!!! note "Generated from the checked-out source"\n    Signatures, field defaults, and available docstrings below are generated by `scripts/build_api_reference.py`. Constructor defaults are not a recommendation for every analysis. Semantic choices are explained in the [API guide](../api-guide.md).\n',
    ]
    for name in names:
        node, path, namespace = CATALOG[name]
        source = f"https://github.com/ljchang/multimodalsrm/blob/main/{path.relative_to(ROOT)}#L{node.lineno}"
        out.extend([f"## {name}\n", f"`from {namespace} import {name}` · [Source]({source})\n"])
        if isinstance(node, ast.FunctionDef):
            out.extend([signature(name, arguments(node)), doc(node)])
            continue
        init = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
        )
        if init:
            out.append(signature(name, arguments(init)))
        else:
            fs = fields(node)
            out.append(
                signature(
                    name,
                    [
                        f.target.id
                        + (
                            "=" + default(f).replace(" (new per instance)", "")
                            if f.value is not None
                            else ""
                        )
                        for f in fs
                    ],
                )
            )
            if fs:
                out.extend(["\n| Field | Type | Default |", "| --- | --- | --- |"])
                out.extend(
                    f"| `{f.target.id}` | `{ast.unparse(f.annotation).replace('|', '&#124;')}` | `{default(f)}` |"
                    for f in fs
                )
                out.append("")
        if doc(node):
            out.append(doc(node) + "\n")
        for method in node.body:
            if not isinstance(method, ast.FunctionDef) or method.name.startswith("_"):
                continue
            is_property = any(
                isinstance(d, ast.Name) and d.id in ("property", "cached_property")
                for d in method.decorator_list
            )
            out.append(f"### {name}.{method.name}\n")
            out.append(
                f"`{name}.{method.name}` (property)\n"
                if is_property
                else signature(f"{name}.{method.name}", arguments(method))
            )
            if doc(method):
                out.append(doc(method) + "\n")
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    listed = [name for _, names in PAGES.values() for name in names]
    if set(listed) != set(CATALOG) or len(listed) != len(set(listed)):
        raise SystemExit(f"API inventory differs from exports: {set(listed) ^ set(CATALOG)}")
    stale = []
    for key, (title, names) in PAGES.items():
        path = ROOT / f"docs/api/{key}.md"
        content = render(key, title, names)
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit("Regenerate API reference: " + ", ".join(stale))
    print(
        f"{'Checked' if args.check else 'Generated'} {len(PAGES)} pages covering {len(listed)} public exports"
    )


if __name__ == "__main__":
    main()
