"""Export the completed block-filter and smaller-Gaussian qualification studies."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("local_data/gp-block-filter-2026-09-23")
    )
    args = parser.parse_args()
    result = dict(scope="Research only; no production backend or supported-order changes", cases={})
    cases = (
        "gamma3-k6",
        "gaussian-k3",
        "gamma3-k6-cache64",
        "gaussian-k3-cache64",
        "gamma3-k6-original-cache64",
    )
    rows = [
        "| Model / calculation | Warm speedup | Compile reference / candidate (s) | Estimated memory reference / candidate (GB) | Accuracy points passing |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "dense": "block prediction",
        "compact": "compact Joseph",
        "original": "original update",
    }
    for name in cases:
        d = json.loads((args.directory / (name + ".json")).read_text())
        assert d["status"] == "finished", name
        result["cases"][name] = d
        for method in d["all_passed"]:
            prefix = "Gamma K=6" if d["arguments"]["candidate"] == "gamma3" else "Gaussian K=3"
            cache = ", reuse" if d["arguments"].get("transition_cache", 0) else ""
            label = prefix + ", " + labels[method] + cache
            p = d["points"][0][method]
            r, c = d["measurements"]["reference"], d["measurements"][method]
            passed = sum(p[method]["passed"] for p in d["points"])
            rows.append(
                f"| {label} | {p['speedup']:.2f}x | {r['compile_seconds']:.1f} / {c['compile_seconds']:.1f} | "
                f"{r['compiler_memory']['estimated_live_bytes'] / 1e9:.2f} / {c['compiler_memory']['estimated_live_bytes'] / 1e9:.2f} | {passed}/8 |"
            )
    result["checks"] = {}
    for name in (
        "gamma-small-numerics",
        "gaussian-small-numerics",
        "transition-reuse-check",
        "gamma-interval-counts",
        "gaussian-interval-counts",
        "gaussian-map-orders",
    ):
        d = json.loads((args.directory / (name + ".json")).read_text())
        assert d["status"] in ("finished", "inspected"), name
        result["checks"][name] = d
    orders = [
        "| Gaussian order | States (K=1) | Full-box covariance bound | Objective error at existing MAP | Largest gradient change | Projected gradient | Local MAP gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["checks"]["gaussian-map-orders"]["orders"]:
        p = next(p for p in row["points"] if p["name"] == "saved_MAP")
        orders.append(
            f"| {row['order']} | {row['dimension']} | {row['covariance_bound']:.3g} | "
            f"{p['objective_absolute_difference']:.3g} | {p['gradient_max_absolute_difference']:.3g} | "
            f"{p['projected_gradient']:.3g} | {'Pass' if p['map_local_gate_passed'] else 'Fail'} |"
        )
    root = Path(__file__).resolve().parents[1]
    (root / "docs/performance/2026-09-23-gp-block-filter-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    path = root / "docs/performance/2026-09-23-gp-block-filter.md"
    report = path.read_text()
    for marker, content in (("TIMINGS", "\n".join(rows)), ("ORDERS", "\n".join(orders))):
        start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
        report, count = re.subn(
            re.escape(start) + ".*?" + re.escape(end),
            lambda _: start + "\n\n" + content + "\n\n" + end,
            report,
            flags=re.S,
        )
        assert count == 1
    path.write_text(report)
    print("\n".join(rows + [""] + orders))


if __name__ == "__main__":
    main()
