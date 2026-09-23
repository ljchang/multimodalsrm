"""Probe smaller Gaussian response banks; does not change supported orders.

These finite-band spectral integrals are numerical diagnostics, not certified
covariance or posterior-error bounds. The spectral reference is the full
Gaussian with the production finite-support L2 normalizer. The separate L1
qualification uses the production truncated Gaussian response.

Example:
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/qualify_gp_gaussian_orders.py \
        --output /tmp/gaussian-orders.json
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import simpson

from multimodalsrm.bayesian.state_space_gaussian_rational import (
    _FULL_MASS,
    _fit,
    qualify_impulse,
)


def probe(order):
    start = time.perf_counter()
    poles, residues = _fit(order)
    qualified = qualify_impulse(poles, residues)
    grids = []
    for count in (8193, 16385):
        u = np.r_[0.0, np.geomspace(1e-7, 1e4, count)]
        approximate = (1 / (1j * u[:, None] - poles) @ residues) * np.exp(6j * u)
        target = _FULL_MASS * np.exp(-u * u / 2)
        cases = []
        for width in (0.5, 1.0, 2.0):
            omega = u / width
            approx_H, H = np.sqrt(width) * approximate, np.sqrt(width) * target
            for length_scale in (1.0, 3.0, 10.0):
                rate = np.sqrt(3) / length_scale
                spectrum = 4 * rate**3 / (rate * rate + omega * omega) ** 2
                cross = simpson(spectrum * abs(approx_H - H), x=omega) / np.pi
                self_error = (
                    simpson(spectrum * abs(abs(approx_H) ** 2 - abs(H) ** 2), x=omega) / np.pi
                )
                variance = simpson(spectrum * abs(H) ** 2, x=omega) / np.pi
                cases.append(
                    dict(
                        width=width,
                        length_scale=length_scale,
                        cross_covariance_error_bound=float(cross),
                        self_covariance_error_bound=float(self_error),
                        variance=float(variance),
                        normalized_cross_bound=float(cross / np.sqrt(variance)),
                        relative_self_bound=float(self_error / variance),
                    )
                )
        grids.append(cases)
    return dict(
        order=order,
        fit_and_integral_seconds=time.perf_counter() - start,
        l1_bound=qualified["numerical_l1_bound"],
        worst_normalized_cross=max(c["normalized_cross_bound"] for c in grids[-1]),
        worst_relative_self=max(c["relative_self_bound"] for c in grids[-1]),
        integral_refinement_max_difference=max(
            abs(a[field] - b[field])
            for a, b in zip(*grids)
            for field in ("cross_covariance_error_bound", "self_covariance_error_bound", "variance")
        ),
        cases=grids[-1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        qualification="Finite-band numerical estimates; not posterior-error certificates",
        measurements=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for order in (6, 8, 12, 16, 20, 24):
        row = probe(order)
        result["measurements"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
