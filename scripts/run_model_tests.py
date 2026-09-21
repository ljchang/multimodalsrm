"""Run one deterministic file shard against the installed model package."""

import argparse
import subprocess
import sys
from pathlib import Path


def select_files(directory: Path, shard: int, count: int) -> list[Path]:
    if count < 1 or not 0 <= shard < count:
        raise ValueError("Invalid shard index or count")
    files = sorted(directory.glob("test_*.py"))
    selected = files[shard::count]
    if not selected:
        raise ValueError("Empty model test shard")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("core", "bayesian"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--report", default="results.xml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = select_files(root / "tests" / args.suite, args.shard, args.count)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            "--durations=10",
            f"--junitxml={args.report}",
            *map(str, selected),
        ],
        cwd=root,
        check=False,
    )
    if result.returncode:
        return result.returncode
    return subprocess.call(
        [sys.executable, str(root / "scripts/check_test_report.py"), args.report], cwd=root
    )


if __name__ == "__main__":
    raise SystemExit(main())
