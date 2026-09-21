"""Run one deterministic file shard against the installed model package."""

import argparse
import json
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


def select_smoke(root: Path, suite: str) -> list[str]:
    cases = json.loads((root / "tests/smoke-tests.json").read_text()).get(suite)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Missing smoke cases for {suite}")
    directory = (root / "tests" / suite).resolve()
    selected = []
    for case in cases:
        if not isinstance(case, str):
            raise ValueError("Invalid smoke selector")
        filename, *nodes = case.split("::")
        path = (root / filename).resolve()
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or not path.name.startswith("test_")
            or path.suffix != ".py"
            or any(not node for node in nodes)
        ):
            raise ValueError(f"Invalid smoke selector: {case}")
        selected.append("::".join([str(path), *nodes]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("core", "bayesian"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--report", default="results.xml")
    parser.add_argument("--smoke", action="store_true", help="Run the curated fast checks only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.smoke and (args.shard != 0 or args.count != 1):
        parser.error("Smoke checks are not sharded")
    selected = (
        select_smoke(root, args.suite)
        if args.smoke
        else select_files(root / "tests" / args.suite, args.shard, args.count)
    )
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
