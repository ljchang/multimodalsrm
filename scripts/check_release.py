"""Validate the model handoff before enabling model CI or package publication."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from packaging.version import Version


def check(root: Path, *, allow_pending: bool = False, tag: str | None = None) -> bool:
    """Return readiness; reject incomplete integration and invalid release tags."""
    root = root.resolve()
    manifest = json.loads((root / "model-integration.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported integration manifest schema")
    status = manifest.get("status")
    if status == "awaiting-model-import" and allow_pending and tag is None:
        return False
    if status != "integrated":
        raise ValueError("Model integration is incomplete; package publication is blocked")
    if manifest.get("source_repository") != "https://github.com/ljchang/shared-response-models":
        raise ValueError("Model source repository does not match the agreed handoff")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Model source commit must be a complete lowercase Git SHA")
    suites = manifest.get("test_suites", {})
    for suite in ("core", "bayesian"):
        files = suites.get(suite)
        if not isinstance(files, list) or not files:
            raise ValueError(f"Model test suite {suite} is missing")
        directory = (root / "tests" / suite).resolve()
        for item in files:
            if not isinstance(item, str):
                raise ValueError(f"Invalid path in test suite {suite}")
            path = (root / item).resolve()
            if (
                not path.is_relative_to(directory)
                or not path.is_file()
                or not path.name.startswith("test_")
                or path.suffix != ".py"
            ):
                raise ValueError(f"Missing or invalid file in test suite {suite}: {item}")
    if tag is not None:
        tree = ast.parse((root / "src/multimodalsrm/_version.py").read_text())
        versions = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        ]
        if len(versions) != 1:
            raise ValueError("Expected a single literal package version")
        version = Version(versions[0])
        if tag != f"v{version}" or version.is_devrelease or version.local:
            raise ValueError("Release tag must match a non-development, non-local package version")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--tag")
    args = parser.parse_args()
    try:
        ready = check(Path(__file__).resolve().parents[1], **vars(args))
    except (ValueError, OSError, TypeError, SyntaxError) as error:
        print(f"Release readiness: {error}", file=sys.stderr)
        return 1
    print(f"ready={str(ready).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
