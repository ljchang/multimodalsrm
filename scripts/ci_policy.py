"""Select fast CI jobs without skipping the required workflow itself."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT_DOCS = {"README.md", "CHANGELOG.md", "CONTRIBUTING.md"}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def classify(paths: list[str]) -> str:
    """Unknown paths and an unavailable/empty diff always receive model checks."""
    if paths and all(
        path in ROOT_DOCS
        or (path.startswith("docs/") and PurePosixPath(path).suffix.lower() in DOC_SUFFIXES)
        for path in paths
    ):
        return "docs"
    return "code"


def changed_files(event_name: str, event: dict) -> list[str]:
    if event_name == "pull_request":
        pr = event.get("pull_request", {})
        before, after = pr.get("base", {}).get("sha"), pr.get("head", {}).get("sha")
        separator = "..."
    elif event_name == "push":
        before, after = event.get("before"), event.get("after")
        separator = ".."
    else:
        return []
    if any(
        not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None or sha == "0" * 40
        for sha in (before, after)
    ):
        return []
    try:
        raw = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", before + separator + after, "--"],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [name for name in raw.decode("utf-8", errors="replace").split("\0") if name]


def check_results(profile: str, quality: str, core: str, bayesian: str) -> None:
    if quality != "success" or profile not in {"docs", "code"}:
        raise ValueError("CI quality checks or profile selection did not succeed")
    expected = "skipped" if profile == "docs" else "success"
    if core != expected or bayesian != expected:
        raise ValueError(f"Expected both model jobs to be {expected}, got {core} and {bayesian}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    if args.check_results:
        check_results(
            os.environ.get("PROFILE", ""),
            os.environ.get("QUALITY", ""),
            os.environ.get("CORE", ""),
            os.environ.get("BAYESIAN", ""),
        )
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    files = changed_files(os.environ["GITHUB_EVENT_NAME"], event)
    profile = classify(files)
    output = f"profile={profile}\n"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.write(output)
    print(output.strip())
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
            handle.write(
                f"CI profile: **{profile}** ({len(files)} changed paths). "
                + (
                    "Documentation checks; model jobs are intentionally skipped.\n"
                    if profile == "docs"
                    else "Core regressions and Bayesian smoke checks; full suites run before release.\n"
                )
            )


if __name__ == "__main__":
    main()
