"""Docs-only routing must fail closed for code, configuration and unknown diffs."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ci_policy", ROOT / "scripts/ci_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize(
    "paths",
    [
        ["README.md"],
        ["docs/testing.md", "docs/figures/example.svg", "CHANGELOG.md"],
        ["CONTRIBUTING.md", "docs/nested/guide.rst"],
        ["zensical.toml", "requirements-docs.txt", "docs/index.md"],
    ],
)
def test_prose_and_documentation_assets_are_lightweight(paths):
    assert policy.classify(paths) == "docs"


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["README.md", "src/multimodalsrm/kernels.py"],
        ["docs/example.py"],
        ["docs/test-migration.json"],
        ["examples/gp_map_quickstart.py"],
        ["tests/bayesian/test_bayesian_model.py"],
        [".github/workflows/ci.yml"],
        ["pyproject.toml"],
        ["MANIFEST.in"],
        ["model-integration.json"],
        ["unknown-file"],
        ["src/multimodalsrm/old.py", "docs/old.md"],
    ],
)
def test_code_mixed_unknown_and_renamed_code_run_model_checks(paths):
    assert policy.classify(paths) == "code"


def test_pr_diff_uses_merge_base_and_reports_both_sides_of_renames(monkeypatch):
    observed = []

    def diff(command, **kwargs):
        observed.append(command)
        return b"docs/a.md\x00src/deleted.py\x00"

    monkeypatch.setattr(policy.subprocess, "check_output", diff)
    event = {"pull_request": {"base": {"sha": "a" * 40}, "head": {"sha": "b" * 40}}}
    assert policy.changed_files("pull_request", event) == ["docs/a.md", "src/deleted.py"]
    assert "a" * 40 + "..." + "b" * 40 in observed[0]
    assert "--no-renames" in observed[0]


def test_main_push_uses_before_and_after(monkeypatch):
    observed = []

    def diff(command, **kwargs):
        observed.append(command)
        return b"README.md\x00"

    monkeypatch.setattr(policy.subprocess, "check_output", diff)
    assert policy.changed_files("push", {"before": "a" * 40, "after": "b" * 40}) == ["README.md"]
    assert "a" * 40 + ".." + "b" * 40 in observed[0]


@pytest.mark.parametrize(
    "event_name,event",
    [("workflow_dispatch", {}), ("push", {}), ("push", {"before": "0" * 40, "after": "b" * 40})],
)
def test_missing_history_or_manual_run_uses_code_profile(event_name, event):
    assert policy.changed_files(event_name, event) == []


def test_inaccessible_diff_falls_back_to_code(monkeypatch):
    def failed(*args, **kwargs):
        raise policy.subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(policy.subprocess, "check_output", failed)
    assert policy.changed_files("push", {"before": "a" * 40, "after": "b" * 40}) == []


@pytest.mark.parametrize(
    "profile,core,bayesian", [("docs", "skipped", "skipped"), ("code", "success", "success")]
)
def test_successful_expected_jobs_pass(profile, core, bayesian):
    policy.check_results(profile, "success", core, bayesian)


@pytest.mark.parametrize(
    "profile,quality,core,bayesian",
    [
        ("code", "success", "skipped", "success"),
        ("code", "success", "success", "failure"),
        ("code", "success", "cancelled", "success"),
        ("docs", "failure", "skipped", "skipped"),
        ("docs", "success", "failure", "skipped"),
        ("", "success", "skipped", "skipped"),
    ],
)
def test_failed_cancelled_missing_or_unexpected_jobs_block_merge(profile, quality, core, bayesian):
    with pytest.raises(ValueError):
        policy.check_results(profile, quality, core, bayesian)
