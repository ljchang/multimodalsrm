"""Fast PR checks must never replace the full release-verification dependency."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    # BaseLoader keeps YAML's "on" key and expression strings literal.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_full_suite_is_manual_or_reusable_and_retains_both_platforms():
    full = workflow("full-tests.yml")
    assert set(full["on"]) == {"workflow_dispatch", "workflow_call"}
    matrix = full["jobs"]["models"]["strategy"]["matrix"]
    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest"}
    assert matrix["shard"] == list(map(str, range(8)))
    assert len(matrix["include"]) == 4
    command = full["jobs"]["models"]["steps"][-1]["run"]
    assert "--smoke" not in command
    assert "--count" in command
    assert "models" in full["jobs"]["required"]["needs"]


def test_release_waits_for_full_verification_and_publishes_its_artifacts():
    release = workflow("release.yml")
    verify = release["jobs"]["verify"]
    assert verify["uses"] == "./.github/workflows/full-tests.yml"
    assert verify["with"]["require_models"] == "true"
    assert verify["with"]["source_ref"] == "${{ needs.resolve.outputs.sha }}"
    for name in ("publish-pypi", "publish-testpypi"):
        job = release["jobs"][name]
        assert job["needs"] == "verify"
        assert job["steps"][0]["with"]["name"] == "python-distributions"


def test_fast_workflow_has_stable_gate_and_no_top_level_path_skip():
    fast = workflow("ci.yml")
    assert "pull_request" in fast["on"]
    assert not fast["on"]["pull_request"]
    assert "paths" not in fast["on"]["push"]
    assert "paths-ignore" not in fast["on"]["push"]
    assert fast["concurrency"]["cancel-in-progress"] == "true"
    assert fast["jobs"]["required"]["name"] == "Required checks"
    assert set(fast["jobs"]["required"]["needs"]) == {"quality", "core", "bayesian"}
    for name in ("core", "bayesian"):
        assert fast["jobs"][name]["if"] == "needs.quality.outputs.profile == 'code'"
    assert "--smoke" in fast["jobs"]["bayesian"]["steps"][-1]["run"]
