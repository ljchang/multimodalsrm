"""Catch accidental publication before model integration or under a wrong tag."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).parents[2] / "scripts" / "check_release.py"
    assert path.is_file(), "Release readiness gate has not been implemented"
    spec = importlib.util.spec_from_file_location("release_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def integrated(tmp_path):
    (tmp_path / "src/multimodalsrm").mkdir(parents=True)
    (tmp_path / "src/multimodalsrm/_version.py").write_text('__version__ = "0.1.0"\n')
    for suite in ("core", "bayesian"):
        folder = tmp_path / "tests" / suite
        folder.mkdir(parents=True)
        (folder / "test_workflow.py").write_text("def test_workflow():\n    assert True\n")
    (tmp_path / "model-integration.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "integrated",
                "source_repository": "https://github.com/ljchang/shared-response-models",
                "source_commit": "a" * 40,
                "test_suites": {
                    "core": ["tests/core/test_workflow.py"],
                    "bayesian": ["tests/bayesian/test_workflow.py"],
                },
            }
        )
    )
    return tmp_path


def test_pending_integration_blocks_publication_but_allows_infrastructure(gate, integrated):
    path = integrated / "model-integration.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "awaiting-model-import"
    path.write_text(json.dumps(manifest))
    assert gate.check(integrated, allow_pending=True) is False
    with pytest.raises(ValueError, match="integration"):
        gate.check(integrated)


@pytest.mark.parametrize("damage", ["missing", "empty", "escape", "wrong_suite"])
def test_incomplete_or_outside_test_suites_block_release(gate, integrated, damage):
    path = integrated / "model-integration.json"
    manifest = json.loads(path.read_text())
    if damage == "missing":
        (integrated / "tests/core/test_workflow.py").unlink()
    elif damage == "empty":
        manifest["test_suites"]["core"] = []
    elif damage == "escape":
        manifest["test_suites"]["core"] = ["../test_workflow.py"]
    else:
        manifest["test_suites"]["core"] = ["tests/bayesian/test_workflow.py"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="suite"):
        gate.check(integrated, tag="v0.1.0")


@pytest.mark.parametrize("commit", [None, "main", "a" * 39, "g" * 40])
def test_unfrozen_source_identity_blocks_release(gate, integrated, commit):
    path = integrated / "model-integration.json"
    manifest = json.loads(path.read_text())
    manifest["source_commit"] = commit
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="commit"):
        gate.check(integrated)


@pytest.mark.parametrize("repository", [None, "", "https://example.com/unrelated"])
def test_unexpected_source_repository_blocks_release(gate, integrated, repository):
    path = integrated / "model-integration.json"
    manifest = json.loads(path.read_text())
    manifest["source_repository"] = repository
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="repository"):
        gate.check(integrated)


@pytest.mark.parametrize(
    "version,tag",
    [
        ("0.1.0", "v0.2.0"),
        ("0.1.0.dev0", "v0.1.0.dev0"),
        ("0.1.0+local", "v0.1.0+local"),
        ("0.1.0", "main"),
    ],
)
def test_wrong_or_unpublishable_version_blocks_release(gate, integrated, version, tag):
    (integrated / "src/multimodalsrm/_version.py").write_text(f'__version__ = "{version}"\n')
    with pytest.raises(ValueError, match="version|tag"):
        gate.check(integrated, tag=tag)


@pytest.mark.parametrize("version", ["0.1.0", "0.1.0rc1"])
def test_integrated_candidate_with_matching_tag_can_proceed(gate, integrated, version):
    (integrated / "src/multimodalsrm/_version.py").write_text(f'__version__ = "{version}"\n')
    assert gate.check(integrated, tag=f"v{version}") is True
