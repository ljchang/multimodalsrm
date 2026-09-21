"""Every test file must belong to exactly one nonempty CI shard."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "model_test_sharding", Path(__file__).resolve().parents[2] / "scripts/run_model_tests.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shards_cover_all_files_once(tmp_path):
    expected = set()
    for index in range(19):
        path = tmp_path / f"test_case_{index}.py"
        path.touch()
        expected.add(path)
    (tmp_path / "helpers.py").touch()
    shards = [module.select_files(tmp_path, index, 8) for index in range(8)]
    flattened = [path for shard in shards for path in shard]
    assert set(flattened) == expected
    assert len(flattened) == len(expected)


@pytest.mark.parametrize("shard,count", [(-1, 1), (1, 1), (0, 0), (0, -1)])
def test_invalid_shard_rejected(tmp_path, shard, count):
    with pytest.raises(ValueError, match="Invalid"):
        module.select_files(tmp_path, shard, count)


def test_empty_shard_rejected(tmp_path):
    with pytest.raises(ValueError, match="Empty"):
        module.select_files(tmp_path, 0, 1)


def test_smoke_manifest_selects_existing_cases_in_the_correct_suite(tmp_path):
    directory = tmp_path / "tests/bayesian"
    directory.mkdir(parents=True)
    (directory / "test_model.py").touch()
    manifest = tmp_path / "tests/smoke-tests.json"
    manifest.write_text(json.dumps({"bayesian": ["tests/bayesian/test_model.py::test_fit"]}))
    assert module.select_smoke(tmp_path, "bayesian") == [
        str(directory / "test_model.py") + "::test_fit"
    ]


@pytest.mark.parametrize(
    "cases",
    [
        [],
        ["tests/bayesian/test_missing.py"],
        ["tests/core/test_other.py"],
        ["../test_outside.py"],
        [42],
    ],
)
def test_invalid_smoke_manifest_is_rejected(tmp_path, cases):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/smoke-tests.json").write_text(json.dumps({"bayesian": cases}))
    with pytest.raises(ValueError):
        module.select_smoke(tmp_path, "bayesian")
