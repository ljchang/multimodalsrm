"""A model suite skipped in its entirety must never qualify a release."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "body, expected",
    [
        ("import pytest\n@pytest.mark.skip\ndef test_model(): pass\n", 1),
        ("def test_model(): assert 1 == 2\n", 1),
        ("def test_model(): assert 1 + 1 == 2\n", 0),
        (
            "import pytest\n@pytest.mark.skip\ndef test_optional(): pass\n"
            "def test_model(): assert 1 + 1 == 2\n",
            0,
        ),
    ],
)
def test_real_pytest_report_requires_executed_passing_model_test(tmp_path, body, expected):
    script = Path(__file__).parents[2] / "scripts/check_test_report.py"
    assert script.is_file(), "Model suite execution gate has not been implemented"
    (tmp_path / "test_model.py").write_text(body)
    report = tmp_path / "results.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--junitxml", str(report), "test_model.py"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    checked = subprocess.run(
        [sys.executable, str(script), str(report)],
        text=True,
        capture_output=True,
    )
    assert checked.returncode == expected, checked.stdout + checked.stderr
