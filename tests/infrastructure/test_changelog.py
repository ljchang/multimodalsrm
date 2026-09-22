"""The docs changelog page must stay in sync with the root changelog.

``docs/changelog.md`` is generated from ``CHANGELOG.md`` by
``scripts/sync_changelog.py``. Nothing rebuilds it during ``zensical build``, so
an edit to the root file that forgets to sync would publish a stale page. These
assert the page is current, that the nav points at it, and that released
sections are not used as a scratch area for unreleased work.
"""

import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sync_changelog", ROOT / "scripts/sync_changelog.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)

CHANGELOG = ROOT / "CHANGELOG.md"
PAGE = ROOT / "docs/changelog.md"


def test_docs_page_matches_the_root_changelog():
    expected = sync.render(CHANGELOG.read_text(encoding="utf-8"))
    assert PAGE.exists(), "docs/changelog.md is missing; run scripts/sync_changelog.py"
    assert PAGE.read_text(encoding="utf-8") == expected, (
        "docs/changelog.md is stale; run: python scripts/sync_changelog.py"
    )


def test_generated_page_is_marked_as_generated():
    assert sync.MARKER in PAGE.read_text(encoding="utf-8")


def test_navigation_exposes_the_changelog():
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    flattened = repr(config["project"]["nav"])
    assert "changelog.md" in flattened, "zensical.toml nav does not include changelog.md"


def test_changelog_keeps_an_unreleased_section():
    text = CHANGELOG.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text, (
        "CHANGELOG.md must keep an [Unreleased] section so merged work has a "
        "home that does not claim it shipped in a released version"
    )


def test_released_sections_are_dated_and_ordered_after_unreleased():
    text = CHANGELOG.read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\](.*)$", text, re.M)
    assert headings, "CHANGELOG.md has no version headings"
    assert headings[0][0] == "Unreleased", (
        "[Unreleased] must be the first version heading so new entries land there"
    )
    for version, rest in headings[1:]:
        assert re.search(r"—\s*\d{4}-\d{2}-\d{2}", rest), (
            f"released section [{version}] must carry an ISO date, got: {rest!r}"
        )
