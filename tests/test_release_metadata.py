"""Release metadata must stay in sync: version constant, pyproject, changelog."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import wavrail

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_latest_release_matches_version() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    assert match is not None, "no released version heading found"
    assert match.group(1) == wavrail.__version__


def test_pyproject_version_matches_package() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == wavrail.__version__


def test_version_stays_on_patch_line() -> None:
    major, minor, patch = wavrail.__version__.split(".")
    assert (major, minor) == ("0", "1")
    assert int(patch) >= 1
