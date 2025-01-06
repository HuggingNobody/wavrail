"""Smoke tests verifying the package metadata."""

from __future__ import annotations

import wavrail


def test_version_is_string() -> None:
    assert isinstance(wavrail.__version__, str)
    assert wavrail.__version__


def test_version_is_semver_like() -> None:
    parts = wavrail.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
