"""Shared pytest configuration for the wavrail test suite."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip model tests when the optional ``torch`` extra is unavailable.

    The acoustic-model modules pull in heavy dependencies that the
    default test environment may not have installed. Detect the extra
    once at collection time and tag skipped items clearly.

    Only tests that carry an explicit ``@pytest.mark.model`` marker
    are skipped; the path-based keyword inclusion used by pytest is
    ignored so that synthetic-dataset tests in the same directory can
    still run without the optional extra.
    """

    has_torch = _torch_available()
    skip_model = pytest.mark.skip(reason="torch extra not installed")
    for item in items:
        if item.get_closest_marker("model") and not has_torch:
            item.add_marker(skip_model)


def _torch_available() -> bool:
    if os.environ.get("WAVRAIL_SKIP_TORCH_CHECK") == "1":
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
