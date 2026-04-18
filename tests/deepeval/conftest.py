"""Configuration for DeepEval LLM evaluation tests.

These tests require OPENAI_API_KEY. When the key is absent the entire
directory is skipped automatically so that ``uv run pytest tests/`` stays
green without special flags.
"""

import os

import pytest

# Skip all tests in this directory when OPENAI_API_KEY is not set.
_requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping DeepEval tests",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the skip marker to every test in tests/deepeval/."""
    for item in items:
        if "/deepeval/" in str(item.fspath):
            item.add_marker(_requires_openai)
