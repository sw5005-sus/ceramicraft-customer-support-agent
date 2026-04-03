"""Shared test fixtures and configuration."""

import pytest


@pytest.fixture(autouse=True)
def _reset_checkpointer():
    """Reset the module-level checkpointer between tests to prevent state pollution."""
    import ceramicraft_customer_support_agent.graph as graph_mod

    original = graph_mod._checkpointer
    graph_mod._checkpointer = None
    yield
    graph_mod._checkpointer = original
