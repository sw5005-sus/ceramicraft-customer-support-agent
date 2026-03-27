"""Tests for the FastAPI serve module (production entrypoint)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch get_tools and build_agent before importing app to avoid real MCP calls
with patch("ceramicraft_customer_support_agent.mcp_client.get_tools", new_callable=AsyncMock) as _mock_gt:
    _mock_gt.return_value = []
    from serve import app, _agent_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the agent cache between tests."""
    _agent_cache.clear()
    yield
    _agent_cache.clear()


@pytest.fixture()
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


def test_health_check(client):
    """GET /health should return ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_returns_reply(mock_get_tools, mock_build, client):
    """POST /chat should return the agent's reply."""
    mock_get_tools.return_value = []

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Hello! How can I help?"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    resp = client.post("/chat", json={"message": "hi", "thread_id": "t1"})

    assert resp.status_code == 200
    assert resp.json() == {"reply": "Hello! How can I help?"}


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_default_thread_id(mock_get_tools, mock_build, client):
    """POST /chat should use 'default' thread_id when not provided."""
    mock_get_tools.return_value = []

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "response"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    resp = client.post("/chat", json={"message": "test"})

    assert resp.status_code == 200

    call_args = mock_agent.ainvoke.call_args
    config = call_args[1]["config"]
    assert config["configurable"]["thread_id"] == "default"


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_extracts_bearer_token(mock_get_tools, mock_build, client):
    """POST /chat should extract Bearer token from Authorization header."""
    mock_get_tools.return_value = []

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    resp = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer mytoken123"},
    )

    assert resp.status_code == 200

    call_args = mock_agent.ainvoke.call_args
    state = call_args[0][0]
    assert state["auth_token"] == "mytoken123"


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_without_token(mock_get_tools, mock_build, client):
    """POST /chat should set auth_token=None when no header."""
    mock_get_tools.return_value = []

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200

    call_args = mock_agent.ainvoke.call_args
    state = call_args[0][0]
    assert state["auth_token"] is None


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_handles_dict_messages(mock_get_tools, mock_build, client):
    """POST /chat should handle dict format messages."""
    mock_get_tools.return_value = []

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [{"role": "assistant", "content": "dict response"}]
    }
    mock_build.return_value = mock_agent

    resp = client.post("/chat", json={"message": "test"})

    assert resp.status_code == 200
    assert resp.json() == {"reply": "dict response"}


@patch("serve.build_agent")
@patch("serve.get_tools")
def test_chat_fallback_on_empty_content(mock_get_tools, mock_build, client):
    """POST /chat should return fallback when AI message has empty content."""
    mock_get_tools.return_value = []

    empty_msg = MagicMock()
    empty_msg.type = "ai"
    empty_msg.content = ""

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [empty_msg]}
    mock_build.return_value = mock_agent

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert "couldn't process" in resp.json()["reply"]


@patch("serve.get_tools")
def test_chat_returns_500_on_exception(mock_get_tools, client):
    """POST /chat should return 500 on agent failure."""
    mock_get_tools.side_effect = RuntimeError("boom")

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 500
    assert "went wrong" in resp.json()["reply"]


def test_reset_returns_ok(client):
    """POST /reset should return ok status."""
    resp = client.post("/reset?thread_id=my-thread")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "my-thread" in data["message"]


def test_chat_requires_message(client):
    """POST /chat should reject request without message."""
    resp = client.post("/chat", json={})

    assert resp.status_code == 422  # Validation error


def test_openapi_docs_available(client):
    """GET /docs should return the Swagger UI."""
    resp = client.get("/docs")
    assert resp.status_code == 200
