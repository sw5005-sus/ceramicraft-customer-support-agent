"""Tests for the FastAPI serve module (production entrypoint)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch get_tools and build_graph before importing app to avoid real MCP calls
with (
    patch(
        "ceramicraft_customer_support_agent.mcp_client.get_tools",
        new_callable=AsyncMock,
    ) as _mock_gt,
    patch("serve.build_checkpointer", new_callable=AsyncMock),
):
    _mock_gt.return_value = []
    from serve import app, _agent_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Pre-populate agent cache with mocks so lifespan init is not needed."""
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [], "intent": "chitchat"})
    mock_checkpointer = MagicMock()
    mock_checkpointer.adelete_thread = AsyncMock()

    _agent_cache["agent"] = mock_agent
    _agent_cache["checkpointer"] = mock_checkpointer
    yield
    _agent_cache.clear()


@pytest.fixture()
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


def test_health_check(client):
    """GET /cs-agent/v1/ping should return ok when agent is initialised."""
    resp = client.get("/cs-agent/v1/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_check_not_ready(client):
    """GET /cs-agent/v1/ping should return 503 when agent is not yet initialised."""
    _agent_cache.clear()
    resp = client.get("/cs-agent/v1/ping")
    assert resp.status_code == 503
    assert resp.json()["status"] == "starting"


def test_chat_returns_reply_with_thread_id(client):
    """POST /chat should return the agent's reply and a thread_id."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Hello! How can I help?"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post("/chat", json={"message": "hi", "thread_id": "t1"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "Hello! How can I help?"
    assert data["thread_id"] == "t1"


def test_chat_generates_thread_id_when_omitted(client):
    """POST /chat without thread_id should auto-generate one."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Hello!"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "Hello!"
    assert len(data["thread_id"]) == 32  # uuid4 hex


def test_chat_preserves_explicit_thread_id(client):
    """POST /chat with explicit thread_id should use it, not generate."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post("/chat", json={"message": "hi", "thread_id": "my-thread-42"})

    assert resp.status_code == 200
    assert resp.json()["thread_id"] == "my-thread-42"

    # Verify the agent was called with the explicit thread_id
    call_config = _agent_cache["agent"].ainvoke.call_args[1]["config"]
    assert call_config["configurable"]["thread_id"] == "my-thread-42"


def test_chat_extracts_bearer_token(client):
    """POST /chat should extract Bearer token from Authorization header."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post(
        "/chat",
        json={"message": "hi", "thread_id": "t-token"},
        headers={"Authorization": "Bearer mytoken123"},
    )

    assert resp.status_code == 200

    call_args = _agent_cache["agent"].ainvoke.call_args
    state = call_args[0][0]
    assert state["auth_token"] == "mytoken123"


def test_chat_without_token(client):
    """POST /chat should set auth_token=None when no header."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200

    call_args = _agent_cache["agent"].ainvoke.call_args
    state = call_args[0][0]
    assert state["auth_token"] is None


def test_chat_handles_dict_messages(client):
    """POST /chat should handle dict format messages."""
    _agent_cache["agent"].ainvoke.return_value = {
        "messages": [{"role": "assistant", "content": "dict response"}]
    }

    resp = client.post("/chat", json={"message": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "dict response"
    assert data["thread_id"]  # auto-generated


def test_chat_fallback_on_empty_content(client):
    """POST /chat should return fallback when AI message has empty content."""
    empty_msg = MagicMock()
    empty_msg.type = "ai"
    empty_msg.content = ""

    _agent_cache["agent"].ainvoke.return_value = {"messages": [empty_msg]}

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    data = resp.json()
    assert "couldn't process" in data["reply"]
    assert data["thread_id"]  # still returned


def test_chat_returns_500_on_exception(client):
    """POST /chat should return 500 on agent failure with thread_id."""
    _agent_cache["agent"].ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    resp = client.post("/chat", json={"message": "hi"})

    assert resp.status_code == 500
    data = resp.json()
    assert "went wrong" in data["reply"]
    assert data["thread_id"]  # still returned even on error


def test_chat_500_preserves_explicit_thread_id(client):
    """POST /chat 500 should preserve the caller's thread_id."""
    _agent_cache["agent"].ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    resp = client.post("/chat", json={"message": "hi", "thread_id": "err-thread"})

    assert resp.status_code == 500
    assert resp.json()["thread_id"] == "err-thread"


def test_reset_returns_ok(client):
    """POST /reset should return ok status when checkpointer is cached."""
    resp = client.post("/reset?thread_id=my-thread")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "my-thread" in data["message"]


def test_chat_requires_message(client):
    """POST /chat should reject request without message."""
    resp = client.post("/chat", json={"thread_id": "t1"})

    assert resp.status_code == 422  # Validation error


def test_chat_accepts_without_thread_id(client):
    """POST /chat should accept request without thread_id (auto-generated)."""
    # This just validates the request model accepts it; agent will fail
    # without mock but the 422 should NOT happen.
    resp = client.post("/chat", json={"message": "hi"})

    # Could be 200 or 500 depending on agent availability, but NOT 422
    assert resp.status_code != 422


def test_reset_requires_thread_id(client):
    """POST /reset should reject request without thread_id."""
    resp = client.post("/reset")

    assert resp.status_code == 422  # Validation error


def test_reset_clears_memory_saver(client):
    """POST /reset should call adelete_thread on the cached checkpointer."""
    thread_id = "test-reset-thread"
    resp = client.post(f"/reset?thread_id={thread_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    _agent_cache["checkpointer"].adelete_thread.assert_called_once_with(thread_id)


def test_reset_returns_500_on_error(client):
    """POST /reset should return 500 if adelete_thread raises."""
    _agent_cache["checkpointer"].adelete_thread = AsyncMock(
        side_effect=Exception("db error")
    )

    resp = client.post("/reset?thread_id=bad-thread")
    assert resp.status_code == 500
    assert resp.json()["status"] == "error"


@patch("serve.set_auth_token")
def test_chat_sets_auth_context(mock_set_token, client):
    """POST /chat should call set_auth_token before agent invocation."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ok"

    _agent_cache["agent"].ainvoke.return_value = {"messages": [ai_msg]}

    resp = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer ctx_token_123"},
    )

    assert resp.status_code == 200
    mock_set_token.assert_called_once_with("ctx_token_123")


def test_openapi_docs_available(client):
    """GET /docs should return the Swagger UI."""
    resp = client.get("/docs")
    assert resp.status_code == 200


# ---------- SSE /chat/stream tests ----------


def _parse_sse(raw_text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = ""
    for line in raw_text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "":
            if current_event is not None:
                events.append(
                    {"event": current_event, "data": json.loads(current_data)}
                )
                current_event = None
                current_data = ""
    return events


def _make_stream_agent(ai_content="Hello!", intent="chitchat"):
    """Create a mock agent that supports astream with updates mode."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = ai_content

    async def _astream(state, config, stream_mode="updates"):
        yield {"input_guard": {"blocked": False}}
        yield {"classifier": {"intent": intent, "last_intent": intent}}
        yield {intent: {"messages": [ai_msg]}}
        yield {"guard": {"messages": []}}

    mock_agent = MagicMock()
    mock_agent.astream = _astream
    # Keep ainvoke for /chat tests
    mock_agent.ainvoke = AsyncMock(
        return_value={"messages": [ai_msg], "intent": intent}
    )
    return mock_agent


def test_chat_stream_returns_sse_events(client):
    """POST /chat/stream should return SSE events."""
    _agent_cache["agent"] = _make_stream_agent("Hi there!", "chitchat")

    resp = client.post(
        "/chat/stream",
        json={"message": "hello", "thread_id": "t-stream"},
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]

    assert "guarding" in event_types
    assert "classifying" in event_types
    assert "processing" in event_types
    assert "reply" in event_types
    assert "done" in event_types

    # Check reply content
    reply_event = next(e for e in events if e["event"] == "reply")
    assert reply_event["data"]["content"] == "Hi there!"

    # Check done has thread_id
    done_event = next(e for e in events if e["event"] == "done")
    assert done_event["data"]["thread_id"] == "t-stream"


def test_chat_stream_includes_intent_in_processing(client):
    """POST /chat/stream processing event should include the intent."""
    _agent_cache["agent"] = _make_stream_agent("OK", "order")

    resp = client.post("/chat/stream", json={"message": "my orders"})

    events = _parse_sse(resp.text)
    processing_event = next(e for e in events if e["event"] == "processing")
    assert processing_event["data"]["intent"] == "order"
    assert processing_event["data"]["stage"] == "order"


def test_chat_stream_auto_generates_thread_id(client):
    """POST /chat/stream without thread_id should auto-generate one."""
    _agent_cache["agent"] = _make_stream_agent()

    resp = client.post("/chat/stream", json={"message": "hi"})

    events = _parse_sse(resp.text)
    done_event = next(e for e in events if e["event"] == "done")
    assert len(done_event["data"]["thread_id"]) == 32


def test_chat_stream_handles_error(client):
    """POST /chat/stream should emit error event on agent failure."""

    async def _failing_astream(state, config, stream_mode="updates"):
        raise RuntimeError("boom")
        yield  # noqa: RET503  — make it an async generator

    mock_agent = MagicMock()
    mock_agent.astream = _failing_astream
    _agent_cache["agent"] = mock_agent

    resp = client.post(
        "/chat/stream",
        json={"message": "hi", "thread_id": "t-err"},
    )

    assert resp.status_code == 200  # SSE always starts 200
    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]
    assert "error" in event_types
    assert "done" in event_types


def test_chat_stream_extracts_bearer_token(client):
    """POST /chat/stream should pass Bearer token to set_auth_token."""
    _agent_cache["agent"] = _make_stream_agent()

    with patch("serve.set_auth_token") as mock_set:
        resp = client.post(
            "/chat/stream",
            json={"message": "hi"},
            headers={"Authorization": "Bearer stream_tok"},
        )

    assert resp.status_code == 200
    mock_set.assert_called_with("stream_tok")


def test_chat_stream_fallback_on_empty_reply(client):
    """POST /chat/stream should return fallback when no AI content."""

    async def _empty_astream(state, config, stream_mode="updates"):
        yield {"input_guard": {"blocked": False}}
        yield {"classifier": {"intent": "chitchat"}}
        yield {"chitchat": {"messages": []}}
        yield {"guard": {"messages": []}}

    mock_agent = MagicMock()
    mock_agent.astream = _empty_astream
    _agent_cache["agent"] = mock_agent

    resp = client.post("/chat/stream", json={"message": "hi"})

    events = _parse_sse(resp.text)
    reply_event = next(e for e in events if e["event"] == "reply")
    assert "couldn't process" in reply_event["data"]["content"]


def test_chat_stream_no_cache_headers(client):
    """POST /chat/stream should include no-cache and no-buffering headers."""
    _agent_cache["agent"] = _make_stream_agent()

    resp = client.post("/chat/stream", json={"message": "hi"})

    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"
