"""Tests for the MCP server module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from ceramicraft_customer_support_agent import mcp_server


def test_create_mcp_server_returns_fastmcp():
    """create_mcp_server should return a FastMCP instance."""
    mcp = mcp_server.create_mcp_server()
    assert mcp is not None
    assert mcp.name == "CeramiCraft Customer Support"


def test_create_mcp_server_registers_tools():
    """create_mcp_server should register chat and reset tools."""
    mcp = mcp_server.create_mcp_server()
    tools = mcp._tool_manager._tools
    assert "chat" in tools
    assert "reset" in tools


# --- _extract_bearer_token tests ---


def test_extract_bearer_token_from_headers():
    """Should extract token from context headers."""
    ctx = MagicMock()
    ctx.headers = {"authorization": "Bearer abc123"}
    assert mcp_server._extract_bearer_token(ctx) == "abc123"


def test_extract_bearer_token_from_meta_extra_token():
    """Should extract token from meta.extra.token."""
    ctx = MagicMock(spec=[])
    ctx.meta = MagicMock()
    ctx.meta.extra = {"token": "xyz789"}
    assert mcp_server._extract_bearer_token(ctx) == "xyz789"


def test_extract_bearer_token_from_meta_extra_authorization():
    """Should extract token from meta.extra.authorization."""
    ctx = MagicMock(spec=[])
    ctx.meta = MagicMock()
    ctx.meta.extra = {"authorization": "Bearer def456"}
    assert mcp_server._extract_bearer_token(ctx) == "def456"


def test_extract_bearer_token_none_when_missing():
    """Should return None when no token is available."""
    ctx = MagicMock(spec=[])
    assert mcp_server._extract_bearer_token(ctx) is None


def test_extract_bearer_token_raw_token_in_extra():
    """Should return raw token string from meta.extra.token."""
    ctx = MagicMock(spec=[])
    ctx.meta = MagicMock()
    ctx.meta.extra = {"token": "raw_token_no_bearer"}
    assert mcp_server._extract_bearer_token(ctx) == "raw_token_no_bearer"


def test_extract_bearer_token_empty_extra_dict():
    """Should return None when extra dict has no relevant keys."""
    ctx = MagicMock(spec=[])
    ctx.meta = MagicMock()
    ctx.meta.extra = {"unrelated": "value"}
    assert mcp_server._extract_bearer_token(ctx) is None


# --- chat tool tests ---


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_returns_agent_reply(mock_session, mock_discover, mock_build):
    """chat tool should return the agent's AI reply."""
    # Mock MCP session
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    # Mock tools
    mock_discover.return_value = [MagicMock()]

    # Mock agent response
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Here are some ceramic cups!"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    ctx.headers = {"authorization": "Bearer test_token"}
    result = await chat_fn(ctx=ctx, message="find cups", thread_id="t1")

    assert result == {"reply": "Here are some ceramic cups!"}
    mock_session.assert_called_once_with(token="test_token")

    # Verify the agent was called with the new state format
    call_args = mock_agent.ainvoke.call_args
    state = call_args[0][0]
    assert "messages" in state
    assert "auth_token" in state
    assert "needs_confirm" in state
    assert "confirmed" in state
    assert state["auth_token"] == "test_token"


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_without_token(mock_session, mock_discover, mock_build):
    """chat tool should work without auth token (PUBLIC tools)."""
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_discover.return_value = [MagicMock()]

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "We have vases and cups."

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock(spec=[])  # No headers
    result = await chat_fn(ctx=ctx, message="what products?", thread_id="t1")

    assert result == {"reply": "We have vases and cups."}
    mock_session.assert_called_once_with(token=None)

    # Verify auth_token is None in state
    call_args = mock_agent.ainvoke.call_args
    state = call_args[0][0]
    assert state["auth_token"] is None


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_skips_non_ai_messages(mock_session, mock_discover, mock_build):
    """chat should skip tool messages and return the AI one."""
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_discover.return_value = []

    tool_msg = MagicMock()
    tool_msg.type = "tool"
    tool_msg.content = "raw data"

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Summary."

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [tool_msg, ai_msg]}
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock(spec=[])
    result = await chat_fn(ctx=ctx, message="test", thread_id="t1")
    assert result == {"reply": "Summary."}


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_handles_dict_messages(mock_session, mock_discover, mock_build):
    """chat should handle dict format messages from the new graph."""
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_discover.return_value = []

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [{"role": "assistant", "content": "Here's your response!"}]
    }
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock(spec=[])
    result = await chat_fn(ctx=ctx, message="test", thread_id="t1")
    assert result == {"reply": "Here's your response!"}


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_fallback_when_no_ai_content(
    mock_session, mock_discover, mock_build
):
    """chat should return fallback when AI message has empty content."""
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_discover.return_value = []

    empty_msg = MagicMock()
    empty_msg.type = "ai"
    empty_msg.content = ""

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [empty_msg]}
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock(spec=[])
    result = await chat_fn(ctx=ctx, message="hi", thread_id="t1")
    assert result == {"reply": "I'm sorry, I couldn't process your request."}


@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_raises_tool_error_on_exception(mock_session):
    """chat should raise ToolError when agent invocation fails."""
    mock_session.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock(spec=[])
    with pytest.raises(ToolError, match="something went wrong"):
        await chat_fn(ctx=ctx, message="hi", thread_id="t1")


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.mcp_session")
async def test_chat_initializes_state_correctly(
    mock_session, mock_discover, mock_build
):
    """chat should initialize state with correct default values."""
    mock_sess = AsyncMock()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_discover.return_value = []

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [{"role": "assistant", "content": "test response"}]
    }
    mock_build.return_value = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    ctx.headers = {"authorization": "Bearer test123"}
    await chat_fn(ctx=ctx, message="hello", thread_id="thread123")

    # Check the initial state passed to agent
    call_args = mock_agent.ainvoke.call_args
    initial_state = call_args[0][0]

    expected_state = {
        "messages": [{"role": "user", "content": "hello"}],
        "auth_token": "test123",
        "needs_confirm": False,
        "confirmed": False,
    }

    assert initial_state == expected_state

    # Check the config
    config = call_args[1]["config"]
    assert config["configurable"]["thread_id"] == "thread123"


# --- reset tool tests ---


async def test_reset_returns_ok():
    """reset should return ok status with thread info."""
    mcp = mcp_server.create_mcp_server()
    reset_fn = mcp._tool_manager._tools["reset"].fn

    result = await reset_fn(thread_id="my-thread")
    assert result["status"] == "ok"
    assert "my-thread" in result["message"]


async def test_reset_default_thread():
    """reset should work with default thread_id."""
    mcp = mcp_server.create_mcp_server()
    reset_fn = mcp._tool_manager._tools["reset"].fn

    result = await reset_fn(thread_id="default")
    assert result["status"] == "ok"
    assert "default" in result["message"]


# --- health endpoint tests ---


def test_create_mcp_server_has_health_route():
    """create_mcp_server should register a /health custom route."""
    mcp = mcp_server.create_mcp_server()
    route_paths = [r.path for r in mcp._custom_starlette_routes]
    assert "/health" in route_paths
