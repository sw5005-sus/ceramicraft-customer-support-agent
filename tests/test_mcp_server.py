"""Tests for the MCP server module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ceramicraft_customer_support_agent import mcp_server


@pytest.fixture(autouse=True)
def _reset_agent():
    """Reset module-level agent before each test."""
    mcp_server._agent = None
    yield
    mcp_server._agent = None


def test_create_mcp_server_returns_fastmcp():
    """create_mcp_server should return a FastMCP instance."""
    mcp = mcp_server.create_mcp_server()
    assert mcp is not None
    assert mcp.name == "CeramiCraft Customer Support"


def test_create_mcp_server_registers_tools():
    """create_mcp_server should register chat and reset_conversation tools."""
    mcp = mcp_server.create_mcp_server()
    tools = mcp._tool_manager._tools
    assert "chat" in tools
    assert "reset_conversation" in tools


# --- chat tool tests ---


async def test_chat_returns_error_when_agent_not_initialized():
    """chat tool should return error when agent is not set."""
    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    result = await chat_fn(ctx=ctx, message="hello", thread_id="t1")
    assert result == {"error": "Agent not initialized"}


async def test_chat_returns_agent_reply():
    """chat tool should return the agent's AI reply."""
    mock_msg = MagicMock()
    mock_msg.type = "ai"
    mock_msg.content = "Here are some great ceramic cups!"

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [mock_msg]}
    mcp_server._agent = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    result = await chat_fn(ctx=ctx, message="find cups", thread_id="t1")

    assert result == {"reply": "Here are some great ceramic cups!"}
    mock_agent.ainvoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "find cups"}]},
        config={"configurable": {"thread_id": "t1"}},
    )


async def test_chat_skips_non_ai_messages():
    """chat tool should skip tool messages and return the AI one."""
    tool_msg = MagicMock()
    tool_msg.type = "tool"
    tool_msg.content = "raw data"

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Here is a summary."

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [tool_msg, ai_msg]}
    mcp_server._agent = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    result = await chat_fn(ctx=ctx, message="test", thread_id="t1")
    assert result == {"reply": "Here is a summary."}


async def test_chat_fallback_when_no_ai_content():
    """chat tool should return fallback when no AI message has content."""
    empty_msg = MagicMock()
    empty_msg.type = "ai"
    empty_msg.content = ""

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": [empty_msg]}
    mcp_server._agent = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    result = await chat_fn(ctx=ctx, message="hi", thread_id="t1")
    assert result == {"reply": "I'm sorry, I couldn't process your request."}


async def test_chat_fallback_when_no_messages():
    """chat tool should return fallback when response has no messages."""
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": []}
    mcp_server._agent = mock_agent

    mcp = mcp_server.create_mcp_server()
    chat_fn = mcp._tool_manager._tools["chat"].fn

    ctx = MagicMock()
    result = await chat_fn(ctx=ctx, message="hi", thread_id="t1")
    assert result == {"reply": "I'm sorry, I couldn't process your request."}


# --- reset_conversation tool tests ---


async def test_reset_conversation_returns_ok():
    """reset_conversation should return ok status with thread info."""
    mcp = mcp_server.create_mcp_server()
    reset_fn = mcp._tool_manager._tools["reset_conversation"].fn

    result = await reset_fn(thread_id="my-thread")
    assert result["status"] == "ok"
    assert "my-thread" in result["message"]


async def test_reset_conversation_default_thread():
    """reset_conversation should work with default thread_id."""
    mcp = mcp_server.create_mcp_server()
    reset_fn = mcp._tool_manager._tools["reset_conversation"].fn

    result = await reset_fn(thread_id="default")
    assert result["status"] == "ok"
    assert "default" in result["message"]


# --- lifespan tests ---


@patch("ceramicraft_customer_support_agent.mcp_server.build_agent")
@patch("ceramicraft_customer_support_agent.mcp_server.discover_tools")
@patch("ceramicraft_customer_support_agent.mcp_server.streamablehttp_client")
async def test_lifespan_initializes_agent(mock_http_client, mock_discover, mock_build):
    """Lifespan should connect to MCP server and build agent."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    # streamablehttp_client returns (read, write, _)
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_http_client.return_value.__aenter__ = AsyncMock(
        return_value=(mock_read, mock_write, None)
    )
    mock_http_client.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_tools = [MagicMock(), MagicMock()]
    mock_discover.return_value = mock_tools
    mock_build.return_value = MagicMock(name="agent")

    server = MagicMock()

    with patch(
        "ceramicraft_customer_support_agent.mcp_server.ClientSession"
    ) as mock_cs:
        mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

        async with mcp_server._lifespan(server):
            assert mcp_server._agent is not None
            mock_discover.assert_called_once_with(mock_session)
            mock_build.assert_called_once_with(mock_tools)

    # After lifespan exits, agent should be None
    assert mcp_server._agent is None
