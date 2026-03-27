"""Tests for the MCP client module (PersistentMCPClient)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ceramicraft_customer_support_agent.mcp_client import (
    PersistentMCPClient,
    _AuthInterceptor,
    _current_auth_token,
    get_tools,
    reconnect_tools,
    set_auth_token,
)


@pytest.fixture()
def client():
    """Create a fresh PersistentMCPClient for each test."""
    return PersistentMCPClient()


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_get_tools_discovers_on_first_call(mock_load, client):
    """get_tools should discover tools via connection mode on first call."""
    mock_tool = MagicMock(name="search_products")
    mock_load.return_value = [mock_tool]

    tools = await client.get_tools()

    assert len(tools) == 1
    mock_load.assert_called_once()
    # Must pass session=None to enable connection mode
    assert mock_load.call_args[1]["session"] is None
    assert mock_load.call_args[1]["connection"]["transport"] == "streamable_http"
    assert "tool_interceptors" in mock_load.call_args[1]


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_get_tools_returns_cached_on_subsequent_calls(mock_load, client):
    """get_tools should return cached tools without reconnecting."""
    mock_load.return_value = [MagicMock()]

    tools1 = await client.get_tools()
    tools2 = await client.get_tools()

    assert tools1 is tools2
    assert mock_load.call_count == 1


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_reconnect_forces_new_discovery(mock_load, client):
    """reconnect should clear cache and re-discover tools."""
    mock_load.return_value = [MagicMock()]

    await client.get_tools()
    await client.reconnect()

    assert mock_load.call_count == 2


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_get_tools_empty_server(mock_load, client):
    """get_tools should handle server with no tools."""
    mock_load.return_value = []

    tools = await client.get_tools()
    assert tools == []


@patch("ceramicraft_customer_support_agent.mcp_client._mcp_client")
async def test_module_get_tools_delegates(mock_client):
    """Module-level get_tools() should delegate to the singleton."""
    mock_client.get_tools = AsyncMock(return_value=[MagicMock()])
    tools = await get_tools()
    mock_client.get_tools.assert_awaited_once()
    assert len(tools) == 1


@patch("ceramicraft_customer_support_agent.mcp_client._mcp_client")
async def test_module_reconnect_tools_delegates(mock_client):
    """Module-level reconnect_tools() should delegate to the singleton."""
    mock_client.reconnect = AsyncMock(return_value=[MagicMock()])
    tools = await reconnect_tools()
    mock_client.reconnect.assert_awaited_once()
    assert len(tools) == 1


# ── _AuthInterceptor tests ──


async def test_auth_interceptor_injects_token():
    """_AuthInterceptor should inject Bearer token into headers."""
    interceptor = _AuthInterceptor()
    handler = AsyncMock(return_value="result")

    request = MagicMock()
    request.headers = None

    token = _current_auth_token.set("test_jwt_token")
    try:
        await interceptor(request, handler)
    finally:
        _current_auth_token.reset(token)

    request.override.assert_called_once()
    call_kwargs = request.override.call_args[1]
    assert call_kwargs["headers"]["authorization"] == "Bearer test_jwt_token"
    handler.assert_awaited_once_with(request.override.return_value)


async def test_auth_interceptor_skips_when_no_token():
    """_AuthInterceptor should pass through when no token set."""
    interceptor = _AuthInterceptor()
    handler = AsyncMock(return_value="result")

    request = MagicMock()

    token = _current_auth_token.set(None)
    try:
        result = await interceptor(request, handler)
    finally:
        _current_auth_token.reset(token)

    request.override.assert_not_called()
    handler.assert_awaited_once_with(request)
    assert result == "result"


async def test_auth_interceptor_merges_existing_headers():
    """_AuthInterceptor should merge with existing headers."""
    interceptor = _AuthInterceptor()
    handler = AsyncMock(return_value="result")

    request = MagicMock()
    request.headers = {"x-custom": "value"}

    token = _current_auth_token.set("my_token")
    try:
        await interceptor(request, handler)
    finally:
        _current_auth_token.reset(token)

    call_kwargs = request.override.call_args[1]
    assert call_kwargs["headers"]["authorization"] == "Bearer my_token"
    assert call_kwargs["headers"]["x-custom"] == "value"


def test_set_auth_token():
    """set_auth_token should update the context variable."""
    token = _current_auth_token.set(None)
    try:
        set_auth_token("abc123")
        assert _current_auth_token.get() == "abc123"

        set_auth_token(None)
        assert _current_auth_token.get() is None
    finally:
        _current_auth_token.reset(token)


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_get_tools_passes_auth_interceptor(mock_load, client):
    """get_tools should pass _AuthInterceptor to load_mcp_tools."""
    mock_load.return_value = []

    await client.get_tools()

    interceptors = mock_load.call_args[1]["tool_interceptors"]
    assert len(interceptors) == 1
    assert isinstance(interceptors[0], _AuthInterceptor)


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_connection_mode_session_is_none(mock_load, client):
    """load_mcp_tools must be called with session=None for headers to work."""
    mock_load.return_value = []

    await client.get_tools()

    assert mock_load.call_args[1]["session"] is None
    assert mock_load.call_args[1]["connection"] is not None
