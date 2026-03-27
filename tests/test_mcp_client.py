"""Tests for the MCP client module (PersistentMCPClient)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ceramicraft_customer_support_agent.mcp_client import (
    PersistentMCPClient,
    get_tools,
    reconnect_tools,
)


@pytest.fixture()
def client():
    """Create a fresh PersistentMCPClient for each test."""
    return PersistentMCPClient()


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_get_tools_connects_on_first_call(mock_http, mock_cs, mock_load, client):
    """get_tools should open a session and discover tools on first call."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_tool1 = MagicMock(name="search_products")
    mock_tool2 = MagicMock(name="get_product")
    mock_load.return_value = [mock_tool1, mock_tool2]

    tools = await client.get_tools()

    assert len(tools) == 2
    mock_load.assert_called_once_with(mock_session)
    mock_session.initialize.assert_called_once()


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_get_tools_returns_cached_on_subsequent_calls(
    mock_http, mock_cs, mock_load, client
):
    """get_tools should return cached tools without reconnecting."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_load.return_value = [MagicMock()]

    tools1 = await client.get_tools()
    tools2 = await client.get_tools()

    assert tools1 is tools2
    # load_mcp_tools should only be called once (cached)
    assert mock_load.call_count == 1


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_reconnect_forces_new_session(mock_http, mock_cs, mock_load, client):
    """reconnect should close old session and create a new one."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_load.return_value = [MagicMock()]

    await client.get_tools()
    await client.reconnect()

    # Should have called load_mcp_tools twice (initial + reconnect)
    assert mock_load.call_count == 2


async def test_close_handles_no_session():
    """_close should handle case where no session exists."""
    client = PersistentMCPClient()
    # Should not raise
    await client._close()


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_get_tools_empty_server(mock_http, mock_cs, mock_load, client):
    """get_tools should handle server with no tools."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

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
