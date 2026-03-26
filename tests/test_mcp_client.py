"""Tests for the MCP client module."""

from unittest.mock import AsyncMock, MagicMock, patch

from ceramicraft_customer_support_agent.mcp_client import discover_tools, mcp_session


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_discover_tools_returns_tools(mock_load):
    """discover_tools should return tools from the MCP session."""
    mock_tool1 = MagicMock(name="search_products")
    mock_tool2 = MagicMock(name="get_product")
    mock_load.return_value = [mock_tool1, mock_tool2]

    session = AsyncMock()
    tools = await discover_tools(session)

    assert len(tools) == 2
    mock_load.assert_called_once_with(session)


@patch("ceramicraft_customer_support_agent.mcp_client.load_mcp_tools")
async def test_discover_tools_empty_server(mock_load):
    """discover_tools should handle server with no tools."""
    mock_load.return_value = []

    session = AsyncMock()
    tools = await discover_tools(session)

    assert tools == []


@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_mcp_session_without_token(mock_http, mock_cs):
    """mcp_session should connect without auth headers when no token."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

    async with mcp_session() as session:
        assert session is mock_session

    # No auth header
    call_kwargs = mock_http.call_args
    assert call_kwargs.kwargs.get("headers") == {}


@patch("ceramicraft_customer_support_agent.mcp_client.ClientSession")
@patch("ceramicraft_customer_support_agent.mcp_client.streamablehttp_client")
async def test_mcp_session_with_token(mock_http, mock_cs):
    """mcp_session should pass Authorization header when token is provided."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_http.return_value.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock(), None)
    )
    mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

    async with mcp_session(token="eyJtest") as session:
        assert session is mock_session

    call_kwargs = mock_http.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer eyJtest"
