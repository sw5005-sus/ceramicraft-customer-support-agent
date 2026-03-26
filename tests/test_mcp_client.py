"""Tests for the MCP client module."""

from unittest.mock import AsyncMock, MagicMock, patch

from ceramicraft_customer_support_agent.mcp_client import discover_tools


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
