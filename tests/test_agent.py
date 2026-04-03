"""Tests for the agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

from ceramicraft_customer_support_agent.agent import build_agent


@patch("ceramicraft_customer_support_agent.agent.build_graph", new_callable=AsyncMock)
async def test_build_agent_calls_build_graph(mock_build_graph):
    """build_agent should call build_graph with the right args."""
    mock_tool = MagicMock()
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    agent = await build_agent([mock_tool])

    mock_build_graph.assert_called_once_with([mock_tool])
    assert agent is mock_graph


@patch("ceramicraft_customer_support_agent.agent.build_graph", new_callable=AsyncMock)
async def test_build_agent_with_empty_tools(mock_build_graph):
    """build_agent should work with empty tool list."""
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    agent = await build_agent([])

    mock_build_graph.assert_called_once_with([])
    assert agent is mock_graph


@patch("ceramicraft_customer_support_agent.agent.build_graph", new_callable=AsyncMock)
@patch("ceramicraft_customer_support_agent.agent.logger")
async def test_build_agent_logs_tool_count(mock_logger, mock_build_graph):
    """build_agent should log the number of tools."""
    mock_tools = [MagicMock(), MagicMock(), MagicMock()]
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    await build_agent(mock_tools)

    mock_logger.info.assert_called_once()
    log_call = mock_logger.info.call_args[0]
    assert "3 tools" in log_call[0] or "3" in str(log_call[1:])


@patch("ceramicraft_customer_support_agent.agent.build_graph", new_callable=AsyncMock)
async def test_build_agent_backward_compatibility(mock_build_graph):
    """build_agent should maintain backward compatibility with old interface."""
    mock_tool1 = MagicMock()
    mock_tool1.name = "test_tool"

    mock_tool2 = MagicMock()
    mock_tool2.name = "another_tool"

    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    tools = [mock_tool1, mock_tool2]
    result = await build_agent(tools)

    assert result is mock_graph
    mock_build_graph.assert_called_once_with(tools)


@patch("ceramicraft_customer_support_agent.agent.build_graph", new_callable=AsyncMock)
async def test_build_agent_handles_various_tool_types(mock_build_graph):
    """build_agent should handle different types of tool objects."""
    mock_graph = MagicMock()
    mock_build_graph.return_value = mock_graph

    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.description = "First tool"

    tool2 = MagicMock()
    tool2.name = "tool2"

    tools = [tool1, tool2]
    result = await build_agent(tools)

    assert result is mock_graph
    mock_build_graph.assert_called_once_with(tools)
