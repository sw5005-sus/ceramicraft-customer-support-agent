"""Tests for the graph module."""

from unittest.mock import MagicMock, patch

from ceramicraft_customer_support_agent.graph import (
    AgentState,
    _checkpointer,
    build_graph,
    route_by_intent,
)


def test_agent_state_structure():
    """AgentState should have the required fields."""
    state: AgentState = {
        "messages": [],
        "intent": "browse",
        "auth_token": "token123",
        "needs_confirm": False,
        "confirmed": True,
    }

    assert "messages" in state
    assert "intent" in state
    assert "auth_token" in state
    assert "needs_confirm" in state
    assert "confirmed" in state


@patch("ceramicraft_customer_support_agent.graph.build_classifier")
@patch("ceramicraft_customer_support_agent.graph.build_guard")
@patch("ceramicraft_customer_support_agent.graph.build_browse_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_cart_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_order_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_review_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_account_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_chitchat_node")
@patch("ceramicraft_customer_support_agent.graph.build_escalate_node")
@patch("ceramicraft_customer_support_agent.graph.StateGraph")
def test_build_graph_creates_all_nodes(
    mock_state_graph,
    mock_escalate,
    mock_chitchat,
    mock_account,
    mock_review,
    mock_order,
    mock_cart,
    mock_browse,
    mock_guard,
    mock_classifier,
):
    """build_graph should create all required nodes."""
    mock_tools = [MagicMock()]
    mock_graph_instance = MagicMock()
    mock_state_graph.return_value = mock_graph_instance
    mock_graph_instance.compile.return_value = MagicMock()

    # Mock all the builders
    mock_classifier.return_value = MagicMock()
    mock_guard.return_value = MagicMock()
    mock_browse.return_value = MagicMock()
    mock_cart.return_value = MagicMock()
    mock_order.return_value = MagicMock()
    mock_review.return_value = MagicMock()
    mock_account.return_value = MagicMock()
    mock_chitchat.return_value = MagicMock()
    mock_escalate.return_value = MagicMock()

    result = build_graph(mock_tools)

    # Check that StateGraph was created with AgentState
    mock_state_graph.assert_called_once_with(AgentState)

    # Check that all nodes were added
    expected_nodes = [
        "classifier",
        "browse",
        "cart",
        "order",
        "review",
        "account",
        "chitchat",
        "escalate",
        "guard",
    ]

    add_node_calls = [
        call[0][0] for call in mock_graph_instance.add_node.call_args_list
    ]
    for node in expected_nodes:
        assert node in add_node_calls

    # Check that entry point was set
    mock_graph_instance.set_entry_point.assert_called_once_with("classifier")

    # Check that finish point was set
    mock_graph_instance.set_finish_point.assert_called_once_with("guard")

    # Check compilation was called
    mock_graph_instance.compile.assert_called_once()

    assert result is mock_graph_instance.compile.return_value


def test_route_by_intent_with_valid_intent():
    """route_by_intent should return the intent value."""
    state = {"intent": "browse", "messages": []}

    result = route_by_intent(state)  # ty: ignore[invalid-argument-type]

    assert result == "browse"


def test_route_by_intent_with_missing_intent():
    """route_by_intent should default to chitchat if no intent."""
    state = {"messages": []}

    result = route_by_intent(state)  # ty: ignore[invalid-argument-type]

    assert result == "chitchat"


def test_route_by_intent_with_all_intents():
    """route_by_intent should work with all expected intents."""
    intents = ["browse", "cart", "order", "review", "account", "chitchat", "escalate"]

    for intent in intents:
        state = {"intent": intent, "messages": []}
        result = route_by_intent(state)  # ty: ignore[invalid-argument-type]
        assert result == intent


@patch("ceramicraft_customer_support_agent.graph.build_classifier")
@patch("ceramicraft_customer_support_agent.graph.build_guard")
@patch("ceramicraft_customer_support_agent.graph.build_browse_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_cart_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_order_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_review_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_account_subgraph")
@patch("ceramicraft_customer_support_agent.graph.build_chitchat_node")
@patch("ceramicraft_customer_support_agent.graph.build_escalate_node")
@patch("ceramicraft_customer_support_agent.graph.StateGraph")
def test_build_graph_with_empty_tools(
    mock_state_graph,
    mock_escalate,
    mock_chitchat,
    mock_account,
    mock_review,
    mock_order,
    mock_cart,
    mock_browse,
    mock_guard,
    mock_classifier,
):
    """build_graph should work with empty tools list."""
    mock_graph_instance = MagicMock()
    mock_state_graph.return_value = mock_graph_instance
    mock_graph_instance.compile.return_value = MagicMock()

    # Mock all the builders
    mock_classifier.return_value = MagicMock()
    mock_guard.return_value = MagicMock()
    mock_browse.return_value = MagicMock()
    mock_cart.return_value = MagicMock()
    mock_order.return_value = MagicMock()
    mock_review.return_value = MagicMock()
    mock_account.return_value = MagicMock()
    mock_chitchat.return_value = MagicMock()
    mock_escalate.return_value = MagicMock()

    result = build_graph([])

    # Should still create the graph structure
    assert result is mock_graph_instance.compile.return_value

    # Should have passed empty tools to all subgraph builders
    mock_browse.assert_called_once()
    mock_cart.assert_called_once()
    mock_order.assert_called_once()
    mock_review.assert_called_once()
    mock_account.assert_called_once()


@patch("ceramicraft_customer_support_agent.graph.logger")
def test_route_by_intent_logs_routing(mock_logger):
    """route_by_intent should log the routing decision."""
    state = {"intent": "cart", "messages": []}

    route_by_intent(state)  # ty: ignore[invalid-argument-type]

    mock_logger.info.assert_called_once_with("Routing to %s based on intent", "cart")


def test_checkpointer_is_shared():
    """Module-level _checkpointer should be a single MemorySaver instance."""
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(_checkpointer, MemorySaver)
