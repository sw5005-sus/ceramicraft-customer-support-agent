"""Tests for the graph module."""

from unittest.mock import MagicMock, patch

import pytest

from ceramicraft_customer_support_agent.graph import (
    AgentState,
    _trim_messages,
    build_checkpointer,
    build_graph,
    route_by_intent,
)


def test_agent_state_structure():
    """AgentState should have the required fields."""
    state: AgentState = {
        "messages": [],
        "intent": "browse",
        "last_intent": "",
        "auth_token": "token123",
    }

    assert "messages" in state
    assert "intent" in state
    assert "last_intent" in state
    assert "auth_token" in state


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
async def test_build_graph_creates_all_nodes(
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

    result = await build_graph(mock_tools, checkpointer=MagicMock())

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

    # Check conditional routing edges were added from classifier
    mock_graph_instance.add_conditional_edges.assert_called_once()
    routing_arg = mock_graph_instance.add_conditional_edges.call_args[0]
    assert routing_arg[0] == "classifier"
    routing_map = mock_graph_instance.add_conditional_edges.call_args[0][2]
    for intent in [
        "browse",
        "cart",
        "order",
        "review",
        "account",
        "chitchat",
        "escalate",
    ]:
        assert intent in routing_map

    # Check all domain nodes have an edge to guard
    add_edge_calls = {call[0] for call in mock_graph_instance.add_edge.call_args_list}
    for domain in [
        "browse",
        "cart",
        "order",
        "review",
        "account",
        "chitchat",
        "escalate",
    ]:
        assert (domain, "guard") in add_edge_calls

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
async def test_build_graph_with_empty_tools(
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

    result = await build_graph([], checkpointer=MagicMock())

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


# --- _trim_messages tests ---


def test_trim_messages_under_limit():
    """Messages count < max should not be trimmed."""
    msgs = [{"role": "user", "content": str(i)} for i in range(5)]
    result = _trim_messages(msgs, 10)
    assert result == msgs


def test_trim_messages_over_limit():
    """Messages count > max should keep only the most recent N."""
    msgs = [{"role": "user", "content": str(i)} for i in range(25)]
    result = _trim_messages(msgs, 10)
    assert len(result) == 10
    assert result == msgs[-10:]


def test_trim_messages_zero_limit():
    """limit=0 should not trim (disabled)."""
    msgs = [{"role": "user", "content": str(i)} for i in range(100)]
    result = _trim_messages(msgs, 0)
    assert result == msgs


def test_trim_messages_exact_limit():
    """Messages count == max should not be trimmed."""
    msgs = [{"role": "user", "content": str(i)} for i in range(20)]
    result = _trim_messages(msgs, 20)
    assert result == msgs


# --- build_checkpointer tests ---


@patch("ceramicraft_customer_support_agent.graph.get_settings")
async def test_build_checkpointer_no_postgres_host_raises(mock_settings):
    """DATABASE_URL empty (POSTGRES_HOST unset) should raise RuntimeError."""
    mock_cfg = MagicMock()
    mock_cfg.DATABASE_URL = ""
    mock_settings.return_value = mock_cfg

    with pytest.raises(RuntimeError, match="PostgreSQL is not configured"):
        await build_checkpointer()


@patch("ceramicraft_customer_support_agent.graph.get_settings")
async def test_build_checkpointer_propagates_connection_error(mock_settings):
    """PostgreSQL connection failure should propagate, not silently fall back."""
    mock_cfg = MagicMock()
    mock_cfg.DATABASE_URL = "postgresql+psycopg://bad:pw@localhost/nodb"
    mock_settings.return_value = mock_cfg

    with (
        patch(
            "ceramicraft_customer_support_agent.graph.AsyncConnectionPool",
            side_effect=Exception("connection refused"),
        ),
        pytest.raises(Exception, match="connection refused"),
    ):
        await build_checkpointer()


# --- Settings.DATABASE_URL property tests ---


def test_settings_database_url_assembled():
    """DATABASE_URL is assembled from individual vars."""
    from ceramicraft_customer_support_agent.config import Settings

    s = Settings(
        POSTGRES_USER="ceramicraft",
        POSTGRES_PASSWORD="ceramicraft123",
        POSTGRES_HOST="postgres",
        POSTGRES_PORT=5432,
        CS_AGENT_DB_NAME="cs_agent_db",
    )
    assert s.DATABASE_URL == (
        "postgresql+psycopg://ceramicraft:ceramicraft123@postgres:5432/cs_agent_db"
    )


def test_settings_database_url_empty_when_no_host():
    """DATABASE_URL is empty when POSTGRES_HOST is not set."""
    from ceramicraft_customer_support_agent.config import Settings

    s = Settings(_env_file=None)  # ty: ignore[unknown-argument]
    assert s.DATABASE_URL == ""
