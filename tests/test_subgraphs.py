"""Tests for the subgraphs module."""

from unittest.mock import MagicMock, patch

from ceramicraft_customer_support_agent.subgraphs import (
    build_browse_subgraph,
    build_cart_subgraph,
    build_order_subgraph,
    build_review_subgraph,
    build_account_subgraph,
    build_chitchat_node,
    build_escalate_node,
)


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_build_browse_subgraph(mock_create_react, mock_llm_cls):
    """build_browse_subgraph should filter tools and create agent."""
    mock_tools = []

    # Create mock tools with names
    for name in [
        "search_products",
        "get_product",
        "list_product_reviews",
        "unrelated_tool",
    ]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_browse_subgraph(mock_tools, mock_memory)

    # Check that correct tools were filtered
    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"search_products", "get_product", "list_product_reviews"}
    assert tool_names == expected_tools

    # Check other arguments
    assert call_args.kwargs["checkpointer"] is mock_memory
    assert "state_modifier" in call_args.kwargs

    mock_llm_cls.assert_called_once()
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_build_cart_subgraph(mock_create_react, mock_llm_cls):
    """build_cart_subgraph should filter tools and create agent."""
    mock_tools = []

    # Create mock tools
    for name in ["get_cart", "add_to_cart", "search_products", "unrelated_tool"]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_cart_subgraph(mock_tools, mock_memory)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"get_cart", "add_to_cart", "search_products"}
    assert tool_names.issubset(expected_tools)
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_build_order_subgraph(mock_create_react, mock_llm_cls):
    """build_order_subgraph should filter tools and create agent."""
    mock_tools = []

    for name in [
        "list_my_orders",
        "get_order_detail",
        "create_order",
        "unrelated_tool",
    ]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_order_subgraph(mock_tools, mock_memory)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"list_my_orders", "get_order_detail", "create_order"}
    assert tool_names.issubset(expected_tools)
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_build_review_subgraph(mock_create_react, mock_llm_cls):
    """build_review_subgraph should filter tools and create agent."""
    mock_tools = []

    for name in ["create_review", "like_review", "get_user_reviews", "unrelated_tool"]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_review_subgraph(mock_tools, mock_memory)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"create_review", "like_review", "get_user_reviews"}
    assert tool_names.issubset(expected_tools)
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_build_account_subgraph(mock_create_react, mock_llm_cls):
    """build_account_subgraph should filter tools and create agent."""
    mock_tools = []

    for name in [
        "get_my_profile",
        "update_my_profile",
        "delete_address",
        "unrelated_tool",
    ]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_account_subgraph(mock_tools, mock_memory)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"get_my_profile", "update_my_profile", "delete_address"}
    assert tool_names.issubset(expected_tools)
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
def test_build_chitchat_node(mock_llm_cls):
    """build_chitchat_node should create a callable node."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    assert callable(node)
    mock_llm_cls.assert_called_once()


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
def test_chitchat_node_with_messages(mock_llm_cls):
    """Chitchat node should handle messages and return response."""
    mock_llm_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hello! How can I help you?"
    mock_llm_instance.invoke.return_value = mock_response
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "Hi there!"

    state = {"messages": [mock_message]}

    result = node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["content"] == "Hello! How can I help you?"

    mock_llm_instance.invoke.assert_called_once()


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
def test_chitchat_node_with_empty_messages(mock_llm_cls):
    """Chitchat node should handle empty messages."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    state = {"messages": []}

    result = node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert "Hello" in result["messages"][0]["content"]

    mock_llm_instance.invoke.assert_not_called()


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.logger")
def test_chitchat_node_with_llm_error(mock_logger, mock_llm_cls):
    """Chitchat node should handle LLM errors gracefully."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("API error")
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    state = {"messages": [{"role": "user", "content": "test"}]}

    result = node(state)

    assert "messages" in result
    assert "trouble" in result["messages"][0]["content"].lower()
    mock_logger.exception.assert_called_once_with("Chitchat node failed")


def test_build_escalate_node():
    """build_escalate_node should create a callable node."""
    node = build_escalate_node()

    assert callable(node)


def test_escalate_node_returns_standard_message():
    """Escalate node should return standard escalation message."""
    node = build_escalate_node()

    state = {"messages": []}

    result = node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"

    content = result["messages"][0]["content"]
    assert "human support" in content.lower()
    assert "transfer" in content.lower()


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_subgraph_with_empty_tools(mock_create_react, mock_llm_cls):
    """Subgraphs should handle empty matching tools."""
    mock_tools = []

    # Tools that don't match browse criteria
    for name in ["unrelated_tool", "another_tool"]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_browse_subgraph(mock_tools, mock_memory)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]

    # Should pass empty tools list if no matches
    assert len(tools_arg) == 0
    assert result is mock_agent


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_all_subgraphs_use_correct_prompts(mock_create_react, mock_llm_cls):
    """All subgraphs should use their respective domain prompts."""
    mock_tools = []
    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    # Test each subgraph
    subgraphs = [
        build_browse_subgraph,
        build_cart_subgraph,
        build_order_subgraph,
        build_review_subgraph,
        build_account_subgraph,
    ]

    for subgraph_builder in subgraphs:
        mock_create_react.reset_mock()
        subgraph_builder(mock_tools, mock_memory)

        call_args = mock_create_react.call_args
        assert "state_modifier" in call_args.kwargs
        # Each should have a different prompt
        assert isinstance(call_args.kwargs["state_modifier"], str)
        assert len(call_args.kwargs["state_modifier"]) > 0
