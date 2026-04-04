"""Tests for the subgraphs module."""

from unittest.mock import MagicMock, patch

from ceramicraft_customer_support_agent.subgraphs import (
    build_browse_subgraph,
    build_cart_subgraph,
    build_order_subgraph,
    build_review_subgraph,
    build_account_subgraph,
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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_browse_subgraph(mock_tools)

    # Check that correct tools were filtered
    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"search_products", "get_product", "list_product_reviews"}
    assert tool_names == expected_tools

    # Subgraphs should NOT have a checkpointer (stateless per-invocation)
    assert "checkpointer" not in call_args.kwargs

    assert "prompt" in call_args.kwargs

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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_cart_subgraph(mock_tools)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"get_cart", "add_to_cart", "search_products"}
    assert tool_names.issubset(expected_tools)
    assert "checkpointer" not in call_args.kwargs
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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_order_subgraph(mock_tools)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"list_my_orders", "get_order_detail", "create_order"}
    assert tool_names.issubset(expected_tools)
    assert "checkpointer" not in call_args.kwargs
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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_review_subgraph(mock_tools)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"create_review", "like_review", "get_user_reviews"}
    assert tool_names.issubset(expected_tools)
    assert "checkpointer" not in call_args.kwargs
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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_account_subgraph(mock_tools)

    call_args = mock_create_react.call_args
    tools_arg = call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools_arg}

    expected_tools = {"get_my_profile", "update_my_profile", "delete_address"}
    assert tool_names.issubset(expected_tools)
    assert "checkpointer" not in call_args.kwargs
    assert result is mock_agent


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

    mock_agent = MagicMock()
    mock_create_react.return_value = mock_agent

    result = build_browse_subgraph(mock_tools)

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
        subgraph_builder(mock_tools)

        call_args = mock_create_react.call_args
        assert "prompt" in call_args.kwargs
        # Each should have a different prompt
        assert isinstance(call_args.kwargs["prompt"], str)
        assert len(call_args.kwargs["prompt"]) > 0


@patch("ceramicraft_customer_support_agent.subgraphs.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.subgraphs.create_react_agent")
def test_tools_have_handle_tool_error_enabled(mock_create_react, mock_llm_cls):
    """All tools passed to subgraphs should have handle_tool_error=True."""
    mock_tools = []
    for name in ["search_products", "get_product", "list_product_reviews"]:
        tool = MagicMock()
        tool.name = name
        mock_tools.append(tool)

    mock_create_react.return_value = MagicMock()
    build_browse_subgraph(mock_tools)

    call_args = mock_create_react.call_args
    for tool in call_args.kwargs["tools"]:
        assert tool.handle_tool_error is True
