"""Tests for the nodes module (chitchat and escalate)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ceramicraft_customer_support_agent.nodes import (
    build_chitchat_node,
    build_escalate_node,
)


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
def test_build_chitchat_node(mock_llm_cls):
    """build_chitchat_node should create a callable node."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    assert callable(node)
    mock_llm_cls.assert_called_once()


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
async def test_chitchat_node_with_messages(mock_llm_cls):
    """Chitchat node should handle messages and return response."""
    mock_llm_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hello! How can I help you?"
    mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "Hi there!"

    state = {"messages": [mock_message]}

    result = await node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["content"] == "Hello! How can I help you?"

    mock_llm_instance.ainvoke.assert_called_once()

    # Verify role mapping: "human" → "user", not passed raw
    call_args = mock_llm_instance.ainvoke.call_args[0][0]
    user_msgs = [m for m in call_args if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Hi there!"
    # Should NOT contain "human" as a role
    assert all(m["role"] in ("system", "user", "assistant") for m in call_args)


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
async def test_chitchat_node_with_empty_messages(mock_llm_cls):
    """Chitchat node should handle empty messages."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    state = {"messages": []}

    result = await node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert "Hello" in result["messages"][0]["content"]

    mock_llm_instance.ainvoke.assert_not_called()


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
async def test_chitchat_node_propagates_llm_error(mock_llm_cls):
    """Chitchat node should let LLM errors propagate to the caller."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.ainvoke = AsyncMock(side_effect=Exception("API error"))
    mock_llm_cls.return_value = mock_llm_instance

    node = build_chitchat_node()

    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "test"

    state = {"messages": [mock_message]}

    with pytest.raises(Exception, match="API error"):
        await node(state)


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
