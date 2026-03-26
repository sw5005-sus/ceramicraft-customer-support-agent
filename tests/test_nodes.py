"""Tests for the nodes module (chitchat and escalate)."""

from unittest.mock import MagicMock, patch

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


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
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


@patch("ceramicraft_customer_support_agent.nodes.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.nodes.logger")
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
