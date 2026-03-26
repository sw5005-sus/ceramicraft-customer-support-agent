"""Tests for the agent module."""

from unittest.mock import MagicMock, patch

from ceramicraft_customer_support_agent.agent import build_agent, get_memory


@patch("ceramicraft_customer_support_agent.agent.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.agent.create_agent")
def test_build_agent_calls_create_agent(mock_create, mock_llm_cls):
    """build_agent should call create_agent with the right args."""
    mock_tool = MagicMock()
    mock_create.return_value = MagicMock()

    agent = build_agent([mock_tool])

    mock_llm_cls.assert_called_once()
    mock_create.assert_called_once()
    assert agent is mock_create.return_value

    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["tools"] == [mock_tool]


@patch("ceramicraft_customer_support_agent.agent.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.agent.create_agent")
def test_build_agent_uses_configured_model(mock_create, mock_llm_cls):
    """build_agent should use the model from settings."""
    mock_create.return_value = MagicMock()

    build_agent([])

    call_kwargs = mock_llm_cls.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


@patch("ceramicraft_customer_support_agent.agent.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.agent.create_agent")
def test_build_agent_with_empty_tools(mock_create, mock_llm_cls):
    """build_agent should work with empty tool list."""
    mock_create.return_value = MagicMock()

    agent = build_agent([])

    assert agent is mock_create.return_value
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["tools"] == []


@patch("ceramicraft_customer_support_agent.agent.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.agent.create_agent")
def test_build_agent_passes_system_prompt(mock_create, mock_llm_cls):
    """build_agent should pass the system prompt."""
    mock_create.return_value = MagicMock()

    build_agent([])

    call_kwargs = mock_create.call_args
    assert "system_prompt" in call_kwargs.kwargs
    assert "CeramiCraft" in call_kwargs.kwargs["system_prompt"]


@patch("ceramicraft_customer_support_agent.agent.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.agent.create_agent")
def test_build_agent_uses_shared_memory(mock_create, mock_llm_cls):
    """build_agent should use the shared MemorySaver."""
    mock_create.return_value = MagicMock()

    build_agent([])

    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs["checkpointer"] is get_memory()


def test_get_memory_returns_same_instance():
    """get_memory should always return the same MemorySaver."""
    assert get_memory() is get_memory()
