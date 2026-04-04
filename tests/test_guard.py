"""Tests for the guard module."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from ceramicraft_customer_support_agent.guard import (
    SENSITIVE_OPERATIONS,
    build_guard,
)


def test_sensitive_operations_defined():
    """SENSITIVE_OPERATIONS should contain expected operations."""
    expected_sensitive = {"delete_address", "confirm_receipt", "create_order"}
    assert SENSITIVE_OPERATIONS == expected_sensitive


def test_build_guard():
    """build_guard should return a callable."""
    guard = build_guard()

    assert callable(guard)


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_auth_error_no_token(mock_logger):
    """Guard should add auth message when auth error detected and no token."""
    guard = build_guard()

    # Mock state with auth error in messages
    state = {
        "auth_token": None,
        "messages": [
            {"role": "assistant", "content": "Authentication required to access cart"},
        ],
    }

    result = guard(state)

    assert "messages" in result
    assert len(result["messages"]) == 1

    auth_message = result["messages"][0]["content"]
    assert "log in" in auth_message.lower()
    assert "sign in" in auth_message.lower()

    mock_logger.info.assert_called_once_with("Added auth requirement message")


def test_guard_node_with_auth_error_has_token():
    """Guard should not add auth message when token exists."""
    guard = build_guard()

    state = {
        "auth_token": "valid_token",
        "messages": [
            {"role": "assistant", "content": "Authentication required to access cart"},
        ],
    }

    result = guard(state)

    # Should not add any new messages
    assert result == {}


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_confirmation_needed_via_tool_message(mock_logger):
    """Guard should add confirmation when a sensitive ToolMessage is detected."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            ToolMessage(
                content="Address deleted", name="delete_address", tool_call_id="tc1"
            ),
        ],
    }

    result = guard(state)

    assert "messages" in result
    assert "needs_confirm" in result

    assert result["needs_confirm"] is True
    assert len(result["messages"]) == 1

    confirm_message = result["messages"][0]["content"]
    assert "sure" in confirm_message.lower()
    assert "delete this address" in confirm_message.lower()

    mock_logger.info.assert_called_once_with(
        "Added confirmation requirement for %s", "delete_address"
    )


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_confirmation_needed_via_ai_tool_calls(mock_logger):
    """Guard should detect sensitive ops from AIMessage.tool_calls."""
    guard = build_guard()

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "delete_address", "args": {"id": "123"}}],
    )

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [ai_msg],
    }

    result = guard(state)

    assert "messages" in result
    assert result["needs_confirm"] is True

    mock_logger.info.assert_called_once_with(
        "Added confirmation requirement for %s", "delete_address"
    )


def test_guard_node_with_confirmation_already_given():
    """Guard should not add confirmation when already confirmed."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": True,
        "messages": [
            ToolMessage(
                content="Address deleted", name="delete_address", tool_call_id="tc1"
            ),
        ],
    }

    result = guard(state)

    # Should not add confirmation message, but should reset flags
    assert result == {"confirmed": False, "needs_confirm": False}


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_confirm_receipt(mock_logger):
    """Guard should handle confirm_receipt operation specifically."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            ToolMessage(
                content="Receipt confirmed", name="confirm_receipt", tool_call_id="tc1"
            ),
        ],
    }

    result = guard(state)

    assert "messages" in result
    assert "needs_confirm" in result

    confirm_message = result["messages"][0]["content"]
    assert "received your order" in confirm_message.lower()
    assert "satisfied" in confirm_message.lower()

    mock_logger.info.assert_called_once_with(
        "Added confirmation requirement for %s", "confirm_receipt"
    )


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_create_order(mock_logger):
    """Guard should handle create_order as a sensitive operation."""
    guard = build_guard()

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "create_order", "args": {}}],
    )

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [ai_msg],
    }

    result = guard(state)

    assert "messages" in result
    assert result["needs_confirm"] is True

    confirm_message = result["messages"][0]["content"]
    assert "order" in confirm_message.lower()

    mock_logger.info.assert_called_once_with(
        "Added confirmation requirement for %s", "create_order"
    )


def test_guard_node_with_no_sensitive_operations():
    """Guard should not add messages for non-sensitive operations."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            ToolMessage(
                content="Found products", name="search_products", tool_call_id="tc1"
            ),
        ],
    }

    result = guard(state)

    # Should not add any messages
    assert result == {}


def test_guard_node_no_false_positive_on_content_text():
    """Guard should NOT trigger on content that merely mentions tool names."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            {
                "role": "assistant",
                "content": "I can help you delete_address if you want.",
            },
        ],
    }

    result = guard(state)

    # Plain dict messages have no .name attribute — should not trigger
    assert result == {}


def test_guard_node_with_empty_messages():
    """Guard should handle empty messages gracefully."""
    guard = build_guard()

    state = {"auth_token": "token", "messages": []}

    result = guard(state)

    assert result == {}


def test_guard_node_with_malformed_messages():
    """Guard should handle malformed messages gracefully."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "messages": [
            "invalid message format",
            {"role": "user"},  # missing content
            {"content": "missing role"},
        ],
    }

    result = guard(state)

    # Should not crash, might return empty dict
    assert isinstance(result, dict)


def test_guard_node_checks_recent_messages_only():
    """Guard should only check the last 5 messages."""
    guard = build_guard()

    # Create 10 messages, sensitive ToolMessage in first message (should be ignored)
    messages = []
    messages.append(
        ToolMessage(content="deleted", name="delete_address", tool_call_id="tc1")
    )

    for i in range(9):
        messages.append({"role": "user", "content": f"normal message {i}"})

    state = {"auth_token": "token", "confirmed": False, "messages": messages}

    result = guard(state)

    # Should not add confirmation since sensitive op was in old messages
    assert result == {}


def test_guard_node_with_multiple_auth_keywords():
    """Guard should detect various auth-related keywords."""
    guard = build_guard()

    auth_phrases = [
        "authentication required",
        "login required",
        "unauthorized access",
        "unauthenticated request",
    ]

    for phrase in auth_phrases:
        state = {
            "auth_token": None,
            "messages": [
                {"role": "assistant", "content": f"Error: {phrase}"},
            ],
        }

        result = guard(state)

        assert "messages" in result, f"Failed to detect: {phrase}"
        assert "log in" in result["messages"][0]["content"].lower()


def test_guard_node_no_false_positive_on_author():
    """Guard should NOT trigger on words like 'author' or 'authority'."""
    guard = build_guard()

    false_positive_phrases = [
        "The author of this review is John",
        "Local authority regulations apply",
        "authored by the team",
    ]

    for phrase in false_positive_phrases:
        state = {
            "auth_token": None,
            "messages": [
                {"role": "assistant", "content": phrase},
            ],
        }

        result = guard(state)

        assert result == {}, f"False positive on: {phrase}"


def test_guard_node_preserves_other_state():
    """Guard should not modify unrelated state fields."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [{"role": "user", "content": "normal message"}],
        "intent": "browse",
        "custom_field": "should_remain",
    }

    result = guard(state)

    # Should return empty dict, not modifying other fields
    assert result == {}
    # Original state should not be modified (guard doesn't mutate)
    assert "custom_field" in state
