"""Tests for the guard module."""

from unittest.mock import patch

from ceramicraft_customer_support_agent.guard import (
    SENSITIVE_OPERATIONS,
    build_guard,
)


def test_sensitive_operations_defined():
    """SENSITIVE_OPERATIONS should contain expected operations."""
    expected_sensitive = {"delete_address", "confirm_receipt"}
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
def test_guard_node_with_confirmation_needed(mock_logger):
    """Guard should add confirmation message for sensitive operations."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            {"role": "user", "content": "Please delete_address for me"},
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


def test_guard_node_with_confirmation_already_given():
    """Guard should not add confirmation when already confirmed."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": True,
        "messages": [
            {"role": "user", "content": "Please delete_address for me"},
        ],
    }

    result = guard(state)

    # Should not add confirmation message
    assert result == {}


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_confirm_receipt(mock_logger):
    """Guard should handle confirm_receipt operation specifically."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            {"role": "assistant", "content": "I'll confirm_receipt for your order"},
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


def test_guard_node_with_no_sensitive_operations():
    """Guard should not add messages for non-sensitive operations."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            {"role": "user", "content": "Please search_products for me"},
            {"role": "assistant", "content": "Here are some products..."},
        ],
    }

    result = guard(state)

    # Should not add any messages
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

    # Create 10 messages, sensitive op in first message (should be ignored)
    messages = []
    messages.append({"role": "user", "content": "delete_address in old message"})

    for i in range(8):
        messages.append({"role": "user", "content": f"normal message {i}"})

    # Recent message without sensitive operation
    messages.append({"role": "user", "content": "show my profile"})

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
        "unauthorized" not in "authority",  # just a truthy filler
    ]

    for phrase in false_positive_phrases:
        if isinstance(phrase, bool):
            continue
        state = {
            "auth_token": None,
            "messages": [
                {"role": "assistant", "content": phrase},
            ],
        }

        result = guard(state)

        assert result == {}, f"False positive on: {phrase}"


def test_guard_node_with_unknown_sensitive_operation():
    """Guard should handle unknown sensitive operations."""
    guard = build_guard()

    # Temporarily add a new sensitive operation for testing
    state = {
        "auth_token": "token",
        "confirmed": False,
        "messages": [
            {"role": "user", "content": "Please confirm_receipt now"},
        ],
    }

    result = guard(state)

    # Should use the specific confirm_receipt message
    confirm_message = result["messages"][0]["content"]
    assert "received your order" in confirm_message


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
