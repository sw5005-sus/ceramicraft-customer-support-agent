"""Tests for the guard module."""

from unittest.mock import patch

from ceramicraft_customer_support_agent.guard import build_guard


def test_build_guard():
    """build_guard should return a callable."""
    guard = build_guard()
    assert callable(guard)


@patch("ceramicraft_customer_support_agent.guard.logger")
def test_guard_node_with_auth_error_no_token(mock_logger):
    """Guard should add auth message when auth error detected and no token."""
    guard = build_guard()

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
    assert result == {}


def test_guard_node_with_no_auth_issues():
    """Guard should return empty dict for normal messages."""
    guard = build_guard()

    state = {
        "auth_token": "token",
        "messages": [
            {"role": "assistant", "content": "Here are your products."},
        ],
    }

    result = guard(state)
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
            {"role": "user"},
            {"content": "missing role"},
        ],
    }

    result = guard(state)
    assert isinstance(result, dict)


def test_guard_node_checks_recent_messages_only():
    """Guard should only check the last 5 messages."""
    guard = build_guard()

    messages = [
        {"role": "assistant", "content": "unauthorized access error"},
    ]
    for i in range(9):
        messages.append({"role": "user", "content": f"normal message {i}"})

    state = {"auth_token": None, "messages": messages}

    result = guard(state)
    # Auth error is in first message (outside last 5), should not trigger
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
        "messages": [{"role": "user", "content": "normal message"}],
        "intent": "browse",
        "custom_field": "should_remain",
    }

    result = guard(state)
    assert result == {}
    assert "custom_field" in state
