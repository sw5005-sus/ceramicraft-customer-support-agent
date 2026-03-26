"""Tests for prompt templates."""

from ceramicraft_customer_support_agent.prompts import SYSTEM_PROMPT


def test_system_prompt_is_nonempty_string():
    """System prompt should be a non-empty string."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 100


def test_system_prompt_mentions_ceramicraft():
    """System prompt should mention the platform name."""
    assert "CeramiCraft" in SYSTEM_PROMPT


def test_system_prompt_mentions_capabilities():
    """System prompt should outline key capabilities."""
    for keyword in ["product", "cart", "order", "review", "account"]:
        assert keyword.lower() in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_auth_distinction():
    """System prompt should distinguish login-required vs public capabilities."""
    assert (
        "login required" in SYSTEM_PROMPT.lower() or "Login required" in SYSTEM_PROMPT
    )


def test_system_prompt_mentions_error_handling():
    """System prompt should guide error handling behavior."""
    assert "error" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_language_matching():
    """System prompt should instruct language matching."""
    assert "language" in SYSTEM_PROMPT.lower()
