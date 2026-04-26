"""Tests for the prompts module."""

from unittest.mock import MagicMock, patch

from ceramicraft_customer_support_agent.prompts import (
    ACCOUNT_PROMPT,
    BROWSE_PROMPT,
    CART_PROMPT,
    CHITCHAT_PROMPT,
    ORDER_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
    get_account_prompt,
    get_browse_prompt,
    get_cart_prompt,
    get_chitchat_prompt,
    get_order_prompt,
    get_prompt,
    get_review_prompt,
    get_system_prompt,
)


def test_system_prompt_exists():
    """SYSTEM_PROMPT should exist and contain key information."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert "CeramiCraft" in SYSTEM_PROMPT
    assert "ceramic" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_capabilities():
    """SYSTEM_PROMPT should mention key capabilities."""
    assert "search products" in SYSTEM_PROMPT.lower()
    assert "cart" in SYSTEM_PROMPT.lower()
    assert "orders" in SYSTEM_PROMPT.lower()
    assert "reviews" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_auth():
    """SYSTEM_PROMPT should mention login requirements."""
    assert "login" in SYSTEM_PROMPT.lower()
    assert "auth" in SYSTEM_PROMPT.lower() or "log in" in SYSTEM_PROMPT.lower()
    assert "read product reviews" in SYSTEM_PROMPT.lower()
    assert "Search products and view product details" in SYSTEM_PROMPT


def test_browse_prompt_exists():
    """BROWSE_PROMPT should exist and be appropriate for product browsing."""
    assert isinstance(BROWSE_PROMPT, str)
    assert len(BROWSE_PROMPT) > 0
    assert "product" in BROWSE_PROMPT.lower()
    assert "search" in BROWSE_PROMPT.lower()
    assert "CeramiCraft" in BROWSE_PROMPT


def test_cart_prompt_exists():
    """CART_PROMPT should exist and be appropriate for cart management."""
    assert isinstance(CART_PROMPT, str)
    assert len(CART_PROMPT) > 0
    assert "cart" in CART_PROMPT.lower()
    assert "add" in CART_PROMPT.lower()
    assert "CeramiCraft" in CART_PROMPT


def test_order_prompt_exists():
    """ORDER_PROMPT should exist and be appropriate for order management."""
    assert isinstance(ORDER_PROMPT, str)
    assert len(ORDER_PROMPT) > 0
    assert "order" in ORDER_PROMPT.lower()
    assert "CeramiCraft" in ORDER_PROMPT


def test_review_prompt_exists():
    """REVIEW_PROMPT should exist and be appropriate for review management."""
    assert isinstance(REVIEW_PROMPT, str)
    assert len(REVIEW_PROMPT) > 0
    assert "review" in REVIEW_PROMPT.lower()
    assert "CeramiCraft" in REVIEW_PROMPT


def test_account_prompt_exists():
    """ACCOUNT_PROMPT should exist and be appropriate for account management."""
    assert isinstance(ACCOUNT_PROMPT, str)
    assert len(ACCOUNT_PROMPT) > 0
    assert "account" in ACCOUNT_PROMPT.lower()
    assert "profile" in ACCOUNT_PROMPT.lower()
    assert "CeramiCraft" in ACCOUNT_PROMPT


def test_chitchat_prompt_exists():
    """CHITCHAT_PROMPT should exist and be appropriate for general conversation."""
    assert isinstance(CHITCHAT_PROMPT, str)
    assert len(CHITCHAT_PROMPT) > 0
    assert "CeramiCraft" in CHITCHAT_PROMPT
    assert "customer service" in CHITCHAT_PROMPT.lower()


def test_domain_prompts_are_distinct():
    """Each domain prompt should be unique and focused."""
    domain_prompts = [
        BROWSE_PROMPT,
        CART_PROMPT,
        ORDER_PROMPT,
        REVIEW_PROMPT,
        ACCOUNT_PROMPT,
        CHITCHAT_PROMPT,
    ]

    # All should be different
    for i, prompt1 in enumerate(domain_prompts):
        for j, prompt2 in enumerate(domain_prompts):
            if i != j:
                assert prompt1 != prompt2


def test_browse_prompt_mentions_key_concepts():
    """BROWSE_PROMPT should mention browsing-specific concepts."""
    assert "search" in BROWSE_PROMPT.lower()
    assert "product" in BROWSE_PROMPT.lower()
    assert "detail" in BROWSE_PROMPT.lower()
    assert "price" in BROWSE_PROMPT.lower()


def test_cart_prompt_mentions_key_concepts():
    """CART_PROMPT should mention cart-specific concepts."""
    assert "cart" in CART_PROMPT.lower()
    assert "add" in CART_PROMPT.lower()
    assert "quantity" in CART_PROMPT.lower() or "quantities" in CART_PROMPT.lower()
    assert "total" in CART_PROMPT.lower()


def test_order_prompt_mentions_key_concepts():
    """ORDER_PROMPT should mention order-specific concepts."""
    assert "order" in ORDER_PROMPT.lower()
    assert "history" in ORDER_PROMPT.lower()
    assert "status" in ORDER_PROMPT.lower()
    assert "delivery" in ORDER_PROMPT.lower()
    assert "get_my_profile" in ORDER_PROMPT
    assert "list_my_addresses" in ORDER_PROMPT
    assert "default address" in ORDER_PROMPT.lower()


def test_review_prompt_mentions_key_concepts():
    """REVIEW_PROMPT should mention review-specific concepts."""
    assert "review" in REVIEW_PROMPT.lower()
    assert "write" in REVIEW_PROMPT.lower() or "writing" in REVIEW_PROMPT.lower()
    assert "product" in REVIEW_PROMPT.lower()
    assert "own review history" in REVIEW_PROMPT.lower()
    assert "product ID" in REVIEW_PROMPT


def test_account_prompt_mentions_key_concepts():
    """ACCOUNT_PROMPT should mention account-specific concepts."""
    assert "account" in ACCOUNT_PROMPT.lower()
    assert "profile" in ACCOUNT_PROMPT.lower()
    assert "address" in ACCOUNT_PROMPT.lower()
    assert "privacy" in ACCOUNT_PROMPT.lower()


def test_chitchat_prompt_mentions_key_concepts():
    """CHITCHAT_PROMPT should mention conversation-specific concepts."""
    assert "friendly" in CHITCHAT_PROMPT.lower()
    assert (
        "conversation" in CHITCHAT_PROMPT.lower()
        or "conversational" in CHITCHAT_PROMPT.lower()
    )
    assert "ceramic" in CHITCHAT_PROMPT.lower()


def test_all_prompts_mention_company():
    """All prompts should mention CeramiCraft."""
    prompts = [
        SYSTEM_PROMPT,
        BROWSE_PROMPT,
        CART_PROMPT,
        ORDER_PROMPT,
        REVIEW_PROMPT,
        ACCOUNT_PROMPT,
        CHITCHAT_PROMPT,
    ]

    for prompt in prompts:
        assert "CeramiCraft" in prompt


def test_prompts_have_guidelines():
    """Domain prompts should have guidelines or focus areas."""
    domain_prompts = [
        BROWSE_PROMPT,
        CART_PROMPT,
        ORDER_PROMPT,
        REVIEW_PROMPT,
        ACCOUNT_PROMPT,
        CHITCHAT_PROMPT,
    ]

    for prompt in domain_prompts:
        # Should have either "Focus on:" or "Guidelines:"
        assert (
            "Focus on:" in prompt
            or "Guidelines:" in prompt
            or "focus on" in prompt.lower()
        )


def test_prompts_are_comprehensive():
    """Prompts should provide sufficient context for agents."""
    domain_prompts = [
        BROWSE_PROMPT,
        CART_PROMPT,
        ORDER_PROMPT,
        REVIEW_PROMPT,
        ACCOUNT_PROMPT,
        CHITCHAT_PROMPT,
    ]

    for prompt in domain_prompts:
        # Should be substantial (more than just a title)
        assert len(prompt) > 100
        assert len(prompt.split()) > 20


# ---------------------------------------------------------------------------
# Tests for get_prompt() and convenience functions
# ---------------------------------------------------------------------------


def _clear_prompt_cache():
    """Clear the module-level prompt cache between tests."""
    import ceramicraft_customer_support_agent.prompts as p

    p._prompt_cache.clear()


def _mock_settings_no_mlflow():
    cfg = MagicMock()
    cfg.MLFLOW_TRACKING_URI = ""
    return cfg


def _mock_settings_with_mlflow(uri: str = "http://localhost:5000"):
    cfg = MagicMock()
    cfg.MLFLOW_TRACKING_URI = uri
    return cfg


def test_get_prompt_fallback_no_mlflow_uri():
    """get_prompt should return fallback when MLFLOW_TRACKING_URI is not set."""
    _clear_prompt_cache()
    with patch(
        "ceramicraft_customer_support_agent.prompts.get_settings",
        return_value=_mock_settings_no_mlflow(),
    ):
        result = get_prompt("SYSTEM_PROMPT", "my fallback")
    assert result == "my fallback"


def test_get_prompt_fallback_on_mlflow_error():
    """get_prompt should return fallback when MLflow raises (e.g. prompt not found)."""
    _clear_prompt_cache()
    with (
        patch(
            "ceramicraft_customer_support_agent.prompts.get_settings",
            return_value=_mock_settings_with_mlflow(),
        ),
        patch(
            "ceramicraft_customer_support_agent.prompts.mlflow.MlflowClient",
            side_effect=Exception("prompt not found"),
        ),
    ):
        result = get_prompt("SYSTEM_PROMPT", "fallback text")
    assert result == "fallback text"


def test_get_prompt_cached():
    """get_prompt should use cache on second call."""
    _clear_prompt_cache()

    import ceramicraft_customer_support_agent.prompts as p

    # Pre-populate cache
    p._prompt_cache["MY_PROMPT"] = "cached value"
    result = get_prompt("MY_PROMPT", "should not be returned")
    assert result == "cached value"


def test_existing_constants_unchanged():
    """Existing module-level constants must remain unchanged."""
    assert SYSTEM_PROMPT is not None
    assert BROWSE_PROMPT is not None
    assert CART_PROMPT is not None
    assert ORDER_PROMPT is not None
    assert REVIEW_PROMPT is not None
    assert ACCOUNT_PROMPT is not None
    assert CHITCHAT_PROMPT is not None
    # Ensure they're still the same strings
    assert "CeramiCraft" in SYSTEM_PROMPT
    assert "CeramiCraft" in BROWSE_PROMPT


def test_convenience_functions_return_strings():
    """Convenience get_*_prompt() functions should return non-empty strings."""
    _clear_prompt_cache()
    with patch(
        "ceramicraft_customer_support_agent.prompts.get_settings",
        return_value=_mock_settings_no_mlflow(),
    ):
        assert isinstance(get_system_prompt(), str)
        assert isinstance(get_browse_prompt(), str)
        assert isinstance(get_cart_prompt(), str)
        assert isinstance(get_order_prompt(), str)
        assert isinstance(get_review_prompt(), str)
        assert isinstance(get_account_prompt(), str)
        assert isinstance(get_chitchat_prompt(), str)


def test_convenience_functions_return_fallbacks_without_mlflow():
    """Without MLflow URI, convenience functions should return hardcoded prompts."""
    _clear_prompt_cache()
    with patch(
        "ceramicraft_customer_support_agent.prompts.get_settings",
        return_value=_mock_settings_no_mlflow(),
    ):
        assert get_system_prompt() == SYSTEM_PROMPT
        _clear_prompt_cache()
        assert get_browse_prompt() == BROWSE_PROMPT
