"""Tests for the prompts module."""

from ceramicraft_customer_support_agent.prompts import (
    SYSTEM_PROMPT,
    BROWSE_PROMPT,
    CART_PROMPT,
    ORDER_PROMPT,
    REVIEW_PROMPT,
    ACCOUNT_PROMPT,
    CHITCHAT_PROMPT,
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


def test_review_prompt_mentions_key_concepts():
    """REVIEW_PROMPT should mention review-specific concepts."""
    assert "review" in REVIEW_PROMPT.lower()
    assert "write" in REVIEW_PROMPT.lower() or "writing" in REVIEW_PROMPT.lower()
    assert "product" in REVIEW_PROMPT.lower()


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
