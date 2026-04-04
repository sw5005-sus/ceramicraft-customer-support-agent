"""Domain-specific subgraph agents."""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # ty: ignore[deprecated]

from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.prompts import (
    get_account_prompt,
    get_browse_prompt,
    get_cart_prompt,
    get_order_prompt,
    get_review_prompt,
    get_system_prompt,
)

logger = logging.getLogger(__name__)

# Tool name sets for each domain
BROWSE_TOOLS = {"search_products", "get_product", "list_product_reviews"}
CART_TOOLS = {
    "get_cart",
    "add_to_cart",
    "update_cart_item",
    "remove_cart_item",
    "estimate_cart_price",
    "search_products",
}
ORDER_TOOLS = {
    "list_my_orders",
    "get_order_detail",
    "confirm_receipt",
    "get_order_stats",
    "create_order",
}
REVIEW_TOOLS = {
    "create_review",
    "like_review",
    "get_user_reviews",
    "list_product_reviews",
}
ACCOUNT_TOOLS = {
    "get_my_profile",
    "update_my_profile",
    "list_my_addresses",
    "create_address",
    "update_address",
    "delete_address",
}


def _get_llm() -> ChatOpenAI:
    """Create a shared LLM instance from settings."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,  # ty: ignore[unknown-argument]
        api_key=settings.OPENAI_API_KEY,  # ty: ignore[unknown-argument]
    )


def _filter_tools(all_tools: Sequence[BaseTool], names: set[str]) -> list[BaseTool]:
    """Filter tools by name set."""
    return [tool for tool in all_tools if tool.name in names]


def _build_domain_subgraph(
    all_tools: Sequence[BaseTool],
    tool_names: set[str],
    domain_prompt: str,
    domain: str,
) -> Any:
    """Build a domain-specific ReAct subgraph.

    The effective system prompt is the base system prompt (cross-domain rules)
    followed by the domain-specific instructions, separated by a horizontal
    rule.  This ensures cross-cutting rules (language, privacy, tone) are
    always active regardless of which domain handles the request.

    Args:
        all_tools: Complete list of discovered tools.
        tool_names: Set of tool names for this domain.
        domain_prompt: Domain-specific system prompt.
        domain: Domain name for logging.

    Returns:
        Compiled LangGraph agent for the domain.
    """
    tools = _filter_tools(all_tools, tool_names)
    combined_prompt = f"{get_system_prompt()}\n\n---\n\n{domain_prompt}"
    # Subgraphs are stateless per-invocation; the *main* graph owns the
    # checkpointer for multi-turn persistence.  Giving subgraphs their own
    # checkpointer causes cross-request state leaks (tool_call messages from
    # one invocation polluting the next).
    agent = create_react_agent(  # ty: ignore[deprecated]
        model=_get_llm(),
        tools=tools,
        prompt=combined_prompt,
    )
    logger.info("Built %s subgraph with %d tools", domain, len(tools))
    return agent


def build_browse_subgraph(all_tools: Sequence[BaseTool]) -> Any:
    """Build the browse domain subgraph.

    Handles product search, viewing details, and reading reviews.
    """
    return _build_domain_subgraph(
        all_tools, BROWSE_TOOLS, get_browse_prompt(), "browse"
    )


def build_cart_subgraph(all_tools: Sequence[BaseTool]) -> Any:
    """Build the cart domain subgraph.

    Handles shopping cart operations and product search.
    """
    return _build_domain_subgraph(all_tools, CART_TOOLS, get_cart_prompt(), "cart")


def build_order_subgraph(all_tools: Sequence[BaseTool]) -> Any:
    """Build the order domain subgraph.

    Handles order management and history.
    """
    return _build_domain_subgraph(all_tools, ORDER_TOOLS, get_order_prompt(), "order")


def build_review_subgraph(all_tools: Sequence[BaseTool]) -> Any:
    """Build the review domain subgraph.

    Handles review creation and management.
    """
    return _build_domain_subgraph(
        all_tools, REVIEW_TOOLS, get_review_prompt(), "review"
    )


def build_account_subgraph(all_tools: Sequence[BaseTool]) -> Any:
    """Build the account domain subgraph.

    Handles profile and address management.
    """
    return _build_domain_subgraph(
        all_tools, ACCOUNT_TOOLS, get_account_prompt(), "account"
    )
