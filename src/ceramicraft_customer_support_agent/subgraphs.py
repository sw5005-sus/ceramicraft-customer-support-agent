"""Domain-specific subgraph agents."""

from collections.abc import Callable, Sequence
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # ty: ignore[deprecated]

from ceramicraft_customer_support_agent.config import get_settings

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
    prompt: str,
    checkpointer: MemorySaver,
    domain: str,
) -> Any:
    """Build a domain-specific ReAct subgraph.

    Args:
        all_tools: Complete list of discovered tools.
        tool_names: Set of tool names for this domain.
        prompt: Domain-specific system prompt.
        checkpointer: Shared memory for conversation continuity.
        domain: Domain name for logging.

    Returns:
        Compiled LangGraph agent for the domain.
    """
    tools = _filter_tools(all_tools, tool_names)
    agent = create_react_agent(  # ty: ignore[deprecated]
        model=_get_llm(),
        tools=tools,
        checkpointer=checkpointer,
        prompt=prompt,
    )
    logger.info("Built %s subgraph with %d tools", domain, len(tools))
    return agent


def build_browse_subgraph(
    all_tools: Sequence[BaseTool], checkpointer: MemorySaver
) -> Any:
    """Build the browse domain subgraph.

    Handles product search, viewing details, and reading reviews.
    """
    from ceramicraft_customer_support_agent.prompts import BROWSE_PROMPT

    return _build_domain_subgraph(
        all_tools, BROWSE_TOOLS, BROWSE_PROMPT, checkpointer, "browse"
    )


def build_cart_subgraph(
    all_tools: Sequence[BaseTool], checkpointer: MemorySaver
) -> Any:
    """Build the cart domain subgraph.

    Handles shopping cart operations and product search.
    """
    from ceramicraft_customer_support_agent.prompts import CART_PROMPT

    return _build_domain_subgraph(
        all_tools, CART_TOOLS, CART_PROMPT, checkpointer, "cart"
    )


def build_order_subgraph(
    all_tools: Sequence[BaseTool], checkpointer: MemorySaver
) -> Any:
    """Build the order domain subgraph.

    Handles order management and history.
    """
    from ceramicraft_customer_support_agent.prompts import ORDER_PROMPT

    return _build_domain_subgraph(
        all_tools, ORDER_TOOLS, ORDER_PROMPT, checkpointer, "order"
    )


def build_review_subgraph(
    all_tools: Sequence[BaseTool], checkpointer: MemorySaver
) -> Any:
    """Build the review domain subgraph.

    Handles review creation and management.
    """
    from ceramicraft_customer_support_agent.prompts import REVIEW_PROMPT

    return _build_domain_subgraph(
        all_tools, REVIEW_TOOLS, REVIEW_PROMPT, checkpointer, "review"
    )


def build_account_subgraph(
    all_tools: Sequence[BaseTool], checkpointer: MemorySaver
) -> Any:
    """Build the account domain subgraph.

    Handles profile and address management.
    """
    from ceramicraft_customer_support_agent.prompts import ACCOUNT_PROMPT

    return _build_domain_subgraph(
        all_tools, ACCOUNT_TOOLS, ACCOUNT_PROMPT, checkpointer, "account"
    )


def build_chitchat_node() -> Callable:
    """Build the chitchat node for general conversation.

    Returns pure LLM responses without tools.

    Returns:
        A callable that takes AgentState and returns response.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,  # ty: ignore[unknown-argument]
        api_key=settings.OPENAI_API_KEY,  # ty: ignore[unknown-argument]
    )

    def chitchat_node(state: dict) -> dict:
        """Handle general conversation without tools."""
        from ceramicraft_customer_support_agent.prompts import CHITCHAT_PROMPT

        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [
                    {"role": "assistant", "content": "Hello! How can I help you today?"}
                ]
            }

        # Add system prompt as first message for context
        full_messages = [{"role": "system", "content": CHITCHAT_PROMPT}] + [
            {
                "role": msg.type if hasattr(msg, "type") else "user",
                "content": str(msg.content) if hasattr(msg, "content") else str(msg),
            }
            for msg in messages
        ]

        try:
            response = llm.invoke(full_messages)
            return {"messages": [{"role": "assistant", "content": response.content}]}
        except Exception:
            logger.exception("Chitchat node failed")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I apologize, but I'm having trouble right now. Could you please try again?",
                    }
                ]
            }

    return chitchat_node


def build_escalate_node() -> Callable:
    """Build the escalate node for complex issues.

    Returns:
        A callable that returns a standard escalation message.
    """

    def escalate_node(state: dict) -> dict:  # noqa: ARG001
        """Return standard escalation message."""
        escalation_message = (
            "I understand you need additional help with this issue. "
            "Let me connect you with one of our human support representatives "
            "who can provide more specialized assistance. Please hold on while "
            "I transfer your inquiry."
        )

        return {"messages": [{"role": "assistant", "content": escalation_message}]}

    return escalate_node
