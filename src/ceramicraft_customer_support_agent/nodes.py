"""Lightweight graph nodes (no tool usage)."""

import logging
from collections.abc import Callable

from langchain_openai import ChatOpenAI

from ceramicraft_customer_support_agent.config import get_settings

logger = logging.getLogger(__name__)


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

        # Map LangChain message types to OpenAI roles
        _role_map = {"human": "user", "ai": "assistant", "system": "system"}

        # Add system prompt as first message for context
        full_messages = [{"role": "system", "content": CHITCHAT_PROMPT}] + [
            {
                "role": _role_map.get(msg.type if hasattr(msg, "type") else "", "user"),
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
