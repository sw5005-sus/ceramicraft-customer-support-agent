"""Safety guard node for post-processing responses."""

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Pre-compiled regex for detecting auth-related error messages.
# Uses word boundaries to avoid false positives on "author", "authority", etc.
_AUTH_ERROR_PATTERN = re.compile(
    r"\b(authentication|authorization|log[\s-]?in|unauthorized|unauthenticated)\b",
    re.IGNORECASE,
)


def build_guard() -> Callable:
    """Build the safety guard node.

    Checks for auth requirements and adds intervention messages when needed.

    Confirmation for sensitive operations (create_order, delete_address,
    confirm_receipt) is handled by the LLM itself via domain-specific
    prompt instructions, NOT by the guard.  This avoids the post-execution
    confirmation anti-pattern where tools have already run before the
    guard can intervene.

    Returns:
        A callable that takes AgentState and returns updated state.
    """

    def guard_node(state: dict) -> dict:
        """Apply safety checks and interventions."""
        auth_token = state.get("auth_token")
        messages = state.get("messages", [])

        # Check last messages for auth error patterns
        recent_messages = messages[-5:] if messages else []

        needs_auth_prompt = False

        for msg in recent_messages:
            msg_content = str(msg.content) if hasattr(msg, "content") else str(msg)

            if _AUTH_ERROR_PATTERN.search(msg_content):
                if not auth_token:
                    needs_auth_prompt = True

        updates: dict = {}

        if needs_auth_prompt and not auth_token:
            auth_message = (
                "To access your account information, cart, orders, or create reviews, "
                "you'll need to log in first. Please visit our website to sign in, "
                "then come back and I'll be happy to help!"
            )
            updates["messages"] = [{"role": "assistant", "content": auth_message}]
            logger.info("Added auth requirement message")

        return updates

    return guard_node
