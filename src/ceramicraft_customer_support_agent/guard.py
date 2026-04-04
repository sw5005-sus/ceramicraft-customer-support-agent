"""Safety guard node for post-processing responses."""

import logging
import re
from collections.abc import Callable

from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger(__name__)

# Operations that require confirmation
SENSITIVE_OPERATIONS = {"delete_address", "confirm_receipt", "create_order"}

# Pre-compiled regex for detecting auth-related error messages.
# Uses word boundaries to avoid false positives on "author", "authority", etc.
_AUTH_ERROR_PATTERN = re.compile(
    r"\b(authentication|authorization|log[\s-]?in|unauthorized|unauthenticated)\b",
    re.IGNORECASE,
)


def build_guard() -> Callable:
    """Build the safety guard node.

    Checks for auth requirements and sensitive operations,
    adding intervention messages when needed.

    Returns:
        A callable that takes AgentState and returns updated state.
    """

    def guard_node(state: dict) -> dict:
        """Apply safety checks and interventions."""
        auth_token = state.get("auth_token")
        confirmed = state.get("confirmed", False)
        messages = state.get("messages", [])

        # Check last messages for tool usage patterns
        recent_messages = messages[-5:] if messages else []

        # Look for auth errors or tool calls requiring auth
        needs_auth_prompt = False
        needs_confirmation_prompt = False
        sensitive_op_detected = None

        for msg in recent_messages:
            msg_content = str(msg.content) if hasattr(msg, "content") else str(msg)

            # Check for auth-related errors in responses
            if _AUTH_ERROR_PATTERN.search(msg_content):
                if not auth_token:
                    needs_auth_prompt = True

            # Check for sensitive operations via ToolMessage.name (precise)
            # or AIMessage.tool_calls (requested but maybe not yet executed)
            if isinstance(msg, ToolMessage) and msg.name in SENSITIVE_OPERATIONS:
                sensitive_op_detected = msg.name
                if not confirmed:
                    needs_confirmation_prompt = True
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc.get("name") in SENSITIVE_OPERATIONS:
                        sensitive_op_detected = tc["name"]
                        if not confirmed:
                            needs_confirmation_prompt = True
                        break

        new_messages = []
        updates = {}

        # Handle auth requirement
        if needs_auth_prompt and not auth_token:
            auth_message = (
                "To access your account information, cart, orders, or create reviews, "
                "you'll need to log in first. Please visit our website to sign in, "
                "then come back and I'll be happy to help!"
            )
            new_messages.append({"role": "assistant", "content": auth_message})
            logger.info("Added auth requirement message")

        # Handle confirmation requirement
        if needs_confirmation_prompt and sensitive_op_detected and not confirmed:
            confirm_messages = {
                "delete_address": "Are you sure you want to delete this address? This action cannot be undone.",
                "confirm_receipt": "Please confirm that you have received your order and are satisfied with it.",
                "create_order": "I'm about to place an order for you. Please confirm you'd like to proceed.",
            }

            confirm_message = confirm_messages.get(
                sensitive_op_detected,
                f"This action requires confirmation. Are you sure you want to proceed with {sensitive_op_detected}?",
            )

            new_messages.append({"role": "assistant", "content": confirm_message})
            updates["needs_confirm"] = True
            logger.info("Added confirmation requirement for %s", sensitive_op_detected)

        if new_messages:
            updates["messages"] = new_messages

        return updates

    return guard_node
