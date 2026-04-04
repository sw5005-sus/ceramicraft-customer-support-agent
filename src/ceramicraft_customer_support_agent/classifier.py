"""Intent classification node for routing user queries."""

import logging
import re
from collections.abc import Callable
from enum import Enum

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.prompts import get_classifier_prompt

logger = logging.getLogger(__name__)

# Pattern to detect confirmation messages from the user.
_CONFIRM_PATTERN = re.compile(
    r"^(yes|y|ok|确认|确定|好的|好|是的|是|proceed|go ahead|confirm|confirmed|"
    r"sure|do it|没问题|可以|行|下单|我确认|请继续)\b",
    re.IGNORECASE,
)


class Intent(str, Enum):
    """User intent classifications."""

    BROWSE = "browse"
    CART = "cart"
    ORDER = "order"
    REVIEW = "review"
    ACCOUNT = "account"
    CHITCHAT = "chitchat"
    ESCALATE = "escalate"


class IntentClassification(BaseModel):
    """Intent classification result."""

    intent: Intent = Field(description="The classified intent")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the classification")


def build_classifier() -> Callable:
    """Build the intent classifier node.

    Returns:
        A callable that takes AgentState and returns updated state with intent.
    """
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,  # ty: ignore[unknown-argument]
        api_key=settings.OPENAI_API_KEY,  # ty: ignore[unknown-argument]
    ).with_structured_output(IntentClassification)

    async def classifier_node(state: dict) -> dict:
        """Classify the user's intent from their latest message."""
        messages = state.get("messages", [])
        last_intent = state.get("last_intent", "")
        needs_confirm = state.get("needs_confirm", False)

        # --- Confirmation shortcut ---
        # When the guard has requested confirmation and the user replies
        # with a confirmation phrase, skip classification entirely: mark
        # confirmed=True and re-route to the same intent so the subgraph
        # can retry the sensitive operation.
        if needs_confirm and last_intent:
            user_message = None
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    user_message = msg.content
                    break
            if user_message and _CONFIRM_PATTERN.search(user_message.strip()):
                logger.info(
                    "User confirmed sensitive operation, routing back to %s",
                    last_intent,
                )
                return {
                    "intent": last_intent,
                    "last_intent": last_intent,
                    "confirmed": True,
                    "needs_confirm": False,
                }

        if not messages:
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
                "confirmed": False,
                "needs_confirm": False,
            }

        # Get the latest user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break

        if not user_message:
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
                "confirmed": False,
                "needs_confirm": False,
            }

        try:
            # Format the prompt with the user message and conversation context
            prompt_template = get_classifier_prompt()
            prompt_content = prompt_template.format(
                message=user_message,
                last_intent=last_intent or "none",
            )
            result: IntentClassification = await llm.ainvoke(  # ty: ignore[invalid-assignment]
                [HumanMessage(content=prompt_content)]
            )

            logger.info(
                "Classified intent: %s (confidence: %.2f, last: %s) - %s",
                result.intent,
                result.confidence,
                last_intent or "none",
                result.reasoning,
            )

            return {
                "intent": result.intent.value,
                "last_intent": result.intent.value,
                "confirmed": False,
                "needs_confirm": False,
            }

        except ValidationError:
            logger.exception("Failed to parse classifier output")
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
                "confirmed": False,
                "needs_confirm": False,
            }
        except Exception:
            logger.exception("Intent classification failed")
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
                "confirmed": False,
                "needs_confirm": False,
            }

    return classifier_node
