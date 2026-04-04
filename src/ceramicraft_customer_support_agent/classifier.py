"""Intent classification node for routing user queries."""

import logging
from collections.abc import Callable
from enum import Enum

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.prompts import get_classifier_prompt

logger = logging.getLogger(__name__)


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
        if not messages:
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
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
            }

        except ValidationError:
            logger.exception("Failed to parse classifier output")
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
            }
        except Exception:
            logger.exception("Intent classification failed")
            return {
                "intent": Intent.CHITCHAT.value,
                "last_intent": Intent.CHITCHAT.value,
            }

    return classifier_node
