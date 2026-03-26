"""Intent classification node for routing user queries."""

import logging
from collections.abc import Callable
from enum import Enum

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from ceramicraft_customer_support_agent.config import get_settings

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


CLASSIFIER_PROMPT = """\
You are an intent classifier for a customer support system for CeramiCraft, an online ceramic products store.

Classify the user's message into one of these intents:

- **browse**: Looking for products, searching, viewing product details or reviews
- **cart**: Managing shopping cart (view, add, remove, update items, pricing)
- **order**: Order management (list orders, view details, confirm receipt, order status)
- **review**: Writing or managing reviews (create, like, view personal reviews)
- **account**: Profile or address management (view/update profile, manage addresses)
- **chitchat**: General conversation, greetings, small talk
- **escalate**: Complex issues, complaints, requests for human support

Respond with a JSON object containing the intent, confidence score, and brief reasoning.

User message: {message}
"""


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

    def classifier_node(state: dict) -> dict:
        """Classify the user's intent from their latest message."""
        messages = state.get("messages", [])
        if not messages:
            return {"intent": Intent.CHITCHAT.value}

        # Get the latest user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break

        if not user_message:
            return {"intent": Intent.CHITCHAT.value}

        try:
            # Format the prompt with the user message
            prompt_content = CLASSIFIER_PROMPT.format(message=user_message)
            result: IntentClassification = llm.invoke(  # ty: ignore[invalid-assignment]
                [HumanMessage(content=prompt_content)]
            )

            logger.info(
                "Classified intent: %s (confidence: %.2f) - %s",
                result.intent,
                result.confidence,
                result.reasoning,
            )

            return {"intent": result.intent.value}

        except ValidationError:
            logger.exception("Failed to parse classifier output")
            return {"intent": Intent.CHITCHAT.value}
        except Exception:
            logger.exception("Intent classification failed")
            return {"intent": Intent.CHITCHAT.value}

    return classifier_node
