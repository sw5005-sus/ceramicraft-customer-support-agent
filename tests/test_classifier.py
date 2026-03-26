"""Tests for the classifier module."""

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

import pytest
from pydantic import ValidationError

from ceramicraft_customer_support_agent.classifier import (
    Intent,
    IntentClassification,
    build_classifier,
    CLASSIFIER_PROMPT,
)


def test_intent_enum_values():
    """Intent enum should have all expected values."""
    expected_intents = {
        "browse",
        "cart",
        "order",
        "review",
        "account",
        "chitchat",
        "escalate",
    }

    actual_intents = {intent.value for intent in Intent}

    assert actual_intents == expected_intents


def test_intent_classification_model():
    """IntentClassification should validate properly."""
    classification = IntentClassification(
        intent=Intent.BROWSE,
        confidence=0.85,
        reasoning="User is searching for products",
    )

    assert classification.intent == Intent.BROWSE
    assert classification.confidence == 0.85
    assert classification.reasoning == "User is searching for products"


def test_intent_classification_validation():
    """IntentClassification should validate confidence bounds."""
    # Valid confidence
    IntentClassification(intent=Intent.CART, confidence=0.0, reasoning="Test")

    IntentClassification(intent=Intent.CART, confidence=1.0, reasoning="Test")

    # Invalid confidence should raise ValidationError
    with pytest.raises(ValidationError):
        IntentClassification(intent=Intent.CART, confidence=-0.1, reasoning="Test")

    with pytest.raises(ValidationError):
        IntentClassification(intent=Intent.CART, confidence=1.1, reasoning="Test")


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
def test_build_classifier_creates_llm(mock_llm_cls):
    """build_classifier should create LLM with structured output."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    classifier = build_classifier()

    mock_llm_cls.assert_called_once()
    mock_llm_instance.with_structured_output.assert_called_once_with(
        IntentClassification
    )
    assert callable(classifier)


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
def test_classifier_node_with_valid_message(mock_llm_cls):
    """Classifier node should process valid message and return intent."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    # Mock classification result
    mock_result = IntentClassification(
        intent=Intent.BROWSE,
        confidence=0.9,
        reasoning="User wants to search for products",
    )
    mock_structured_llm.invoke.return_value = mock_result

    classifier = build_classifier()

    # Mock message with human type
    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "I want to find ceramic bowls"

    state = {"messages": [mock_message]}

    result = classifier(state)

    assert result == {"intent": "browse"}
    mock_structured_llm.invoke.assert_called_once()

    # Check that the prompt was formatted correctly
    call_args = mock_structured_llm.invoke.call_args[0][0]
    assert len(call_args) == 1
    assert isinstance(call_args[0], HumanMessage)
    assert "I want to find ceramic bowls" in call_args[0].content


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
def test_classifier_node_with_empty_messages(mock_llm_cls):
    """Classifier node should default to chitchat with empty messages."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    classifier = build_classifier()

    state = {"messages": []}

    result = classifier(state)

    assert result == {"intent": "chitchat"}
    mock_structured_llm.invoke.assert_not_called()


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
def test_classifier_node_with_no_human_messages(mock_llm_cls):
    """Classifier node should default to chitchat with no human messages."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    classifier = build_classifier()

    # Mock AI message
    mock_message = MagicMock()
    mock_message.type = "ai"
    mock_message.content = "How can I help you?"

    state = {"messages": [mock_message]}

    result = classifier(state)

    assert result == {"intent": "chitchat"}
    mock_structured_llm.invoke.assert_not_called()


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.classifier.logger")
def test_classifier_node_with_validation_error(mock_logger, mock_llm_cls):
    """Classifier node should handle validation errors gracefully."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    # Mock validation error
    mock_structured_llm.invoke.side_effect = ValidationError.from_exception_data(
        "test", []
    )

    classifier = build_classifier()

    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "test message"

    state = {"messages": [mock_message]}

    result = classifier(state)

    assert result == {"intent": "chitchat"}
    mock_logger.exception.assert_called_once_with("Failed to parse classifier output")


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
@patch("ceramicraft_customer_support_agent.classifier.logger")
def test_classifier_node_with_generic_exception(mock_logger, mock_llm_cls):
    """Classifier node should handle generic exceptions gracefully."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    # Mock generic exception
    mock_structured_llm.invoke.side_effect = Exception("API error")

    classifier = build_classifier()

    mock_message = MagicMock()
    mock_message.type = "human"
    mock_message.content = "test message"

    state = {"messages": [mock_message]}

    result = classifier(state)

    assert result == {"intent": "chitchat"}
    mock_logger.exception.assert_called_once_with("Intent classification failed")


@patch("ceramicraft_customer_support_agent.classifier.ChatOpenAI")
def test_classifier_node_finds_latest_human_message(mock_llm_cls):
    """Classifier should process the most recent human message."""
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    mock_result = IntentClassification(
        intent=Intent.CART, confidence=0.8, reasoning="User wants to add to cart"
    )
    mock_structured_llm.invoke.return_value = mock_result

    classifier = build_classifier()

    # Create messages with latest human message
    messages = []

    old_human = MagicMock()
    old_human.type = "human"
    old_human.content = "old message"
    messages.append(old_human)

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "ai response"
    messages.append(ai_msg)

    latest_human = MagicMock()
    latest_human.type = "human"
    latest_human.content = "add product to my cart"
    messages.append(latest_human)

    state = {"messages": messages}

    result = classifier(state)

    assert result == {"intent": "cart"}

    # Verify the latest message was used
    call_args = mock_structured_llm.invoke.call_args[0][0]
    assert "add product to my cart" in call_args[0].content
    assert "old message" not in call_args[0].content


def test_classifier_prompt_contains_intents():
    """Classifier prompt should mention all intent options."""
    for intent in Intent:
        assert intent.value in CLASSIFIER_PROMPT

    # Check for key instruction words
    assert "JSON" in CLASSIFIER_PROMPT
    assert "intent" in CLASSIFIER_PROMPT
    assert "confidence" in CLASSIFIER_PROMPT
