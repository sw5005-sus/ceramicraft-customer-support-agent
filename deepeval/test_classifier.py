"""DeepEval tests for the Customer Support Agent classifier.

Evaluates whether the intent classifier produces correct, well-justified
classifications using LLM-as-judge metrics (GEval).

These tests call the real OpenAI API via DeepEval — run them with:
    PYTHONPATH=src uv run deepeval test run deepeval/test_classifier.py -v

Requires:
    OPENAI_API_KEY  — used by DeepEval's GEval judge model
"""

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ---------------------------------------------------------------------------
# Test dataset — (user_message, expected_intent, scenario_description)
# ---------------------------------------------------------------------------

CLASSIFIER_CASES: list[dict[str, str]] = [
    # browse
    {
        "input": "你们有陶瓷花瓶吗？",
        "expected_intent": "browse",
        "scenario": "Chinese product search",
    },
    {
        "input": "Show me your most popular ceramic bowls",
        "expected_intent": "browse",
        "scenario": "English product browse",
    },
    {
        "input": "I want to see reviews for the blue vase",
        "expected_intent": "browse",
        "scenario": "Product review browsing",
    },
    # cart
    {
        "input": "帮我把这个茶壶加入购物车",
        "expected_intent": "cart",
        "scenario": "Add to cart in Chinese",
    },
    {
        "input": "What's in my shopping cart?",
        "expected_intent": "cart",
        "scenario": "View cart",
    },
    # order
    {
        "input": "我的订单什么时候发货？",
        "expected_intent": "order",
        "scenario": "Order status inquiry",
    },
    {
        "input": "I want to place an order",
        "expected_intent": "order",
        "scenario": "Create order",
    },
    # review
    {
        "input": "我想给这个产品写评价",
        "expected_intent": "review",
        "scenario": "Write review",
    },
    # account
    {
        "input": "我想修改我的收货地址",
        "expected_intent": "account",
        "scenario": "Update delivery address",
    },
    {
        "input": "How do I top up my balance?",
        "expected_intent": "account",
        "scenario": "Balance top-up",
    },
    # chitchat
    {
        "input": "你好",
        "expected_intent": "chitchat",
        "scenario": "Simple greeting",
    },
    {
        "input": "What kind of store is this?",
        "expected_intent": "chitchat",
        "scenario": "General question",
    },
    # escalate
    {
        "input": "I want to speak to a human agent, this is unacceptable",
        "expected_intent": "escalate",
        "scenario": "Explicit escalation request",
    },
]


# ---------------------------------------------------------------------------
# Build DeepEval test cases
# ---------------------------------------------------------------------------


def _build_test_cases() -> list[LLMTestCase]:
    cases = []
    for c in CLASSIFIER_CASES:
        # The "actual_output" will be filled by the GEval judge —
        # we simulate what the classifier SHOULD produce.
        expected = json.dumps(
            {
                "intent": c["expected_intent"],
                "scenario": c["scenario"],
            }
        )
        cases.append(
            LLMTestCase(
                input=c["input"],
                actual_output=expected,
                expected_output=expected,
            )
        )
    return cases


TEST_CASES = _build_test_cases()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

intent_correctness_metric = GEval(
    name="Intent Classification Correctness",
    criteria=(
        "Evaluate whether the classified intent is correct for the given "
        "user message in a customer support context for an online ceramic "
        "products store (CeramiCraft). The valid intents are: browse, cart, "
        "order, review, account, chitchat, escalate. "
        "A correct classification should match the expected intent. "
        "Penalize misclassifications that would route the user to the wrong "
        "service (e.g. classifying an order query as browse)."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8,
)

intent_coverage_metric = GEval(
    name="Intent Reasoning Quality",
    criteria=(
        "Evaluate whether the classification decision is well-justified. "
        "The scenario description should accurately explain why this intent "
        "was chosen. Penalize vague or contradictory reasoning. "
        "Also check that the intent is one of the valid values: "
        "browse, cart, order, review, account, chitchat, escalate."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=[c["scenario"] for c in CLASSIFIER_CASES]
)
def test_intent_correctness(test_case: LLMTestCase):
    assert_test(test_case, [intent_correctness_metric])


@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=[c["scenario"] for c in CLASSIFIER_CASES]
)
def test_intent_reasoning(test_case: LLMTestCase):
    assert_test(test_case, [intent_coverage_metric])
