"""Main LangGraph StateGraph for the Customer Support Agent."""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from ceramicraft_customer_support_agent.classifier import build_classifier
from ceramicraft_customer_support_agent.guard import build_guard
from ceramicraft_customer_support_agent.subgraphs import (
    build_account_subgraph,
    build_browse_subgraph,
    build_cart_subgraph,
    build_chitchat_node,
    build_escalate_node,
    build_order_subgraph,
    build_review_subgraph,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the customer support agent graph."""

    messages: Annotated[list, add_messages]
    intent: str  # classifier output
    auth_token: str | None
    needs_confirm: bool
    confirmed: bool


def build_graph(tools: Sequence[BaseTool]) -> Any:
    """Build the main customer support agent graph.

    Creates a StateGraph with intent classification, domain routing,
    and safety guards.

    Args:
        tools: LangChain-compatible tools (discovered from MCP Server).

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    # Shared checkpointer for conversation continuity
    memory = MemorySaver()

    # Create the main graph
    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]

    # Build nodes
    classifier = build_classifier()
    guard = build_guard()

    # Build domain subgraphs
    browse_agent = build_browse_subgraph(tools, memory)
    cart_agent = build_cart_subgraph(tools, memory)
    order_agent = build_order_subgraph(tools, memory)
    review_agent = build_review_subgraph(tools, memory)
    account_agent = build_account_subgraph(tools, memory)
    chitchat_node = build_chitchat_node()
    escalate_node = build_escalate_node()

    # Add nodes to graph
    graph.add_node("classifier", classifier)
    graph.add_node("browse", _wrap_subgraph(browse_agent))
    graph.add_node("cart", _wrap_subgraph(cart_agent))
    graph.add_node("order", _wrap_subgraph(order_agent))
    graph.add_node("review", _wrap_subgraph(review_agent))
    graph.add_node("account", _wrap_subgraph(account_agent))
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("guard", guard)

    # Set entry point
    graph.set_entry_point("classifier")

    # Add routing edges from classifier
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "browse": "browse",
            "cart": "cart",
            "order": "order",
            "review": "review",
            "account": "account",
            "chitchat": "chitchat",
            "escalate": "escalate",
        },
    )

    # All domain nodes go to guard
    for domain in [
        "browse",
        "cart",
        "order",
        "review",
        "account",
        "chitchat",
        "escalate",
    ]:
        graph.add_edge(domain, "guard")

    # Guard is the end
    graph.set_finish_point("guard")

    # Compile with checkpointer
    compiled = graph.compile(checkpointer=memory)

    logger.info(
        "Graph built with %d tools across %d domain subgraphs",
        len(tools),
        5,  # browse, cart, order, review, account
    )

    return compiled


def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent.

    Args:
        state: Current agent state with intent classification.

    Returns:
        Node name to route to.
    """
    intent = state.get("intent", "chitchat")
    logger.info("Routing to %s based on intent", intent)
    return intent


def _wrap_subgraph(subgraph: Any) -> Callable:
    """Wrap a subgraph agent to work with the main graph state.

    Args:
        subgraph: A compiled LangGraph agent from create_react_agent.

    Returns:
        An async callable that extracts messages, invokes the subgraph,
        and returns updates.
    """

    async def subgraph_node(state: AgentState) -> dict:
        """Invoke a domain subgraph with the current state."""
        try:
            messages = state.get("messages", [])

            result = await subgraph.ainvoke(
                {"messages": messages},
                config={"configurable": {"thread_id": "subgraph"}},
            )

            return {"messages": result.get("messages", [])}

        except Exception:
            logger.exception("Subgraph invocation failed")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your request. Please try again.",
                    }
                ]
            }

    return subgraph_node
