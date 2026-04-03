"""Main LangGraph StateGraph for the Customer Support Agent."""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from ceramicraft_customer_support_agent.classifier import build_classifier
from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.guard import build_guard
from ceramicraft_customer_support_agent.nodes import (
    build_chitchat_node,
    build_escalate_node,
)
from ceramicraft_customer_support_agent.subgraphs import (
    build_account_subgraph,
    build_browse_subgraph,
    build_cart_subgraph,
    build_order_subgraph,
    build_review_subgraph,
)

logger = logging.getLogger(__name__)


async def build_checkpointer() -> Any:
    """Build a checkpointer: AsyncPostgresSaver if configured, MemorySaver otherwise.

    Uses AsyncPostgresSaver + AsyncConnectionPool so the checkpointer is
    compatible with the async agent invocation path (ainvoke / astream).
    """
    settings = get_settings()

    if settings.DATABASE_URL:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # psycopg_pool expects libpq DSN, not SQLAlchemy URL.
            conninfo = settings.DATABASE_URL.replace(
                "postgresql+psycopg://", "postgresql://", 1
            )

            async def _configure(conn: Any) -> None:
                conn.row_factory = dict_row

            pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
                conninfo=conninfo,
                max_size=10,
                kwargs={"autocommit": True},
                configure=_configure,
                open=False,
            )
            await pool.open(wait=True)
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            logger.info(
                "Using async PostgreSQL checkpointer: %s", settings.POSTGRES_HOST
            )
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Failed to init PostgreSQL checkpointer, falling back to MemorySaver: %s",
                exc,
            )

    logger.info("Using in-memory checkpointer (MemorySaver)")
    return MemorySaver()


class AgentState(TypedDict):
    """State for the customer support agent graph."""

    messages: Annotated[list, add_messages]
    intent: str  # classifier output
    auth_token: str | None
    needs_confirm: bool
    confirmed: bool


async def build_graph(tools: Sequence[BaseTool], checkpointer: Any = None) -> Any:
    """Build the main customer support agent graph.

    Args:
        tools: LangChain-compatible tools (discovered from MCP Server).
        checkpointer: Persistence backend for conversation history.
                      If None, a new checkpointer is built from config.

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    if checkpointer is None:
        checkpointer = await build_checkpointer()

    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]

    classifier = build_classifier()
    guard = build_guard()

    browse_agent = build_browse_subgraph(tools)
    cart_agent = build_cart_subgraph(tools)
    order_agent = build_order_subgraph(tools)
    review_agent = build_review_subgraph(tools)
    account_agent = build_account_subgraph(tools)
    chitchat_node = build_chitchat_node()
    escalate_node = build_escalate_node()

    graph.add_node("classifier", classifier)
    domain_subgraphs = [
        ("browse", browse_agent),
        ("cart", cart_agent),
        ("order", order_agent),
        ("review", review_agent),
        ("account", account_agent),
    ]
    for name, agent in domain_subgraphs:
        graph.add_node(name, _wrap_subgraph(agent, name))

    graph.add_node("chitchat", chitchat_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("guard", guard)

    graph.set_entry_point("classifier")

    domain_names = [name for name, _ in domain_subgraphs]
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {name: name for name in [*domain_names, "chitchat", "escalate"]},
    )

    for name in [*domain_names, "chitchat", "escalate"]:
        graph.add_edge(name, "guard")

    graph.set_finish_point("guard")

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Graph built with %d tools across %d domain subgraphs",
        len(tools),
        len(domain_subgraphs),
    )

    return compiled


def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "chitchat")
    logger.info("Routing to %s based on intent", intent)
    return intent


def _sanitize_messages(messages: list) -> list:
    """Remove orphaned tool_calls that lack a corresponding ToolMessage.

    The classifier uses ``with_structured_output`` which internally relies on
    function-calling.  This leaves AIMessages with ``tool_calls`` in the
    history but *no* matching ToolMessage, which LangGraph's
    ``_validate_chat_history`` rightfully rejects.

    This helper strips those orphaned AIMessages so downstream subgraphs
    receive a clean history.
    """
    answered_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            answered_ids.add(msg.tool_call_id)

    clean: list = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            if all(tc["id"] in answered_ids for tc in msg.tool_calls):
                clean.append(msg)
        else:
            clean.append(msg)
    return clean


def _trim_messages(messages: list, max_history: int) -> list:
    """Keep only the most recent max_history messages."""
    if max_history <= 0 or len(messages) <= max_history:
        return messages
    return messages[-max_history:]


def _wrap_subgraph(subgraph: Any, domain: str) -> Callable:
    """Wrap a domain subgraph to work within the main graph state.

    Args:
        subgraph: A compiled LangGraph agent from create_react_agent.
        domain: Domain name for logging.

    Returns:
        An async callable that sanitizes messages, invokes the subgraph,
        and returns only the newly produced messages.
    """

    async def subgraph_node(state: AgentState) -> dict:
        """Invoke a domain subgraph with the current state."""
        try:
            messages = _sanitize_messages(state.get("messages", []))
            messages = _trim_messages(messages, get_settings().AGENT_MAX_HISTORY)

            result = await subgraph.ainvoke({"messages": messages})

            new_messages = result.get("messages", [])
            if len(new_messages) > len(messages):
                new_messages = new_messages[len(messages) :]

            return {"messages": new_messages}

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
