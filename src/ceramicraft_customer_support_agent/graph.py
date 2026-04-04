"""Main LangGraph StateGraph for the Customer Support Agent."""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
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


async def build_checkpointer() -> AsyncPostgresSaver:
    """Build an AsyncPostgresSaver backed by an async connection pool.

    Uses AsyncPostgresSaver + AsyncConnectionPool for compatibility with
    the async agent invocation path (ainvoke / astream).

    Raises:
        RuntimeError: If DATABASE_URL is not configured.
        Exception: Any connection or setup error is propagated to the caller.
    """
    settings = get_settings()

    if not settings.DATABASE_URL:
        raise RuntimeError(
            "PostgreSQL is not configured. Set POSTGRES_HOST, POSTGRES_USER, "
            "POSTGRES_PASSWORD, POSTGRES_PORT, and CS_AGENT_DB_NAME."
        )

    # psycopg_pool expects a libpq DSN, not a SQLAlchemy URL.
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
    logger.info("PostgreSQL checkpointer ready: %s", settings.POSTGRES_HOST)
    return checkpointer


class AgentState(TypedDict):
    """State for the customer support agent graph."""

    messages: Annotated[list, add_messages]
    intent: str  # classifier output
    last_intent: str  # previous turn's intent for continuity
    auth_token: str | None


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
        messages = _sanitize_messages(state.get("messages", []))
        messages = _trim_messages(messages, get_settings().AGENT_MAX_HISTORY)

        # Inject auth context so the LLM knows whether to attempt
        # authenticated operations or ask the user to log in.
        auth_token = state.get("auth_token")
        if auth_token:
            auth_hint = SystemMessage(
                content=(
                    "The user is authenticated. You may call tools that "
                    "require login on their behalf without asking them to "
                    "log in."
                )
            )
            messages = [auth_hint, *messages]
        try:
            result = await subgraph.ainvoke({"messages": messages})
        except Exception:
            logger.exception("Subgraph '%s' failed", domain)
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "Sorry, something went wrong while processing your "
                            "request. Please try again or rephrase your question."
                        ),
                    }
                ]
            }

        new_messages = result.get("messages", [])
        # Strip the injected auth hint when counting original messages.
        orig_count = len(messages) - (1 if auth_token else 0)
        # Always slice off the echoed input messages to avoid duplication.
        # Subgraphs typically echo back all input messages + new responses.
        if len(new_messages) >= orig_count:
            new_messages = new_messages[orig_count:]

        # Fallback: if subgraph returned no new messages, provide a generic reply
        if not new_messages:
            new_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "I processed your request but couldn't generate a "
                        "response. Could you please rephrase or try again?"
                    ),
                }
            ]

        return {"messages": new_messages}

    return subgraph_node
