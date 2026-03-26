"""LangGraph ReAct agent for customer support."""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Shared checkpointer — persists conversation state across requests
_memory = MemorySaver()


def get_memory() -> MemorySaver:
    """Return the shared MemorySaver instance."""
    return _memory


def build_agent(tools: list[BaseTool]) -> Any:
    """Build a ReAct agent with the given tools.

    Each call creates a new agent bound to the provided tools,
    but shares the global MemorySaver for conversation continuity.

    Args:
        tools: LangChain-compatible tools (typically discovered from MCP Server).

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,  # ty: ignore[unknown-argument]
        api_key=settings.OPENAI_API_KEY,  # ty: ignore[unknown-argument]
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=_memory,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info(
        "Agent built with model=%s, tools=%d",
        settings.OPENAI_MODEL,
        len(tools),
    )
    return agent
