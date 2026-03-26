"""LangGraph Customer Support Agent - backward compatibility interface."""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool

from ceramicraft_customer_support_agent.graph import build_graph

logger = logging.getLogger(__name__)


def build_agent(tools: Sequence[BaseTool]) -> Any:
    """Build a customer support agent with the given tools.

    This function provides backward compatibility with the old agent interface
    while using the new LangGraph StateGraph implementation underneath.

    Args:
        tools: LangChain-compatible tools (typically discovered from MCP Server).

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    agent = build_graph(tools)

    logger.info(
        "Agent built with %d tools using LangGraph StateGraph",
        len(tools),
    )
    return agent
