"""MCP Client — tool discovery from the CeramiCraft MCP Server."""

import logging

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession

logger = logging.getLogger(__name__)


async def discover_tools(session: ClientSession) -> list:
    """Discover tools from the MCP Server and convert to LangChain format.

    Args:
        session: An initialized MCP ClientSession.

    Returns:
        List of LangChain-compatible tools.
    """
    tools = await load_mcp_tools(session)
    logger.info("Discovered %d tools from MCP Server", len(tools))
    return tools
