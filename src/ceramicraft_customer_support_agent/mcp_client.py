"""MCP Client — tool discovery and per-request session management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ceramicraft_customer_support_agent.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_session(
    token: str | None = None,
) -> AsyncIterator[ClientSession]:
    """Create an MCP client session with optional auth token.

    Usage::

        async with mcp_session(token="eyJ...") as session:
            tools = await discover_tools(session)

    Args:
        token: Optional Bearer token to forward to the MCP Server.

    Yields:
        An initialized MCP ClientSession.
    """
    settings = get_settings()
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with streamablehttp_client(settings.MCP_SERVER_URL, headers=headers) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


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
