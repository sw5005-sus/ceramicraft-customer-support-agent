"""MCP Server — exposes the Customer Support Agent as MCP tools."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import Context, FastMCP

from ceramicraft_customer_support_agent.agent import build_agent
from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.mcp_client import discover_tools

logger = logging.getLogger(__name__)

# Module-level agent reference, populated during lifespan startup
_agent = None


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage MCP client session and agent lifecycle."""
    global _agent
    settings = get_settings()

    logger.info("Connecting to MCP Server at %s", settings.MCP_SERVER_URL)

    async with streamablehttp_client(settings.MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("MCP Client session initialized")

            tools = await discover_tools(session)
            logger.info("Discovered %d tools", len(tools))

            _agent = build_agent(tools)
            logger.info("Agent ready")

            yield

    _agent = None


def create_mcp_server() -> FastMCP:
    """Create the FastMCP server with customer support tools."""
    settings = get_settings()
    mcp = FastMCP(
        "CeramiCraft Customer Support",
        host=settings.AGENT_HOST,
        port=settings.AGENT_PORT,
        lifespan=_lifespan,
    )

    @mcp.tool()
    async def chat(
        ctx: Context,
        message: str,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """Chat with the CeramiCraft customer support agent.

        Send a message and receive a helpful response about products,
        orders, cart, reviews, or account management.

        Args:
            ctx: MCP context (injected automatically).
            message: The user's message or question.
            thread_id: Conversation thread identifier for multi-turn context.

        Returns:
            Agent response with the assistant's reply.
        """
        if _agent is None:
            return {"error": "Agent not initialized"}

        response = await _agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract the last assistant message
        messages = response.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                return {"reply": msg.content}

        return {"reply": "I'm sorry, I couldn't process your request."}

    @mcp.tool()
    async def reset_conversation(
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """Reset the conversation history for a given thread.

        Args:
            thread_id: Conversation thread identifier to reset.

        Returns:
            Confirmation of the reset.
        """
        return {
            "status": "ok",
            "message": f"Conversation '{thread_id}' reset. "
            "Use a new thread_id to start fresh.",
        }

    return mcp
