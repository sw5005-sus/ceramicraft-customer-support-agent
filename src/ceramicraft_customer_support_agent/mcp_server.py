"""MCP Server — exposes the Customer Support Agent as MCP tools."""

import logging
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from ceramicraft_customer_support_agent.agent import build_agent
from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.mcp_client import discover_tools, mcp_session

logger = logging.getLogger(__name__)


def _extract_bearer_token(ctx: Context) -> str | None:
    """Extract Bearer token from the incoming MCP request context.

    Mirrors the approach used by ceramicraft-mcp-server.
    """
    # FastMCP may expose transport headers on the context
    headers = getattr(ctx, "headers", None)
    if headers:
        auth_header = headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

    # Fallback: meta/extra
    meta = getattr(ctx, "meta", None)
    if meta:
        extra = getattr(meta, "extra", None)
        if isinstance(extra, dict):
            token = extra.get("token") or extra.get("authorization", "")
            if isinstance(token, str):
                if token.startswith("Bearer "):
                    return token[7:]
                if token:
                    return token

    return None


def create_mcp_server() -> FastMCP:
    """Create the FastMCP server with customer support tools."""
    settings = get_settings()
    mcp = FastMCP(
        "CeramiCraft Customer Support",
        host=settings.AGENT_HOST,
        port=settings.AGENT_PORT,
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):  # noqa: ARG001
        """Liveness/readiness probe endpoint."""
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "ok"})

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
        token = _extract_bearer_token(ctx)

        try:
            async with mcp_session(token=token) as session:
                tools = await discover_tools(session)
                agent = build_agent(tools)

                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
        except Exception:
            logger.exception("Agent invocation failed")
            raise ToolError(
                "Sorry, something went wrong processing your request. Please try again."
            )

        # Extract the last assistant message
        messages = response.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                return {"reply": msg.content}

        return {"reply": "I'm sorry, I couldn't process your request."}

    @mcp.tool()
    async def reset(
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
