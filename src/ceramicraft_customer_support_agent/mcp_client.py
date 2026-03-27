"""MCP Client — tool discovery + per-request auth injection."""

import asyncio
import contextvars
import logging
from typing import Any

from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    MCPToolCallResult,
)
from langchain_mcp_adapters.tools import load_mcp_tools

from ceramicraft_customer_support_agent.config import get_settings

logger = logging.getLogger(__name__)

# ── Per-request auth token (set before each agent invocation) ──
_current_auth_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_auth_token", default=None
)


def set_auth_token(token: str | None) -> None:
    """Set the auth token for the current request context."""
    _current_auth_token.set(token)


class _AuthInterceptor:
    """Injects the per-request Bearer token into MCP tool call headers.

    Reads from ``_current_auth_token`` context-var so each concurrent
    request carries its own user token.
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Any,
    ) -> MCPToolCallResult:
        token = _current_auth_token.get()
        if token:
            headers = dict(request.headers or {})
            headers["authorization"] = f"Bearer {token}"
            request = request.override(headers=headers)
        return await handler(request)


class PersistentMCPClient:
    """Discovers and caches MCP tools using connection mode.

    ``langchain-mcp-adapters`` only injects interceptor headers when
    ``session=None`` and a ``connection`` config is provided.  Each tool
    call creates a short-lived MCP session via the connection, which
    lets ``_AuthInterceptor`` inject per-request Bearer tokens into the
    HTTP headers.

    Tool *discovery* also goes through the connection (one-shot session
    to list tools), but the tool list is cached — subsequent calls to
    ``get_tools()`` return the cached list without reconnecting.
    """

    def __init__(self) -> None:
        self._tools: list[Any] | None = None
        self._lock = asyncio.Lock()

    async def get_tools(self) -> list[Any]:
        """Return cached tools, discovering on first call."""
        if self._tools is not None:
            return self._tools
        async with self._lock:
            if self._tools is not None:
                return self._tools
            return await self._connect()

    async def _connect(self) -> list[Any]:
        """Discover tools via a temporary connection session."""
        settings = get_settings()

        connection = {
            "transport": "streamable_http",
            "url": settings.MCP_SERVER_URL,
        }

        # session=None forces connection mode:
        # - discovery: creates a temp session to list tools
        # - execution: each tool call creates a temp session with
        #   interceptor headers merged into connection headers
        self._tools = await load_mcp_tools(
            session=None,
            connection=connection,
            tool_interceptors=[_AuthInterceptor()],
        )
        logger.info(
            "MCP tools discovered — %d tools (connection mode)",
            len(self._tools),
        )
        return self._tools

    async def reconnect(self) -> list[Any]:
        """Force-reconnect and re-discover tools."""
        async with self._lock:
            self._tools = None
            return await self._connect()


# Module-level singleton
_mcp_client = PersistentMCPClient()


async def get_tools() -> list[Any]:
    """Get MCP tools via the persistent session (preferred entry-point)."""
    return await _mcp_client.get_tools()


async def reconnect_tools() -> list[Any]:
    """Force-reconnect and re-discover tools."""
    return await _mcp_client.reconnect()
