"""MCP Client — persistent session and tool discovery."""

import asyncio
import contextvars
import logging
from typing import Any

from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    MCPToolCallResult,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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
    """Maintains a long-lived MCP session so tool handles stay valid.

    ``load_mcp_tools()`` binds each tool to the session that created it.
    If that session closes, the tools raise ``ClosedResourceError``.
    This class keeps the session open for the lifetime of the process
    and exposes a reconnect path in case the upstream server restarts.
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._session_cm: Any = None  # ClientSession context manager
        self._tools: list[Any] | None = None
        self._transport_cm: Any = None  # streamablehttp_client context manager
        self._lock = asyncio.Lock()

    async def get_tools(self) -> list[Any]:
        """Return cached tools, connecting on first call."""
        if self._tools is not None and self._session is not None:
            return self._tools
        async with self._lock:
            # Double-check after acquiring lock
            if self._tools is not None and self._session is not None:
                return self._tools
            return await self._connect()

    async def _connect(self) -> list[Any]:
        """Open a persistent MCP session and discover tools."""
        await self._close()

        settings = get_settings()
        headers: dict[str, str] = {}

        # Open streamable-http transport — we hold references to keep
        # the underlying connection alive.
        self._transport_cm = streamablehttp_client(
            settings.MCP_SERVER_URL, headers=headers
        )
        read, write, _ = await self._transport_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        self._tools = await load_mcp_tools(
            self._session,
            tool_interceptors=[_AuthInterceptor()],
        )
        logger.info(
            "Persistent MCP session established — discovered %d tools",
            len(self._tools),
        )
        return self._tools

    async def reconnect(self) -> list[Any]:
        """Force-reconnect (e.g. after upstream restart)."""
        async with self._lock:
            return await self._connect()

    async def _close(self) -> None:
        """Best-effort cleanup of existing session."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._session = None
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_cm = None
        self._tools = None


# Module-level singleton
_mcp_client = PersistentMCPClient()


async def get_tools() -> list[Any]:
    """Get MCP tools via the persistent session (preferred entry-point)."""
    return await _mcp_client.get_tools()


async def reconnect_tools() -> list[Any]:
    """Force-reconnect and re-discover tools."""
    return await _mcp_client.reconnect()
