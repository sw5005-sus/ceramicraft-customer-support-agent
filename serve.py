"""Entrypoint for the Customer Support Agent.

Starts both an HTTP (FastAPI/uvicorn) server and a gRPC server,
following the same dual-server pattern as notification-ms.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import dttb
import grpc
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ceramicraft_customer_support_agent.config import get_settings
from ceramicraft_customer_support_agent.graph import build_checkpointer, build_graph
from ceramicraft_customer_support_agent.grpc_service import (
    CustomerSupportAgentServicer,
)
from ceramicraft_customer_support_agent.mcp_client import get_tools, set_auth_token
from ceramicraft_customer_support_agent.mlflow_utils import (
    init_mlflow_tracing,
    tag_trace,
)
from ceramicraft_customer_support_agent.pb import cs_agent_pb2_grpc

# Apply dttb tracebacks for timestamps on exceptions
dttb.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level cache: agent and checkpointer, built once at startup.
_agent_cache: dict[str, Any] = {}


def _extract_bearer_token(request: Request) -> str | None:
    """Extract Bearer token from the HTTP Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Build checkpointer, discover MCP tools and compile graph on startup.

    Failure here is fatal — the service cannot handle requests without a
    working checkpointer and agent.  Log the error and re-raise so the
    process exits and the container scheduler can restart it.
    """
    init_mlflow_tracing()
    checkpointer = await build_checkpointer()
    tools = await get_tools()
    _agent_cache["agent"] = await build_graph(tools, checkpointer=checkpointer)
    _agent_cache["checkpointer"] = checkpointer
    logger.info("Agent pre-warmed with %d tools", len(tools))
    yield


app = FastAPI(
    title="CeramiCraft Customer Support Agent",
    lifespan=lifespan,
)


# ---------- request / response models ----------


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message or question.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID. Omit to start a new conversation; "
        "include the returned thread_id to continue an existing one.",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str = Field(
        ..., description="Thread ID for this conversation. Pass it back to continue."
    )


class ResetResponse(BaseModel):
    status: str
    message: str


# ---------- endpoints ----------


@app.get("/cs-agent/v1/ping")
async def ping():
    """Readiness probe: returns 200 only when the agent is initialised."""
    if "agent" not in _agent_cache:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Chat with the CeramiCraft customer support agent."""
    token = _extract_bearer_token(request)
    thread_id = body.thread_id or uuid.uuid4().hex

    try:
        agent = _agent_cache["agent"]
        set_auth_token(token)

        initial_state = {
            "messages": [{"role": "user", "content": body.message}],
            "auth_token": token,
        }

        response = await agent.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        tag_trace(
            {
                "intent": response.get("intent", "unknown"),
                "authenticated": "true" if token else "false",
                "thread_id": thread_id,
            }
        )
    except Exception:
        logger.exception("Agent invocation failed")
        return JSONResponse(
            status_code=500,
            content={
                "reply": "Sorry, something went wrong processing your request. Please try again.",
                "thread_id": thread_id,
            },
        )

    # Extract the last assistant message
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            return ChatResponse(reply=msg.content, thread_id=thread_id)
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("content")
        ):
            return ChatResponse(reply=msg["content"], thread_id=thread_id)

    return ChatResponse(
        reply="I'm sorry, I couldn't process your request.", thread_id=thread_id
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(thread_id: str):
    """Reset the conversation history for a given thread."""
    checkpointer = _agent_cache["checkpointer"]
    try:
        await checkpointer.adelete_thread(thread_id)
    except Exception:
        logger.exception("Failed to delete checkpoints for thread %s", thread_id)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to reset conversation '{thread_id}'.",
            },
        )

    return ResetResponse(
        status="ok",
        message=f"Conversation '{thread_id}' reset successfully.",
    )


async def _start() -> None:
    """Start both HTTP and gRPC servers (notification-ms pattern)."""
    settings = get_settings()

    # --- HTTP (FastAPI + uvicorn) ---
    http_config = uvicorn.Config(
        app,
        host=settings.AGENT_HOST,
        port=settings.AGENT_PORT,
        log_level="info",
    )
    http_server = uvicorn.Server(http_config)

    # --- gRPC ---
    # Agent and checkpointer are initialised by FastAPI lifespan.
    # We need them for the gRPC servicer, but they're not available until
    # the HTTP server starts.  Use a simple event to synchronise.
    agent_ready = asyncio.Event()
    _original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _patched_lifespan(a):  # noqa: ANN001
        async with _original_lifespan(a) as val:
            agent_ready.set()
            yield val

    app.router.lifespan_context = _patched_lifespan

    async def _run_grpc() -> None:
        await agent_ready.wait()
        grpc_server = grpc.aio.server()
        cs_agent_pb2_grpc.add_CustomerSupportAgentServicer_to_server(
            CustomerSupportAgentServicer(
                agent=_agent_cache["agent"],
                checkpointer=_agent_cache["checkpointer"],
            ),
            grpc_server,
        )
        grpc_addr = f"{settings.AGENT_GRPC_HOST}:{settings.AGENT_GRPC_PORT}"
        grpc_server.add_insecure_port(grpc_addr)
        await grpc_server.start()
        logger.info("gRPC server listening on %s", grpc_addr)
        await grpc_server.wait_for_termination()

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(http_server.serve())
            tg.create_task(_run_grpc())
    except BaseException:
        logger.info("Shutting down servers")
        raise


def main() -> None:
    """Start the Customer Support Agent (HTTP + gRPC)."""
    logger.info("Starting Customer Support Agent (HTTP + gRPC)...")
    asyncio.run(_start())


if __name__ == "__main__":
    main()
