"""Entrypoint for the Customer Support Agent.

Starts both an HTTP (FastAPI/uvicorn) server and a gRPC server,
following the same dual-server pattern as notification-ms.
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import dttb
import grpc
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ceramicraft_customer_support_agent.config import get_settings

# Forward CS_AGENT_LANGSMITH_API_KEY to the env vars that LangChain reads
# directly (LANGCHAIN_API_KEY / LANGCHAIN_TRACING_V2).  This must happen
# before any LangChain import.
_settings_early = get_settings()
if _settings_early.CS_AGENT_LANGSMITH_API_KEY:
    os.environ.setdefault(
        "LANGCHAIN_API_KEY", _settings_early.CS_AGENT_LANGSMITH_API_KEY
    )
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault(
        "LANGCHAIN_PROJECT", _settings_early.CS_AGENT_LANGSMITH_PROJECT
    )
del _settings_early

from ceramicraft_customer_support_agent.graph import build_checkpointer, build_graph  # noqa: E402
from ceramicraft_customer_support_agent.grpc_service import (  # noqa: E402
    CustomerSupportAgentServicer,
)
from ceramicraft_customer_support_agent.mcp_client import get_tools, set_auth_token  # noqa: E402
from ceramicraft_customer_support_agent.mlflow_utils import (  # noqa: E402
    init_mlflow_tracing,
    tag_trace,
)
from ceramicraft_customer_support_agent.pb import cs_agent_pb2_grpc  # noqa: E402

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

# --- CORS ---
_cors_raw = get_settings().CS_AGENT_CORS_ORIGINS.strip()
if _cors_raw:
    _origins = (
        ["*"]
        if _cors_raw == "*"
        else [o.strip() for o in _cors_raw.split(",") if o.strip()]
    )
    app.add_middleware(  # type: ignore[arg-type]
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Nodes in the graph whose completion maps to an SSE stage event.
_STAGE_EVENTS: dict[str, str] = {
    "input_guard": "guarding",
    "classifier": "classifying",
}
# Domain subgraph nodes — all map to "processing".
_DOMAIN_NODES = frozenset(
    {"browse", "cart", "order", "review", "account", "chitchat", "escalate"}
)


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Chat with the CS agent, streaming progress via Server-Sent Events.

    SSE event types:
        classifying  — intent classification started
        processing   — domain subgraph running (data includes intent)
        reply        — final assistant reply text
        error        — an error occurred
        done         — stream complete (data includes thread_id)
    """
    token = _extract_bearer_token(request)
    thread_id = body.thread_id or uuid.uuid4().hex

    async def _generate() -> AsyncGenerator[str, None]:
        try:
            agent = _agent_cache["agent"]
            set_auth_token(token)

            initial_state = {
                "messages": [{"role": "user", "content": body.message}],
                "auth_token": token,
            }

            intent = "unknown"
            reply = ""

            async for chunk in agent.astream(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                # chunk is a dict of {node_name: state_update}
                for node_name, update in chunk.items():
                    # Emit stage events
                    if node_name in _STAGE_EVENTS:
                        yield _sse_event(_STAGE_EVENTS[node_name], {"stage": node_name})
                    elif node_name in _DOMAIN_NODES:
                        # Capture intent from previous classifier update
                        if isinstance(update, dict) and "intent" in update:
                            intent = update["intent"]
                        yield _sse_event(
                            "processing",
                            {"stage": node_name, "intent": intent},
                        )

                    # Track intent from classifier node
                    if (
                        node_name == "classifier"
                        and isinstance(update, dict)
                        and "intent" in update
                    ):
                        intent = update["intent"]

                    # Extract reply from the final messages
                    if isinstance(update, dict) and "messages" in update:
                        for msg in reversed(update["messages"]):
                            if (
                                hasattr(msg, "type")
                                and msg.type == "ai"
                                and msg.content
                            ):
                                reply = msg.content
                                break
                            if (
                                isinstance(msg, dict)
                                and msg.get("role") == "assistant"
                                and msg.get("content")
                            ):
                                reply = msg["content"]
                                break

            if reply:
                yield _sse_event("reply", {"content": reply})
            else:
                yield _sse_event(
                    "reply",
                    {"content": "I'm sorry, I couldn't process your request."},
                )

            tag_trace(
                {
                    "intent": intent,
                    "authenticated": "true" if token else "false",
                    "thread_id": thread_id,
                }
            )

        except Exception:
            logger.exception("Stream chat failed")
            yield _sse_event(
                "error",
                {
                    "message": "Sorry, something went wrong processing your request. "
                    "Please try again."
                },
            )

        yield _sse_event("done", {"thread_id": thread_id})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
        host=settings.CS_AGENT_HTTP_HOST,
        port=settings.CS_AGENT_HTTP_PORT,
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
        grpc_addr = f"{settings.CS_AGENT_GRPC_HOST}:{settings.CS_AGENT_GRPC_PORT}"
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
