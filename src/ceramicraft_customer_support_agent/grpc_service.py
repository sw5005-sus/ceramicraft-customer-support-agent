"""gRPC servicer for the Customer Support Agent."""

import logging
import uuid

import grpc

from .mcp_client import set_auth_token
from .mlflow_utils import tag_trace
from .pb import cs_agent_pb2, cs_agent_pb2_grpc

logger = logging.getLogger(__name__)


class CustomerSupportAgentServicer(
    cs_agent_pb2_grpc.CustomerSupportAgentServicer,
):
    """Async gRPC servicer — mirrors the REST /chat and /reset endpoints."""

    def __init__(self, agent, checkpointer) -> None:  # noqa: ANN001
        self._agent = agent
        self._checkpointer = checkpointer

    async def Chat(
        self,
        request: cs_agent_pb2.ChatRequest,
        context: grpc.aio.ServicerContext,
    ) -> cs_agent_pb2.ChatResponse:
        thread_id = request.thread_id or uuid.uuid4().hex

        # Extract Bearer token from gRPC metadata (same as HTTP Authorization header)
        metadata = dict(context.invocation_metadata())  # ty: ignore[no-matching-overload]
        auth_header = metadata.get("authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else None

        try:
            set_auth_token(token)

            initial_state = {
                "messages": [{"role": "user", "content": request.message}],
                "auth_token": token,
            }

            response = await self._agent.ainvoke(
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
            logger.exception("gRPC Chat failed")
            return cs_agent_pb2.ChatResponse(
                reply="Sorry, something went wrong processing your request. Please try again.",
                thread_id=thread_id,
            )

        # Extract the last assistant message
        messages = response.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                return cs_agent_pb2.ChatResponse(reply=msg.content, thread_id=thread_id)
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("content")
            ):
                return cs_agent_pb2.ChatResponse(
                    reply=msg["content"], thread_id=thread_id
                )

        return cs_agent_pb2.ChatResponse(
            reply="I'm sorry, I couldn't process your request.",
            thread_id=thread_id,
        )

    async def Reset(
        self,
        request: cs_agent_pb2.ResetRequest,
        context: grpc.aio.ServicerContext,
    ) -> cs_agent_pb2.ResetResponse:
        try:
            await self._checkpointer.adelete_thread(request.thread_id)
        except Exception:
            logger.exception("gRPC Reset failed for thread %s", request.thread_id)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to reset conversation '{request.thread_id}'.",
            )
            # unreachable but keeps type checker happy
            return cs_agent_pb2.ResetResponse(status="error", message="")

        return cs_agent_pb2.ResetResponse(
            status="ok",
            message=f"Conversation '{request.thread_id}' reset successfully.",
        )
