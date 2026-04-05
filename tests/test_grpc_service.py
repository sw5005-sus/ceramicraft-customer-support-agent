"""Tests for the gRPC CustomerSupportAgent servicer."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from ceramicraft_customer_support_agent.grpc_service import (
    CustomerSupportAgentServicer,
)
from ceramicraft_customer_support_agent.pb import cs_agent_pb2


def _make_ai_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.type = "ai"
    msg.content = content
    return msg


def _build_servicer(
    agent_response: dict[str, Any] | Exception,
    checkpointer: Any = None,
) -> CustomerSupportAgentServicer:
    agent = AsyncMock()
    if isinstance(agent_response, Exception):
        agent.ainvoke.side_effect = agent_response
    else:
        agent.ainvoke.return_value = agent_response
    if checkpointer is None:
        checkpointer = AsyncMock()
    return CustomerSupportAgentServicer(agent=agent, checkpointer=checkpointer)


@pytest.fixture
def context() -> MagicMock:
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    ctx.abort = AsyncMock()
    return ctx


# ──── Chat tests ────


@pytest.mark.asyncio
async def test_chat_basic(context: MagicMock) -> None:
    servicer = _build_servicer(
        {"messages": [_make_ai_message("Hello!")], "intent": "chitchat"}
    )
    req = cs_agent_pb2.ChatRequest(message="hi", thread_id="t1", auth_token="tok")
    with patch("ceramicraft_customer_support_agent.grpc_service.set_auth_token"):
        resp = await servicer.Chat(req, context)

    assert resp.reply == "Hello!"
    assert resp.thread_id == "t1"


@pytest.mark.asyncio
async def test_chat_auto_thread_id(context: MagicMock) -> None:
    servicer = _build_servicer(
        {"messages": [_make_ai_message("Hi")], "intent": "browse"}
    )
    req = cs_agent_pb2.ChatRequest(message="hello")
    with patch("ceramicraft_customer_support_agent.grpc_service.set_auth_token"):
        resp = await servicer.Chat(req, context)

    assert resp.reply == "Hi"
    assert len(resp.thread_id) == 32  # uuid4().hex


@pytest.mark.asyncio
async def test_chat_agent_error(context: MagicMock) -> None:
    servicer = _build_servicer(RuntimeError("boom"))
    req = cs_agent_pb2.ChatRequest(message="hello", thread_id="t1")
    with patch("ceramicraft_customer_support_agent.grpc_service.set_auth_token"):
        resp = await servicer.Chat(req, context)

    assert "went wrong" in resp.reply
    assert resp.thread_id == "t1"


@pytest.mark.asyncio
async def test_chat_no_ai_message(context: MagicMock) -> None:
    servicer = _build_servicer({"messages": [], "intent": "chitchat"})
    req = cs_agent_pb2.ChatRequest(message="hello", thread_id="t1")
    with patch("ceramicraft_customer_support_agent.grpc_service.set_auth_token"):
        resp = await servicer.Chat(req, context)

    assert "couldn't process" in resp.reply


@pytest.mark.asyncio
async def test_chat_dict_message(context: MagicMock) -> None:
    """Test that dict-style assistant messages are also handled."""
    servicer = _build_servicer(
        {
            "messages": [{"role": "assistant", "content": "Dict reply"}],
            "intent": "browse",
        }
    )
    req = cs_agent_pb2.ChatRequest(message="hello", thread_id="t1")
    with patch("ceramicraft_customer_support_agent.grpc_service.set_auth_token"):
        resp = await servicer.Chat(req, context)

    assert resp.reply == "Dict reply"


# ──── Reset tests ────


@pytest.mark.asyncio
async def test_reset_success(context: MagicMock) -> None:
    checkpointer = AsyncMock()
    servicer = _build_servicer({}, checkpointer=checkpointer)
    req = cs_agent_pb2.ResetRequest(thread_id="t1")
    resp = await servicer.Reset(req, context)

    assert resp.status == "ok"
    checkpointer.adelete_thread.assert_awaited_once_with("t1")


@pytest.mark.asyncio
async def test_reset_failure(context: MagicMock) -> None:
    checkpointer = AsyncMock()
    checkpointer.adelete_thread.side_effect = RuntimeError("db error")
    servicer = _build_servicer({}, checkpointer=checkpointer)
    req = cs_agent_pb2.ResetRequest(thread_id="t1")
    await servicer.Reset(req, context)

    context.abort.assert_awaited_once()
