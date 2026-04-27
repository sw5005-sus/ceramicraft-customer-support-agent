# Customer Support Agent — 开发计划

_最后更新：2026-04-19_

---

## 设计决策

| 问题 | 决定 |
|------|------|
| 对外接口 | FastAPI REST + SSE streaming + gRPC（双服务同进程启动） |
| 对话状态 | PostgreSQL checkpointer（`langgraph-checkpoint-postgres`） |
| 鉴权 | Agent 不验证 token，纯透传给下游 MCP Server |
| 敏感操作 | 下单/确认收货/删地址需二次确认；加购物车不需确认 |
| Agent 架构 | LangGraph StateGraph：输入安全检测 → 意图分类 → 条件路由 → 领域子图（ReAct）→ Auth-error 兜底 Guard |
| LLM | OpenAI GPT-4o |
| 设计原则 | Prompt Engineering + Context Engineering + 渐进式披露 |
| MCP Client | PersistentMCPClient 单例（connection 模式），工具列表动态发现并缓存 |
| 可观测性 | MLflow tracing（autolog，non-inline tracer）+ Prompt Registry + Session grouping；tracing 失败不影响业务 |
| 输入安全 | Pre-LLM regex guard 检测 prompt injection / jailbreak，零延迟零成本 |

---

## 架构设计

### StateGraph 流水线

```
  ┌──────────┐
  │ __start__ │
  └─────┬────┘
        ▼
  ┌─────────────┐
  │ Input Guard │ ← 预检：prompt injection / jailbreak 检测（regex，pre-LLM）
  └─────┬───────┘
        │ blocked → Guard（拒绝消息）
        │ safe ↓
  ┌────────────┐
  │ Classifier │ ← 意图分类（Pydantic structured output，无工具）
  └─────┬──────┘
        │ conditional edges
        ├──→ Domain Subgraphs (browse / cart / order / review / account)
        ├──→ Chitchat Node    (纯 LLM，无工具)
        └──→ Escalate Node    (固定转人工消息)
                │
                ▼
          ┌─────────┐
          │  Guard  │ ← auth-error 兜底与用户友好登录提示
          └─────────┘
                │
                ▼
          ┌──────────┐
          │ __end__  │
          └──────────┘
```

每个领域子图是独立的 `create_react_agent`（无状态，无 checkpointer），只绑定本领域相关工具，减少 token 消耗和幻觉。进入子图前会清洗 chat history 中的 orphan tool messages；截断历史窗口后会再次清洗，避免 OpenAI 接收没有前置 `tool_calls` 的 `tool` 消息。具体工具分配见 `subgraphs.py`。

### 工具发现

Tools 从 ceramicraft-mcp-server 动态发现（启动时通过 PersistentMCPClient 建临时 session 拉取 `list_tools`，结果缓存），转换为 LangChain Tool 格式注入 agent。MCP Server 新增 tool 时 agent 无需改代码。

MCP Client 使用 connection 模式（`session=None, connection=StreamableHttpConnection`），每次 tool 调用建临时 HTTP session，`_AuthInterceptor` 将 per-request Bearer token 注入到 connection headers。

### 对话隔离

通过 `thread_id` 隔离多用户对话。`thread_id` 可选——首次调用不传时服务端自动生成，response 中始终返回。使用 `AsyncPostgresSaver` 跨进程持久化。

### Token 透传链路

```
用户 JWT ──▶ Agent (FastAPI, 提取 Bearer token)
                │
                ├─▶ AgentState.auth_token ──▶ _wrap_subgraph 注入 SystemMessage
                │                              + Guard 做 auth-error 兜底提示
                │
                └─▶ set_auth_token(token)  ──▶ contextvars.ContextVar
                                                     │
                     _AuthInterceptor ◀──────────────┘
                         │
                         ▼
                     下游 MCP Server (ceramicraft-mcp-server)
```

Agent 不做 JWT 验证。token 通过 `contextvars.ContextVar` 传递给 `_AuthInterceptor`，在每次 MCP tool 调用时注入 Authorization header。支持并发请求——每个 asyncio Task 有独立的 context。

### 为什么用 FastAPI 而不是 FastMCP

最初 cs-agent 对外暴露的也是 MCP Server (FastMCP)。但 LangGraph 内部使用 `asyncio.create_task()` 创建任务，与 FastMCP 底层 anyio 的 cancel scope 冲突：

```
RuntimeError: Attempted to exit a cancel scope that isn't the current task's current cancel scope
```

FastAPI 虽然也用 anyio，但不像 FastMCP 那样将 handler 包在严格的 cancel scope 内，因此不受影响。

---

## 已完成里程碑

### Phase 1 — 核心骨架 ✅
Token 透传、MCP Client（connection 模式）、LangGraph StateGraph（7 意图分类 + 5 领域子图 + Guard）、REST API、PostgreSQL checkpointer。

### Phase 2 — 功能模块 ✅
商品咨询、购物车、订单、评价、账户管理全部实现。Intent-routed StateGraph + domain-specific prompts。

### Phase 3 — 质量与上线 ✅
- Prompt 优化（渐进式披露 + 领域专属 prompt）
- 测试（单元 + 集成 + 安全 + DeepEval LLM 评估）
- CI/CD（lint / test / snyk / trivy / sonar / deepeval workflows）
- MLflow Integration（tracing + Prompt Registry + session grouping + evaluation；tracer 使用 non-inline 模式以避免 async contextvar token warning）
- Input Security（pre-LLM regex guard + DeepEval 评估）

### Phase 4 — SSE Streaming + Web UI ✅
- SSE endpoint `POST /chat/stream`（`text/event-stream`，事件：`guarding` → `classifying` → `processing` → `reply` → `done`）
- 使用 LangGraph `agent.astream(stream_mode="updates")` 按节点产出 chunk
- Configurable CORS middleware（`CS_AGENT_CORS_ORIGINS` 环境变量，默认关闭）
- Prompt 强化：ORDER_PROMPT 强制 5 步下单流程 + 购物车自动清空 + 语言匹配
- Web UI（`ceramicraft-customer-support-agent-ui`）：Vue 3 + TypeScript + Zitadel PKCE auth + SSE chat + markdown 渲染 + 会话持久化
- UI CI/CD：lint / test (64 tests, 95% coverage) / snyk / trivy / sonar / deploy workflows

---

## 暂不实现（Backlog）

- Vault 注入 PG 凭证
- gRPC 流式支持（对应 SSE 的 gRPC server-side streaming）
- Zitadel SPA 应用（目前 UI 复用移动端 Native app client_id）
