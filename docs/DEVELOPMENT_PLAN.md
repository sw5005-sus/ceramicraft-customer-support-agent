# Customer Support Agent — 开发计划

_最后更新：2026-03-31_

---

## 设计决策

| 问题 | 决定 |
|------|------|
| 对外接口 | FastAPI REST (`POST /chat`, `POST /reset`, `GET /cs-agent/v1/ping`) + gRPC (`Chat`, `Reset` on port 50051) |
| 对话状态 | PostgreSQL checkpointer（`langgraph-checkpoint-postgres`）；POSTGRES_HOST 必填，未配置则启动报错 |
| 鉴权 | Agent 不验证 token，纯透传给下游 MCP Server（由 MCP Server 统一验证） |
| 敏感操作 | 下单前展示购物车并要求确认，确认收货和删地址也需确认；加购物车不需确认 |
| Agent 架构 | LangGraph StateGraph：意图分类 → 条件路由 → 领域子图（ReAct）→ 安全守卫 |
| LLM | OpenAI GPT-4o |
| 设计原则 | Prompt Engineering + Context Engineering + 渐进式披露 |
| MCP Client | PersistentMCPClient 单例（connection 模式），工具列表缓存，每次调用建临时 session |
| 可观测性 | MLflow tracing（autolog）+ Prompt Registry；tracing 失败不影响业务；prompt fallback 到硬编码默认值 |

---

## 架构设计

### StateGraph 流水线

```
  ┌──────────┐
  │ __start__ │
  └─────┬────┘
        ▼
  ┌────────────┐
  │ Classifier │ ← 意图分类（Pydantic structured output，无工具）
  └─────┬──────┘
        │ conditional edges
        ├──→ Browse Subgraph   (search_products, get_product, list_product_reviews)
        ├──→ Cart Subgraph     (get_cart, add/update/remove_cart_item, estimate_cart_price, search_products)
        ├──→ Order Subgraph    (list_my_orders, get_order_detail, confirm_receipt, get_order_stats, create_order)
        ├──→ Review Subgraph   (create_review, like_review, get_user_reviews, list_product_reviews)
        ├──→ Account Subgraph  (get/update_my_profile, addresses CRUD, get_pay_account, top_up_account)
        ├──→ Chitchat Node     (纯 LLM，无工具)
        └──→ Escalate Node     (固定转人工消息)
                │
                ▼
          ┌─────────┐
          │  Guard  │ ← auth 检查 + 敏感操作确认
          └─────────┘
                │
                ▼
          ┌──────────┐
          │ __end__  │
          └──────────┘
```

每个领域子图是独立的 `create_react_agent`（**无状态，无 checkpointer**），只绑定本领域相关工具，减少 token 消耗和幻觉。

### 工具发现

Tools 从 ceramicraft-mcp-server **动态发现**（启动时通过 PersistentMCPClient 建临时 session 拉取 `list_tools`，结果缓存），转换为 LangChain Tool 格式注入 agent。MCP Server 新增 tool 时 agent 无需改代码。

MCP Client 使用 **connection 模式**（`session=None, connection=StreamableHttpConnection`），每次 tool 调用建临时 HTTP session，`_AuthInterceptor` 将 per-request Bearer token 注入到 connection headers。

### 对话隔离

通过 `thread_id` 隔离多用户对话。每个用户独立的对话历史和上下文。`thread_id` 可选——首次调用不传时服务端自动生成（`uuid4().hex`），response 中始终返回 `thread_id`，后续调用传回即可继续。使用 `AsyncPostgresSaver` 跨进程持久化，重启不丢失对话历史。

### Token 透传链路

```
用户 JWT ──▶ Agent (FastAPI, 提取 Bearer token)
                │
                ├─▶ AgentState.auth_token ──▶ _wrap_subgraph 注入 SystemMessage
                │                              + Guard 兜底检查
                │
                └─▶ set_auth_token(token)  ──▶ contextvars.ContextVar
                                                     │
                     _AuthInterceptor ◀──────────────┘
                         │
                         ▼
                     MCPToolCallRequest.override(headers={"authorization": "Bearer ..."})
                         │
                         ▼
                     下游 MCP Server (ceramicraft-mcp-server)
```

Agent 不做 JWT 验证。token 通过 `contextvars.ContextVar` 传递给 `_AuthInterceptor`（`langchain-mcp-adapters` ToolCallInterceptor），在每次 MCP tool 调用时注入 Authorization header。支持并发请求——每个 asyncio Task 有独立的 context。

### 为什么用 FastAPI 而不是 FastMCP

最初 cs-agent 对外暴露的也是 MCP Server (FastMCP)。但 LangGraph 内部使用 `asyncio.create_task()` 创建任务，与 FastMCP 底层 anyio 的 cancel scope 冲突：

```
RuntimeError: Attempted to exit a cancel scope that isn't the current task's current cancel scope
```

FastAPI 虽然也用 anyio，但不像 FastMCP 那样将 handler 包在严格的 cancel scope 内，因此不受影响。

旧的 `mcp_server.py` 已删除（commit `8fe6218`）。

### 渐进式披露

体现在 System Prompt 设计：首次交互简短问候 + 概要能力介绍，用户追问时按需展开，不主动列出所有功能。

---

## Phase 1 — 核心骨架 ✅

**目标：跑通 MCP Client + LangGraph agent 的完整链路**

### 1.1 Token 透传
- 从 HTTP Authorization header 提取 Bearer token
- 存入 AgentState.auth_token
- `_wrap_subgraph` 注入 `SystemMessage` 告知 LLM 用户认证状态（已认证/未认证）
- Guard 节点对未认证用户的 auth-error 消息追加登录提示
- Subgraph 调用 wrap 在 try/except 中，工具异常返回友好提示而非 500

### 1.2 MCP Client（调下游）
- PersistentMCPClient 单例（connection 模式）连接 ceramicraft-mcp-server（Streamable HTTP）
- 工具发现：启动时建临时 session 拉取可用 tools 列表（结果缓存）
- 每次 tool 调用建临时 session，_AuthInterceptor 注入 per-request auth header

### 1.3 LangGraph Agent
- StateGraph：Classifier → Router → Domain Subgraphs → Guard
- 7 种意图分类：browse / cart / order / review / account / chitchat / escalate
- 5 个领域子图（ReAct agent，stateless per-invocation），各自绑定领域工具
- 2 个轻量节点：chitchat（纯 LLM）、escalate（固定消息）
- Guard 节点：auth 检查 + 敏感操作确认
- PostgreSQL checkpointer（`langgraph-checkpoint-postgres`）管理对话状态（进程级单例，跨请求持久）；POSTGRES_HOST 未配置则启动失败
- `_trim_messages()` 限制传给子图的历史长度（`AGENT_MAX_HISTORY`，默认 20）
- `_sanitize_messages()` 过滤 classifier 产生的 orphaned tool_calls

### 1.4 REST API（对外暴露）
- `POST /chat`：接收用户消息 + 可选 thread_id，返回 agent 回复 + thread_id
- `POST /reset`：重置对话历史（调用 `AsyncPostgresSaver.adelete_thread`）
- `GET /cs-agent/v1/ping`：健康检查
- `GET /docs`：Swagger UI（自动生成）

### 1.5 Health + Serve
- FastAPI lifespan hook 预热 agent
- `serve.py` 入口启动

**交付物：** 能通过 curl / Swagger 对话，agent 能调下游 tools 查商品

---

## Phase 2 — 功能模块 ✅

> Intent-routed StateGraph, connection-mode MCP client with tool caching, domain-specific prompts, guard node.
> 96+ tests, 97% coverage.

### 2.1 商品咨询（PUBLIC，无需 token）
- search_products：搜索商品
- get_product：查看详情
- list_product_reviews：查看评价
- Agent 能力：理解模糊查询、对比推荐、总结评价

### 2.2 购物车管理（USER）
- get_cart, add_to_cart, update_cart_item, remove_cart_item
- estimate_cart_price
- Agent 能力：引导加购、确认操作、展示价格

### 2.3 订单管理（USER）
- list_my_orders, get_order_detail, get_order_stats
- create_order, confirm_receipt（Guard 节点要求确认）
- Agent 能力：查询订单状态、解释物流信息、引导下单

### 2.4 评价互动（USER）
- create_review, get_user_reviews, like_review
- Agent 能力：引导写评价、展示历史评价

### 2.5 账户管理（USER）
- get_my_profile, update_my_profile
- list/create/update/delete_address（delete 需 Guard 确认）
- Agent 能力：查看和修改个人信息、地址管理

### ~~2.6 支付~~ → 已实现（get_pay_account + top_up_account 归入 Account 子图）
### ~~2.7 充值~~ → 已实现（通过 redeem code 充值）

---

## Phase 3 — 质量与上线

### 3.1 Prompt 优化
- Context Engineering：根据对话阶段动态注入上下文
- 渐进式披露：首次交互简洁介绍，按需展开
- Few-shot examples 提升工具调用准确率

### 3.2 测试
- 单元测试：auth、config、tool 注册、intent classification、guard、subgraphs
- 集成测试：mock MCP Server，验证 agent 调用链路
- 对话测试：典型场景覆盖
- serve.py 测试：FastAPI TestClient 验证 REST 端点

### 3.3 CI/CD
- deploy.yml（复用现有模式）
- Helm chart + argocd-deploy 配置
- LangSmith tracing 集成

### 3.4 状态持久化 ✅
- ~~切换到 `langgraph-checkpoint-postgres`~~ — **已完成**
- `POSTGRES_USER/PASSWORD/HOST/PORT` 接入共享 `ceramicraft-postgres`（与 log-ms / notification-ms 一致）
- `cs_agent_db` 由 local-stack postgres init script 自动创建
- Vault 注入 PG 凭证（TODO）

### 3.5 Token 端到端透传
- 将 auth_token 从 AgentState 注入到每次 MCP tool 调用的 headers 中
- 或在 PersistentMCPClient 中支持 per-request token forwarding

### 3.6 MLflow Integration ✅
- MLflow tracing via `mlflow.langchain.autolog()` — auto-captures full LangGraph execution traces
- Prompt Registry integration — prompts loadable from MLflow with local fallback
- Custom span tags via `tag_trace()` — records `intent`, `authenticated`, `thread_id` per request
- Evaluation script `scripts/run_evaluation.py` — 12-case test dataset, logs metrics to `ceramicraft-cs-agent-eval` experiment
- Graceful degradation — app works without MLflow installed/configured

---

## 项目结构

```
ceramicraft-customer-support-agent/
├── serve.py                          # FastAPI 入口（POST /chat, /reset, GET /cs-agent/v1/ping）
├── src/ceramicraft_customer_support_agent/
│   ├── __init__.py
│   ├── config.py                     # 配置（pydantic-settings）
│   ├── agent.py                      # 向后兼容接口（调 build_graph）
│   ├── classifier.py                 # 意图分类节点（Intent enum + Pydantic structured output）
│   ├── graph.py                      # StateGraph 主图（AgentState + build_graph + _wrap_subgraph + auth 注入 + 异常捕获）
│   ├── guard.py                      # 安全守卫节点（auth 兜底检查 + 敏感操作确认）
│   ├── nodes.py                      # 轻量节点（chitchat + escalate，无工具）
│   ├── subgraphs.py                  # 领域子图（5 个 stateless ReAct agent builder）
│   ├── grpc_service.py                   # gRPC servicer (async Chat + Reset, mirrors REST endpoints)
│   ├── mcp_client.py                 # MCP Client（PersistentMCPClient 单例，connection 模式 + auth interceptor）
│   ├── mlflow_utils.py               # MLflow tracing init + tag_trace() helper (tracing failures non-fatal)
│   ├── prompts.py                    # System prompt 模板（主 + 6 个领域）
│   └── pb/                           # Generated protobuf (cs_agent_pb2*.py)
├── protos/
│   └── cs_agent.proto                    # gRPC service definition (Chat + Reset)
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_classifier.py
│   ├── test_config.py
│   ├── test_graph.py
│   ├── test_guard.py
│   ├── test_mcp_client.py            # PersistentMCPClient + AuthInterceptor 测试
│   ├── test_nodes.py
│   ├── test_prompts.py
│   ├── test_serve.py                 # FastAPI 端点测试（新增）
│   ├── test_grpc_service.py          # gRPC servicer 测试（Chat + Reset）
│   └── test_subgraphs.py
├── scripts/
│   └── run_evaluation.py             # MLflow evaluation script (12-case test dataset, logs to ceramicraft-cs-agent-eval)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 暂不实现（Backlog）

- [x] ~~top_up_account / get_pay_account（支付）~~ — 已实现，归入 Account 子图
- [x] ~~register_push_token（通知）~~ — 不实现，应由前端直接调 notification-ms API
- [x] ~~PostgreSQL checkpoint 持久化~~ — 已完成
- [x] ~~Token 端到端透传到下游 MCP Server~~ — 已完成（ContextVar + _AuthInterceptor）
- [x] ~~thread_id 改为可选，服务端自动生成，response 返回~~ — 已完成
- [x] ~~子图拆分（按意图路由）~~ — 已完成
- [x] ~~FastMCP → FastAPI~~ — 已完成（anyio/asyncio 冲突）
- [x] ~~MLflow tracing + Prompt Registry + Evaluation + Span Tags~~ — 已完成
