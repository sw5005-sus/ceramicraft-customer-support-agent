# Customer Support Agent — 开发计划

_最后更新：2026-03-27_

---

## 设计决策

| 问题 | 决定 |
|------|------|
| 对外接口 | MCP Server（Streamable HTTP），其他 agent 通过 MCP 编排 |
| 对话状态 | 先 MemorySaver（内存），上线前切 PostgreSQL |
| 鉴权 | Agent 不验证 token，纯透传给下游 MCP Server（由 MCP Server 统一验证） |
| 敏感操作 | 支付、充值暂不实现；下单和确认收货需 Guard 确认 |
| Agent 架构 | LangGraph StateGraph：意图分类 → 条件路由 → 领域子图（ReAct）→ 安全守卫 |
| LLM | OpenAI GPT-4o |
| 设计原则 | Prompt Engineering + Context Engineering + 渐进式披露 |

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
        ├──→ Account Subgraph  (get/update_my_profile, addresses CRUD)
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

每个领域子图是独立的 `create_react_agent`，只绑定本领域相关工具，减少 token 消耗和幻觉。

### 工具发现

Tools 从 ceramicraft-mcp-server **动态发现**（首次请求时拉取 `list_tools`，结果缓存），转换为 LangChain Tool 格式注入 agent。MCP Server 新增 tool 时 agent 无需改代码。

### 对话隔离

通过 `thread_id` 隔离多用户对话。每个用户独立的对话历史和上下文，`thread_id` 来源于用户标识。共享的 `MemorySaver` 在进程内跨请求持久化。

### Token 透传链路

```
用户 JWT ──▶ Agent (MCP Server, 提取 token) ──▶ ceramicraft-mcp-server (验证 JWT) ──▶ 后端
```

Agent 不做 JWT 验证，只从 MCP Context 提取 Bearer token 并透传给下游。

### 渐进式披露

体现在 System Prompt 设计：首次交互简短问候 + 概要能力介绍，用户追问时按需展开，不主动列出所有功能。

---

## Phase 1 — 核心骨架 ✅

**目标：跑通 MCP Server + MCP Client + LangGraph agent 的完整链路**

### 1.1 Token 透传
- 从 MCP Context 提取 Bearer token
- 调下游 MCP Server 时原样带上（不做验证，不依赖 JWKS）

### 1.2 MCP Client（调下游）
- 连接 ceramicraft-mcp-server（Streamable HTTP）
- Token 透传：调用时带上从 ctx 提取的 Bearer token
- 工具发现：首次请求时从 MCP Server 拉取可用 tools 列表（结果缓存）

### 1.3 LangGraph Agent
- StateGraph：Classifier → Router → Domain Subgraphs → Guard
- 7 种意图分类：browse / cart / order / review / account / chitchat / escalate
- 5 个领域子图（ReAct agent），各自绑定领域工具
- 2 个轻量节点：chitchat（纯 LLM）、escalate（固定消息）
- Guard 节点：auth 检查 + 敏感操作确认
- MemorySaver 管理对话状态（模块级共享，跨请求持久）

### 1.4 MCP Server（对外暴露）
- 暴露 `chat` tool：接收用户消息，返回 agent 回复
- 暴露 `reset` tool：重置对话历史（API 兼容，待持久化后实现）
- 通过 FastMCP + Streamable HTTP 提供服务

### 1.5 Health + Serve
- `/health` 端点
- `serve.py` 入口启动

**交付物：** 能通过 MCP Inspector 对话，agent 能调下游 tools 查商品

---

## Phase 2 — 功能模块 ✅

> Intent-routed StateGraph, per-request MCP session with token forwarding, domain-specific prompts, guard node.
> 96 tests, 97% coverage.

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

### ~~2.6 支付~~ → 暂不实现
### ~~2.7 充值~~ → 暂不实现

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

### 3.3 CI/CD
- deploy.yml（复用现有模式）
- Helm chart + argocd-deploy 配置
- LangSmith tracing 集成

### 3.4 状态持久化
- 切换到 `langgraph-checkpoint-postgres`
- Vault 注入 PG 凭证

---

## 项目结构

```
ceramicraft-customer-support-agent/
├── serve.py                          # 入口
├── src/ceramicraft_customer_support_agent/
│   ├── __init__.py
│   ├── config.py                     # 配置（pydantic-settings）
│   ├── agent.py                      # 向后兼容接口（调 build_graph）
│   ├── classifier.py                 # 意图分类节点（Intent enum + Pydantic structured output）
│   ├── graph.py                      # StateGraph 主图（AgentState + build_graph + _wrap_subgraph）
│   ├── guard.py                      # 安全守卫节点（auth 检查 + 敏感操作确认）
│   ├── nodes.py                      # 轻量节点（chitchat + escalate，无工具）
│   ├── subgraphs.py                  # 领域子图（5 个 ReAct agent builder）
│   ├── mcp_client.py                 # MCP Client（连接 ceramicraft-mcp-server）
│   ├── mcp_server.py                 # MCP Server（对外暴露 chat / reset）
│   └── prompts.py                    # System prompt 模板（主 + 6 个领域）
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_classifier.py
│   ├── test_config.py
│   ├── test_graph.py
│   ├── test_guard.py
│   ├── test_mcp_client.py
│   ├── test_mcp_server.py
│   ├── test_nodes.py
│   ├── test_prompts.py
│   └── test_subgraphs.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 暂不实现（Backlog）

- [ ] top_up_account / get_pay_account（支付）— 需安全措施
- [ ] register_push_token（通知）— 优先级低
- [ ] PostgreSQL checkpoint 持久化
- [x] ~~子图拆分（按意图路由）~~ — 已完成
