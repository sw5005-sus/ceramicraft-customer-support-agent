# Customer Support Agent — 开发计划

_最后更新：2026-03-26_

---

## 设计决策

| 问题 | 决定 |
|------|------|
| 对外接口 | MCP Server（Streamable HTTP），其他 agent 通过 MCP 编排 |
| 对话状态 | 先 MemorySaver（内存），上线前切 PostgreSQL |
| 鉴权 | Agent 不验证 token，纯透传给下游 MCP Server（由 MCP Server 统一验证） |
| 敏感操作 | 支付、充值、下单暂不实现 |
| Agent 架构 | 单 ReAct agent + tools（LangGraph），按需拆子图 |
| LLM | OpenAI GPT-4o |
| 设计原则 | Prompt Engineering + Context Engineering + 渐进式披露 |

---

## 架构设计

### ReAct Agent 状态机

```
        ┌──────────────┐
        │   LLM Node   │ ◀─── System Prompt + 对话历史 + Tool 结果
        └──────┬───────┘
               │
        需要调 tool?
        ┌──────┴──────┐
        │ Yes         │ No
        ▼             ▼
  ┌───────────┐   ┌──────────┐
  │ Tool Node │   │ 返回回复  │
  └─────┬─────┘   └──────────┘
        │
        │ tool 结果
        └──▶ 回到 LLM Node
```

LangGraph `create_react_agent` 实现 Reason → Act → Observe 循环。LLM 自主决定调哪个 tool、调几次、何时回复用户。

### 工具发现

Tools 从 ceramicraft-mcp-server **动态发现**（启动时拉取 `list_tools`），转换为 LangChain Tool 格式注入 agent。MCP Server 新增 tool 时 agent 无需改代码。

### 对话隔离

通过 `thread_id` 隔离多用户对话。每个用户独立的对话历史和上下文，`thread_id` 来源于用户标识。

### Token 透传链路

```
用户 JWT ──▶ Agent (MCP Server, 提取 token) ──▶ ceramicraft-mcp-server (验证 JWT) ──▶ 后端
```

Agent 不做 JWT 验证，只从 MCP Context 提取 Bearer token 并透传给下游。

### 渐进式披露

体现在 System Prompt 设计：首次交互简短问候 + 概要能力介绍，用户追问时按需展开，不主动列出所有功能。

---

## Phase 1 — 核心骨架

**目标：跑通 MCP Server + MCP Client + LangGraph agent 的完整链路**

### 1.1 Token 透传
- 从 MCP Context 提取 Bearer token
- 调下游 MCP Server 时原样带上（不做验证，不依赖 JWKS）

### 1.2 MCP Client（调下游）
- 连接 ceramicraft-mcp-server（Streamable HTTP）
- Token 透传：调用时带上从 ctx 提取的 Bearer token
- 工具发现：启动时从 MCP Server 拉取可用 tools 列表

### 1.3 LangGraph Agent
- 单 ReAct agent（`create_react_agent`）
- 绑定从 MCP Server 发现的 tools
- MemorySaver 管理对话状态
- System prompt：
  - 角色定义（CeramiCraft 客服）
  - 能力边界（渐进式披露：先介绍能做什么，用户追问再展开）
  - 安全约束（不泄露系统信息、不执行敏感操作）

### 1.4 MCP Server（对外暴露）
- 暴露 `chat` tool：接收用户消息，返回 agent 回复
- 暴露 `reset_conversation` tool：重置对话历史
- 通过 FastMCP + Streamable HTTP 提供服务

### 1.5 Health + Serve
- `/health` 端点
- `serve.py` 入口启动

**交付物：** 能通过 MCP Inspector 对话，agent 能调下游 tools 查商品

---

## Phase 2 — 功能模块（渐进实现）

### 2.1 商品咨询（PUBLIC，无需 token）
- search_products：搜索商品
- get_product：查看详情
- list_product_reviews：查看评价
- Agent 能力：理解模糊查询、对比推荐、总结评价

### 2.2 购物车管理（USER）
- get_cart, add_to_cart, update_cart_item, remove_cart_item
- estimate_cart_price
- Agent 能力：引导加购、确认操作、展示价格

### 2.3 订单查询（USER，只读）
- list_my_orders, get_order_detail
- ~~create_order, confirm_receipt~~ → 暂不实现
- Agent 能力：查询订单状态、解释物流信息

### 2.4 评价互动（USER）
- create_review, get_user_reviews, like_review
- Agent 能力：引导写评价、展示历史评价

### 2.5 账户管理（USER）
- get_my_profile, update_my_profile
- list/create/update/delete_address
- Agent 能力：查看和修改个人信息、地址管理

### ~~2.6 支付~~ → 暂不实现
### ~~2.7 下单~~ → 暂不实现

---

## Phase 3 — 质量与上线

### 3.1 Prompt 优化
- Context Engineering：根据对话阶段动态注入上下文
- 渐进式披露：首次交互简洁介绍，按需展开
- Few-shot examples 提升工具调用准确率

### 3.2 测试
- 单元测试：auth、config、tool 注册
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

## 项目结构（目标）

```
ceramicraft-customer-support-agent/
├── serve.py                          # 入口
├── src/ceramicraft_customer_support_agent/
│   ├── __init__.py
│   ├── config.py                     # 配置
│   ├── agent.py                      # LangGraph ReAct agent 定义
│   ├── mcp_client.py                 # MCP Client（连接 ceramicraft-mcp-server）
│   ├── mcp_server.py                 # MCP Server（对外暴露 chat 等 tool）
│   └── prompts.py                    # System prompt 模板
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 暂不实现（Backlog）

- [ ] create_order（下单）— 需人工确认机制
- [ ] confirm_receipt（确认收货）— 需人工确认机制
- [ ] top_up_account / get_pay_account（支付）— 需安全措施
- [ ] register_push_token（通知）— 优先级低
- [ ] PostgreSQL checkpoint 持久化
- [ ] 子图拆分（按意图路由）
