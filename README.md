# ceramicraft-customer-support-agent

Customer Support AI Agent for the CeramiCraft e-commerce platform.

Connects to the [MCP Server](https://github.com/sw5005-sus/ceramicraft-mcp-server) to assist customers with product browsing, cart management, order tracking, reviews, and account operations.

## Architecture

```
                       ┌──────────────────────────────────┐
                       │   Customer Support Agent          │
  User / Orchestrator  │                                   │
  ───MCP (chat)──────▶ │  FastMCP Server                   │
                       │    │                              │
                       │    ├─ extract Bearer token        │
                       │    ├─ MCP Client (per-request)    │
                       │    │   └─ forward token ──────────┼──▶ CeramiCraft MCP Server
                       │    ├─ discover tools              │         │
                       │    ├─ build ReAct agent            │    HTTP (internal)
                       │    └─ invoke (shared memory)       │         │
                       │                                   │         ▼
                       └──────────────────────────────────┘   Backend Services
```

Each `chat` call creates a fresh MCP session with the user's auth token,
discovers available tools, and invokes the LangGraph agent. Conversation
history is preserved across requests via a shared `MemorySaver`.

## Available Tools (via MCP)

| Category | Tools | Auth Level |
|----------|-------|------------|
| Product | search_products, get_product | PUBLIC |
| Cart | get_cart, add_to_cart, update_cart_item, remove_cart_item, estimate_cart_price | USER |
| Order | list_my_orders, get_order_detail | USER |
| Review | list_product_reviews, get_user_reviews, create_review, like_review | PUBLIC / USER |
| User | get_my_profile, update_my_profile, list_my_addresses, create_address, update_address, delete_address | USER |

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `chat` | Send a message to the customer support agent |
| `reset` | Reset conversation history for a thread |

## Development

```bash
# Install
uv sync

# Run
uv run python serve.py

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check src/

# Test
uv run pytest --cov=src/ceramicraft_customer_support_agent --cov-report=term-missing
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_SERVER_URL` | Downstream MCP Server endpoint | `http://mcp-server-svc:8080/mcp` |
| `OPENAI_API_KEY` | OpenAI API key | *(required)* |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` |
| `AGENT_HOST` | Agent server bind address | `0.0.0.0` |
| `AGENT_PORT` | Agent server port | `8080` |
| `LANGSMITH_API_KEY` | LangSmith tracing key | *(optional)* |
| `LANGSMITH_PROJECT` | LangSmith project name | `ceramicraft-cs-agent` |
