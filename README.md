# ceramicraft-customer-support-agent

Customer Support AI Agent for the CeramiCraft e-commerce platform.

Connects to the [MCP Server](https://github.com/sw5005-sus/ceramicraft-mcp-server) to assist customers with product browsing, cart management, order tracking, reviews, and account operations.

## Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │   Customer Support Agent (LangGraph StateGraph)        │
  User / Frontend      │                                                         │
  ─── POST /chat ───▶  │  FastAPI REST                                          │
                       │    │                                                    │
                       │    ├─ extract Bearer token                              │
                       │    ├─ PersistentMCPClient (connection mode, per-call sessions) │
                       │    │   └─ tool handles bound to session ──────────────┼──▶ CeramiCraft MCP Server
                       │    ├─ discover tools (on startup, cached)              │         │
                       │    ├─ build graph agent (once, cached)                 │    HTTP (internal)
                       │    │   │                                               │         │
                       │    │   └─ User Message → Classifier → Router ────────┐│         ▼
                       │    │                         │                        ││   Backend Services
                       │    │                         ▼                        ││
                       │    │     ┌─────────┐  ┌─────────┐  ┌─────────┐       ││
                       │    │     │ Browse  │  │  Cart   │  │ Order   │       ││
                       │    │     │ Subgraph│  │Subgraph │  │Subgraph │       ││
                       │    │     └─────────┘  └─────────┘  └─────────┘       ││
                       │    │            │           │           │            ││
                       │    │     ┌─────────┐  ┌─────────┐  ┌─────────┐       ││
                       │    │     │ Review  │  │Account  │  │Chitchat │       ││
                       │    │     │Subgraph │  │Subgraph │  │  Node   │       ││
                       │    │     └─────────┘  └─────────┘  └─────────┘       ││
                       │    │            │           │           │            ││
                       │    │            └───────────┼───────────┘            ││
                       │    │                        ▼                        ││
                       │    │                  ┌─────────┐                    ││
                       │    │                  │  Guard  │                    ││
                       │    │                  │  Node   │                    ││
                       │    │                  └─────────┘                    ││
                       │    │                        │                        ││
                       │    └─ invoke (AsyncPostgresSaver checkpointer) ─────────────────││
                       │                             ▼                        │
                       │                       Response                       │
                       └────────────────────────────────────────────────────────┘
```

### Graph Flow

1. **User Message** enters via `POST /chat` with optional Bearer token and optional thread_id
2. **Classifier** analyzes intent using LLM (no tools): `browse`, `cart`, `order`, `review`, `account`, `chitchat`, `escalate`
3. **Router** sends to appropriate domain subgraph based on intent
4. **Domain Subgraphs** (stateless ReAct agents with filtered tools):
   - **Browse**: search_products, get_product, list_product_reviews
   - **Cart**: get_cart, add_to_cart, update_cart_item, remove_cart_item, estimate_cart_price, search_products
   - **Order**: list_my_orders, get_order_detail, confirm_receipt, get_order_stats, create_order
   - **Review**: create_review, like_review, get_user_reviews, list_product_reviews
   - **Account**: get_my_profile, update_my_profile, list_my_addresses, create_address, update_address, delete_address
5. **Guard** post-processes responses to check for:
   - Auth requirements (prompts for login if needed)
   - Sensitive operations (requests confirmation for deletions, etc.)

### State Management

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    intent: str                              # Classified user intent
    auth_token: str | None                   # Bearer token from request
    needs_confirm: bool                      # Requires user confirmation
    confirmed: bool                          # User has confirmed action
```

Conversation history is persisted across requests via a shared checkpointer keyed by `thread_id`.
Uses **AsyncPostgresSaver** (via `langgraph-checkpoint-postgres`), backed by the shared
`ceramicraft-postgres` container used by log-ms and notification-ms.
`POSTGRES_HOST` is required — the agent will fail to start if not configured.

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message. Body: `{"message": "...", "thread_id": "..."}`. `thread_id` is optional — omit to start a new conversation; the response always includes `thread_id` to continue. |
| `/reset` | POST | Reset conversation. Query: `?thread_id=...` |
| `/cs-agent/v1/ping` | GET | Readiness probe — returns 503 until agent is initialised |
| `/docs` | GET | Swagger UI (auto-generated) |

## Available Tools (via MCP)

| Category | Tools | Auth Level |
|----------|-------|------------|
| Product | search_products, get_product | PUBLIC |
| Cart | get_cart, add_to_cart, update_cart_item, remove_cart_item, estimate_cart_price | USER |
| Order | list_my_orders, get_order_detail, confirm_receipt, get_order_stats, create_order | USER |
| Review | list_product_reviews, get_user_reviews, create_review, like_review | PUBLIC / USER |
| User | get_my_profile, update_my_profile, list_my_addresses, create_address, update_address, delete_address | USER |

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
uv run ty check

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
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL | *(optional)* |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `ceramicraft-cs-agent` |
| `POSTGRES_USER` | PostgreSQL user — shared with log-ms / notification-ms | *(required)* |
| `POSTGRES_PASSWORD` | PostgreSQL password | *(required)* |
| `POSTGRES_HOST` | PostgreSQL host (e.g. `postgres` inside docker network) | *(required)* |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `CS_AGENT_DB_NAME` | Database name for conversation checkpoints | `cs_agent_db` |
| `AGENT_MAX_HISTORY` | Max messages passed to subgraphs (older trimmed to prevent token explosion) | `20` |
