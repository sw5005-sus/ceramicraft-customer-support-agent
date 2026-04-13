# ceramicraft-customer-support-agent

Customer Support AI Agent for the CeramiCraft e-commerce platform.

Connects to the [MCP Server](https://github.com/sw5005-sus/ceramicraft-mcp-server) to assist customers with product browsing, cart management, order tracking, reviews, and account operations.

## Architecture

```
  User ── POST /chat ──▶ FastAPI ──▶ LangGraph StateGraph
       ── gRPC Chat  ──▶ gRPC   ──┘        │
                                     Classifier (LLM)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                       Browse          Cart/Order      Account/Review
                      Subgraph         Subgraphs        Subgraphs
                          │               │               │
                          └───────┬───────┘───────────────┘
                                  ▼
                          Guard (auth + confirm)
                                  │
                                  ▼
                        MCP Server ──▶ Backend Services
```

**Key components:**

- **Classifier** — LLM-based intent detection: `browse`, `cart`, `order`, `review`, `account`, `chitchat`, `escalate`
- **Domain Subgraphs** — Stateless ReAct agents, each with filtered MCP tools
- **Guard** — Post-processing: auth checks + sensitive operation confirmation (`create_order`, `delete_address`, `confirm_receipt`)
- **PersistentMCPClient** — Singleton; tool list cached at startup. Adding tools on MCP server requires no agent changes
- **AsyncPostgresSaver** — Conversation checkpointer keyed by `thread_id`

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message. Body: `{"message": "...", "thread_id": "..."}`. Omit `thread_id` for new conversation. |
| `/reset` | POST | Reset conversation. Query: `?thread_id=...` |
| `/cs-agent/v1/ping` | GET | Readiness probe (503 until ready) |
| `/docs` | GET | Swagger UI |

## gRPC API

Service `CustomerSupportAgent` on port `50051` (configurable via `CS_AGENT_GRPC_PORT`).

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `Chat` | `ChatRequest(message, thread_id?)` | `ChatResponse(reply, thread_id)` | Same as POST /chat. Auth via metadata key `authorization` (Bearer scheme). |
| `Reset` | `ResetRequest(thread_id)` | `ResetResponse(status, message)` | Same as POST /reset |

Proto definition: [`protos/cs_agent.proto`](protos/cs_agent.proto)

## MCP Tools

| Category | Tools | Auth |
|----------|-------|------|
| Product | `search_products`, `get_product` | Public |
| Cart | `get_cart`, `add_to_cart`, `update_cart_item`, `remove_cart_item`, `estimate_cart_price` | User |
| Order | `list_my_orders`, `get_order_detail`, `confirm_receipt`, `get_order_stats`, `create_order` | User |
| Review | `list_product_reviews`, `get_user_reviews`, `create_review`, `like_review` | Mixed |
| Account | `get_my_profile`, `update_my_profile`, `list_my_addresses`, `create_address`, `update_address`, `delete_address`, `get_pay_account`, `top_up_account` | User |

## Development

```bash
uv sync                    # Install dependencies
uv run python serve.py     # Run server
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run ty check            # Type check
uv run pytest --cov=src/ceramicraft_customer_support_agent --cov-report=term-missing  # Test
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CS_AGENT_MCP_SERVER_URL` | MCP Server endpoint | `http://mcp-server:8080/mcp` |
| `CS_AGENT_OPENAI_API_KEY` | OpenAI API key | *(required)* |
| `CS_AGENT_OPENAI_MODEL` | Model name | `gpt-4o` |
| `CS_AGENT_HTTP_HOST` | HTTP bind address | `0.0.0.0` |
| `CS_AGENT_HTTP_PORT` | HTTP server port | `8080` |
| `CS_AGENT_GRPC_HOST` | gRPC bind address | `[::]` |
| `CS_AGENT_GRPC_PORT` | gRPC server port | `50051` |
| `POSTGRES_USER` | PostgreSQL user | *(required)* |
| `POSTGRES_PASSWORD` | PostgreSQL password | *(required)* |
| `POSTGRES_HOST` | PostgreSQL host | *(required)* |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `CS_AGENT_DB_NAME` | Checkpoint database | `cs_agent_db` |
| `CS_AGENT_MAX_HISTORY` | Max messages to subgraphs | `20` |
| `CS_AGENT_LANGSMITH_API_KEY` | LangSmith API key (forwarded to `LANGCHAIN_API_KEY`) | *(optional)* |
| `CS_AGENT_LANGSMITH_PROJECT` | LangSmith project name | `ceramicraft-cs-agent` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | *(optional)* |
| `CS_AGENT_MLFLOW_EXPERIMENT_NAME` | MLflow experiment | `ceramicraft-cs-agent` |

## Documentation

- [Local Demo Guide](docs/local-demo-guide.md) — Step-by-step local-stack setup and demo walkthrough
- [Development Plan](docs/DEVELOPMENT_PLAN.md) — Roadmap and design decisions
