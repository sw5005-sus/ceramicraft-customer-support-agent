# ceramicraft-customer-support-agent

Customer Support AI Agent for the CeramiCraft e-commerce platform.

Connects to the [MCP Server](https://github.com/sw5005-sus/ceramicraft-mcp-server) to assist customers with product browsing, cart management, order tracking, reviews, and account operations.

## Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │   Customer Support Agent (LangGraph StateGraph)        │
  User / Orchestrator  │                                                         │
  ───MCP (chat)──────▶ │  FastMCP Server                                        │
                       │    │                                                    │
                       │    ├─ extract Bearer token                              │
                       │    ├─ MCP Client (per-request)                         │
                       │    │   └─ forward token ──────────────────────────────┼──▶ CeramiCraft MCP Server
                       │    ├─ discover tools                                   │         │
                       │    ├─ build graph agent                                │    HTTP (internal)
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
                       │    └─ invoke (shared memory)│                        ││
                       │                             ▼                        │
                       │                       Response                       │
                       └────────────────────────────────────────────────────────┘
```

### Graph Flow

1. **User Message** enters the system via MCP chat tool
2. **Classifier** analyzes intent using LLM (no tools): `browse`, `cart`, `order`, `review`, `account`, `chitchat`, `escalate`
3. **Router** sends to appropriate domain subgraph based on intent
4. **Domain Subgraphs** (ReAct agents with filtered tools):
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

Each request creates a fresh MCP session with the user's auth token,
discovers available tools, and invokes the LangGraph StateGraph. Conversation
history is preserved across requests via a shared `MemorySaver`.

## Available Tools (via MCP)

| Category | Tools | Auth Level |
|----------|-------|------------|
| Product | search_products, get_product | PUBLIC |
| Cart | get_cart, add_to_cart, update_cart_item, remove_cart_item, estimate_cart_price | USER |
| Order | list_my_orders, get_order_detail, confirm_receipt, get_order_stats, create_order | USER |
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