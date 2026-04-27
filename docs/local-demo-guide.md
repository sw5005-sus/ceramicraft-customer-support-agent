# Local Demo Guide

Local-stack setup and end-to-end demo for all intent branches.

## 1. Start Local-Stack

```powershell
cd ceramicraft-deploy/local-stack
Copy-Item .env.example .env   # first time only; fill in secrets
docker compose up -d
```

Key ports: user-ms `8083`, product-ms `8081`, order-ms `8082`, comment-ms `8084`, payment-ms `8085`, mcp-server `8088`, postgres `5432`, mlflow `5000`.

## 2. Start CS Agent (Docker)

Create `.env` in `ceramicraft-customer-support-agent/`:

```env
CS_AGENT_MCP_SERVER_URL=http://mcp-server:8080/mcp
CS_AGENT_OPENAI_API_KEY=sk-proj-xxxxx
CS_AGENT_OPENAI_MODEL=gpt-4o
POSTGRES_USER=ceramicraft
POSTGRES_PASSWORD=ceramicraft123
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
CS_AGENT_DB_NAME=cs_agent_db
MLFLOW_TRACKING_URI=http://mlflow:5000
```

> ⚠️ Use Docker service names (`postgres`, `mcp-server`, `mlflow`), not `localhost`.

```powershell
docker compose up -d --build
curl http://localhost:8080/cs-agent/v1/ping  # expect {"status": "ok"}
```

## 3. Get Token (Customer PKCE)

Auth uses cloud Zitadel (Client ID: `361761429302373082`).

```powershell
uv run python scripts/get_token.py          # auto mode (needs redirect URI configured)
uv run python scripts/get_token.py --manual  # manual mode (paste redirect URL)
```

The script handles: PKCE login → token exchange → user registration (`oauth-callback`) → token refresh (to get `local_userid` in JWT).

```powershell
$env:TOKEN = "<output id_token>"
```

> First run completes full registration. Subsequent runs only need re-login for a fresh token.

## 4. Send Messages

### Interactive CLI (Recommended)

```powershell
uv run python scripts/chat.py
```

Enter token → optional thread_id → chat. `Ctrl+C` to exit.

### cURL

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# New conversation (no thread_id)
curl -s -X POST http://localhost:8080/chat `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:TOKEN" `
  -d '{"message": "hello"}' | uv run python -m json.tool

# Continue (use returned thread_id)
$env:THREAD = "<returned thread_id>"
curl -s -X POST http://localhost:8080/chat `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:TOKEN" `
  -d "{`"message`": `"show me some bowls`", `"thread_id`": `"$env:THREAD`"}" | uv run python -m json.tool
```

### SSE Streaming

Use `/chat/stream` for real-time progress events:

```powershell
curl -N -X POST http://localhost:8080/chat/stream `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:TOKEN" `
  -d '{"message": "hello"}'
```

Events arrive as: `guarding` → `classifying` → `processing` → `reply` → `done`.

## 5. Demo Scenarios

| # | Intent | Example Message | Key Behavior |
|---|--------|----------------|--------------|
| 1 | **Chitchat** | "Hi! What is CeramiCraft?" | No tool calls |
| 2 | **Browse** | "Show me ceramic bowls" → "Tell me about product 1" | `search_products` → `get_product` |
| 3 | **Cart** (no token) | "Add product 1 to cart" | Guard: login prompt |
| 4 | **Cart** (with token) | "Add product 1 to cart" → "What's in my cart?" | `add_to_cart` → `get_cart` |
| 5 | **Order** | "Place an order with cart items using address 1" | Domain prompt requires explicit confirmation before `create_order` |
| 6 | **Review** | "I want to review Porcelain Teapot" → "5 stars, great quality!" | `search_products` → `create_review` |
| 7 | **Account** | "Show my addresses" → "Delete address 1" | Domain prompt requires explicit confirmation before `delete_address` |
| 8 | **Escalate** | "I need to speak with a human" | Escalation message |

> Use different `thread_id` per scenario to isolate context.

## 6. Traces & Cleanup

**MLflow:** `http://localhost:5000` — prompt registry and best-effort LangChain traces for intent classification, tool calls, and latency.

**Reset conversation:** `curl -s -X POST "http://localhost:8080/reset?thread_id=<id>"`

**Stop services:**

```powershell
cd ceramicraft-customer-support-agent && docker compose down
cd ceramicraft-deploy/local-stack && docker compose down      # add -v to wipe data
```
