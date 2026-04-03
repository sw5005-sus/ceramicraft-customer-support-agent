"""Configuration for the Customer Support Agent."""

from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # MCP Server (downstream)
    MCP_SERVER_URL: str = "http://mcp-server-svc:8080/mcp"

    # LLM (OpenAI)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # Agent Server
    AGENT_HOST: str = "0.0.0.0"
    AGENT_PORT: int = 8080

    # LangSmith (optional)
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "ceramicraft-cs-agent"

    # MLflow (optional)
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_EXPERIMENT_NAME: str = "ceramicraft-cs-agent"

    # PostgreSQL — individual variables consistent with log-ms / notification-ms
    # These are combined into a connection URL at runtime.
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = ""
    POSTGRES_PORT: int = 5432
    CS_AGENT_DB_NAME: str = "cs_agent_db"

    # Optional: explicit connection URL that overrides the individual variables above.
    # Format: postgresql+psycopg://user:password@host:port/dbname
    # Leave empty to build from POSTGRES_USER/PASSWORD/HOST/PORT, or to use
    # in-memory MemorySaver (when all postgres vars are unset).
    POSTGRES_URL: str = ""

    # Max messages to pass to subgraphs (older messages are trimmed)
    AGENT_MAX_HISTORY: int = 20

    @property
    def postgres_url(self) -> str:
        """Resolve the effective PostgreSQL connection URL.

        Priority:
          1. POSTGRES_URL (explicit override)
          2. Assembled from POSTGRES_USER / PASSWORD / HOST / PORT → cs_agent_db
          3. Empty string → fall back to MemorySaver
        """
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        if self.POSTGRES_USER and self.POSTGRES_HOST:
            return (
                f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.CS_AGENT_DB_NAME}"
            )
        return ""


@cache
def get_settings() -> Settings:
    return Settings()
