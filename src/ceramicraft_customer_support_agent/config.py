"""Configuration for the Customer Support Agent."""

from functools import cache

from pydantic import Field
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

    # PostgreSQL — consistent with log-ms / notification-ms
    POSTGRES_USER: str = Field(default="")
    POSTGRES_PASSWORD: str = Field(default="")
    POSTGRES_HOST: str = Field(default="")
    POSTGRES_PORT: int = Field(default=5432)
    CS_AGENT_DB_NAME: str = "cs_agent_db"

    # Max messages to pass to subgraphs (older messages are trimmed)
    AGENT_MAX_HISTORY: int = 20

    @property
    def DATABASE_URL(self) -> str:
        """Assemble PostgreSQL connection URL from individual vars.

        Returns an empty string when POSTGRES_HOST is not configured,
        which causes build_checkpointer() to fall back to MemorySaver.
        """
        if not self.POSTGRES_HOST:
            return ""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.CS_AGENT_DB_NAME}"
        )


@cache
def get_settings() -> Settings:
    return Settings()
