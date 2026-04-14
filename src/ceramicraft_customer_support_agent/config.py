"""Configuration for the Customer Support Agent."""

from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # MCP Server (downstream)
    CS_AGENT_MCP_SERVER_URL: str = "http://mcp-server-svc:8080/mcp"

    # LLM (OpenAI)
    CS_AGENT_OPENAI_API_KEY: str = ""
    CS_AGENT_OPENAI_MODEL: str = "gpt-4o"

    # Agent Server — HTTP
    CS_AGENT_HTTP_HOST: str = "0.0.0.0"
    CS_AGENT_HTTP_PORT: int = 8080

    # Agent Server — gRPC
    CS_AGENT_GRPC_HOST: str = "[::]"
    CS_AGENT_GRPC_PORT: int = 50051

    # LangSmith
    CS_AGENT_LANGSMITH_API_KEY: str = ""
    CS_AGENT_LANGSMITH_PROJECT: str = "customer-support-agent"

    # MLflow (optional)
    MLFLOW_TRACKING_URI: str = ""
    CS_AGENT_MLFLOW_EXPERIMENT_NAME: str = "customer-support-agent"

    # PostgreSQL — consistent with log-ms / notification-ms
    POSTGRES_USER: str = Field(default="")
    POSTGRES_PASSWORD: str = Field(default="")
    POSTGRES_HOST: str = Field(default="")
    POSTGRES_PORT: int = Field(default=5432)
    CS_AGENT_DB_NAME: str = "cs_agent_db"

    # Max messages to pass to subgraphs (older messages are trimmed)
    CS_AGENT_MAX_HISTORY: int = 20

    @property
    def DATABASE_URL(self) -> str:
        """Assemble PostgreSQL connection URL from individual vars.

        Returns an empty string when POSTGRES_HOST is not configured,
        which causes build_checkpointer() to raise RuntimeError.
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
