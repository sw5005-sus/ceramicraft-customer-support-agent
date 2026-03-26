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


@cache
def get_settings() -> Settings:
    return Settings()
