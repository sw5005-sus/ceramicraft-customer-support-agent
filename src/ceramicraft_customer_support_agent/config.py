"""Configuration for the Customer Support Agent."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # MCP Server
    MCP_SERVER_URL: str = "http://mcp-server-svc:8080/mcp"

    # LLM
    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""

    # LangSmith (optional)
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "ceramicraft-cs-agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()
