"""Tests for configuration module."""

from ceramicraft_customer_support_agent.config import Settings, get_settings


def test_settings_defaults():
    """Settings should load with expected defaults (ignoring .env overrides)."""
    settings = Settings(_env_file=None)  # ty: ignore[unknown-argument]
    assert settings.CS_AGENT_MCP_SERVER_URL == "http://mcp-server-svc:8080/mcp"
    assert settings.CS_AGENT_OPENAI_MODEL == "gpt-4o"
    assert settings.CS_AGENT_HTTP_HOST == "0.0.0.0"
    assert settings.CS_AGENT_HTTP_PORT == 8080
    assert settings.CS_AGENT_GRPC_HOST == "[::]"
    assert settings.CS_AGENT_GRPC_PORT == 50051
    assert settings.CS_AGENT_LANGSMITH_PROJECT == "ceramicraft-cs-agent"


def test_get_settings_returns_settings_instance():
    """get_settings() should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_get_settings_is_cached():
    """get_settings() should return the same object on repeated calls."""
    assert get_settings() is get_settings()
