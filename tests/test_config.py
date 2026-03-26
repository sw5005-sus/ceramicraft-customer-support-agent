"""Smoke tests for the Customer Support Agent."""

from ceramicraft_customer_support_agent.config import Settings, get_settings


def test_settings_defaults():
    """Settings should load with expected defaults."""
    settings = Settings()
    assert settings.MCP_SERVER_URL == "http://mcp-server-svc:8080/mcp"
    assert settings.OPENAI_MODEL == "gpt-4o"
    assert settings.LANGSMITH_PROJECT == "ceramicraft-cs-agent"


def test_get_settings_returns_settings_instance():
    """get_settings() should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
