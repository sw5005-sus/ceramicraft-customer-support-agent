"""Entrypoint for the Customer Support Agent."""

import logging

from ceramicraft_customer_support_agent.mcp_server import create_mcp_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize and start the Customer Support Agent."""
    logger.info("Starting Customer Support Agent...")
    mcp = create_mcp_server()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
