"""Register all CeramiCraft CS Agent prompts to MLflow Prompt Registry.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run scripts/prompt_register.py

Requires MLFLOW_TRACKING_URI to be set.
"""

import os
import sys

import mlflow

from ceramicraft_customer_support_agent.prompts import (
    ACCOUNT_PROMPT,
    BROWSE_PROMPT,
    CART_PROMPT,
    CHITCHAT_PROMPT,
    ORDER_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
)

PROMPTS = [
    ("CS_AGENT_SYSTEM_PROMPT", SYSTEM_PROMPT, "Main system prompt for the CS agent"),
    ("CS_AGENT_BROWSE_PROMPT", BROWSE_PROMPT, "Product browsing specialist prompt"),
    ("CS_AGENT_CART_PROMPT", CART_PROMPT, "Shopping cart specialist prompt"),
    ("CS_AGENT_ORDER_PROMPT", ORDER_PROMPT, "Order management specialist prompt"),
    ("CS_AGENT_REVIEW_PROMPT", REVIEW_PROMPT, "Review specialist prompt"),
    ("CS_AGENT_ACCOUNT_PROMPT", ACCOUNT_PROMPT, "Account management specialist prompt"),
    ("CS_AGENT_CHITCHAT_PROMPT", CHITCHAT_PROMPT, "Casual conversation prompt"),
]


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print(
            "ERROR: MLFLOW_TRACKING_URI environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"Connected to MLflow at: {tracking_uri}")

    for name, template, commit_message in PROMPTS:
        print(f"\nRegistering prompt: {name}")
        pv = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
        )
        print(f"  Registered version: {pv.version}")

        mlflow.genai.set_prompt_alias(
            name=name,
            alias="production",
            version=pv.version,
        )
        print(f"  Alias 'production' -> version {pv.version}")

    print("\nAll prompts registered successfully.")


if __name__ == "__main__":
    main()
