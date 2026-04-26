"""Safety guard node for post-processing responses.

Provides two layers of protection:
1. Auth-error detection — prompts unauthenticated users to log in.
2. Prompt-injection / jailbreak detection — screens user inputs for
   adversarial patterns and blocks them before they reach the LLM.
"""

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth-error detection
# ---------------------------------------------------------------------------

# Pre-compiled regex for detecting auth-related error messages.
# Uses word boundaries to avoid false positives on "author", "authority", etc.
_AUTH_ERROR_PATTERN = re.compile(
    r"\b(authentication|authorization|log[\s-]?in|unauthorized|unauthenticated)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak detection
# ---------------------------------------------------------------------------

# Each pattern targets a well-known prompt-injection technique.
# Order: most specific → most general to minimize false positives.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # 1. Direct "ignore previous instructions" family
    re.compile(
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|system)\s+"
        r"(?:system\s+)?(?:instructions?|prompts?|rules?|guidelines?|constraints?)",
        re.IGNORECASE,
    ),
    # 2. "You are now …" role-override attempts
    #    Negative lookahead excludes benign continuations.
    re.compile(
        r"(?<![#])\byou\s+are\s+now\s+(?:a\s+)?(?!helping|assisting|looking)"
        r"[a-zA-Z]",
        re.IGNORECASE,
    ),
    # 3. "Pretend you are …" / "Act as …" role hijacking
    re.compile(
        r"(?:pretend|act|behave|roleplay|role[\s-]?play)\s+"
        r"(?:to\s+be\s+|as\s+(?:if\s+you\s+(?:are|were)\s+)?|like\s+)"
        r"(?:a\s+)?(?:DAN|jailbr(?:oken|eak)|unrestrict|evil|hack)",
        re.IGNORECASE,
    ),
    # 4. System-prompt extraction attempts
    #    Requires "system" or "your" before the target noun to avoid
    #    false positives like "show me the instructions for this product".
    re.compile(
        r"(?:show|reveal|print|output|display|repeat|echo|leak|dump)"
        r"\s+(?:\w+\s+){0,2}(?:(?:your|the|my)\s+)?system\s+"
        r"(?:prompt|instructions?|rules?|config)",
        re.IGNORECASE,
    ),
    # 4b. "reveal/dump your instructions/prompt" (no "system" needed
    #     when the verb is highly suspicious)
    re.compile(
        r"(?:reveal|leak|dump|echo)\s+(?:\w+\s+){0,2}"
        r"(?:prompt|instructions?|rules?|config)",
        re.IGNORECASE,
    ),
    # 5. Markdown / XML injection to close system context
    re.compile(
        r"```\s*(?:system|end|</)",
        re.IGNORECASE,
    ),
    # 6. "Disregard" / "forget" instructions
    re.compile(
        r"(?:disregard|forget|override|bypass|skip|drop)\s+"
        r"(?:all\s+)?(?:previous|prior|above|system|safety|your)\s+"
        r"(?:system\s+)?(?:instructions?|prompts?|rules?|guidelines?|constraints?|policies)",
        re.IGNORECASE,
    ),
    # 7. Token-smuggling separators ("###", "[INST]", etc.)
    re.compile(
        r"(?:\[/?INST\]|\[/?SYS\]|<\|(?:im_start|im_end|system)\|>|###\s*(?:System|Human|Assistant):)",
        re.IGNORECASE,
    ),
]

# Blocked response when injection is detected.
_INJECTION_BLOCKED_RESPONSE = (
    "I'm sorry, but I can't process that request. "
    "If you have a question about our products or need help with your "
    "account, I'm happy to assist!"
)


def detect_prompt_injection(text: str) -> str | None:
    """Check *text* for prompt-injection patterns.

    Returns the name of the first matching pattern (for logging) or ``None``
    if the input appears safe.
    """
    _pattern_names = [
        "ignore_instructions",
        "role_override",
        "role_hijack",
        "prompt_extraction",
        "prompt_extraction",
        "context_escape",
        "disregard_instructions",
        "token_smuggling",
    ]
    for pattern, name in zip(_INJECTION_PATTERNS, _pattern_names, strict=True):
        if pattern.search(text):
            return name
    return None


# ---------------------------------------------------------------------------
# Guard node builder
# ---------------------------------------------------------------------------


def build_guard() -> Callable:
    """Build the safety guard node.

    Checks for auth requirements and adds intervention messages when needed.

    Confirmation for sensitive operations (create_order, delete_address,
    confirm_receipt) is handled by the LLM itself via domain-specific
    prompt instructions, NOT by the guard.  This avoids the post-execution
    confirmation anti-pattern where tools have already run before the
    guard can intervene.

    Returns:
        A callable that takes AgentState and returns updated state.
    """

    def guard_node(state: dict) -> dict:
        """Apply safety checks and interventions."""
        auth_token = state.get("auth_token")
        messages = state.get("messages", [])

        # Check last messages for auth error patterns
        recent_messages = messages[-5:] if messages else []

        needs_auth_prompt = False

        for msg in recent_messages:
            msg_content = str(msg.content) if hasattr(msg, "content") else str(msg)

            if _AUTH_ERROR_PATTERN.search(msg_content):
                if not auth_token:
                    needs_auth_prompt = True

        updates: dict = {}

        if needs_auth_prompt and not auth_token:
            auth_message = (
                "To access your account information, cart, orders, or reviews, "
                "you'll need to log in first. Please visit our website to sign in, "
                "then come back and I'll be happy to help!"
            )
            updates["messages"] = [{"role": "assistant", "content": auth_message}]
            logger.info("Added auth requirement message")

        return updates

    return guard_node


def build_input_guard() -> Callable:
    """Build a pre-LLM input guard that screens user messages.

    This guard should be called *before* the message reaches any LLM node.
    If a prompt-injection attempt is detected the guard returns a safe
    canned response and sets ``blocked=True`` so downstream nodes can
    short-circuit.

    Returns:
        A callable ``(state) -> dict`` with optional ``blocked`` and
        ``messages`` keys.
    """

    def input_guard_node(state: dict) -> dict:
        """Screen the latest user message for adversarial patterns."""
        messages = state.get("messages", [])
        if not messages:
            return {}

        # Find latest user message
        user_text: str | None = None
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                user_text = str(msg.content) if hasattr(msg, "content") else str(msg)
                break
            # Also handle plain dicts from tests
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = str(msg.get("content", ""))
                break

        if user_text is None:
            return {}

        matched = detect_prompt_injection(user_text)
        if matched is not None:
            logger.warning(
                "Prompt-injection attempt detected (pattern=%s), blocking.",
                matched,
            )
            return {
                "blocked": True,
                "messages": [
                    {"role": "assistant", "content": _INJECTION_BLOCKED_RESPONSE}
                ],
            }

        return {}

    return input_guard_node
