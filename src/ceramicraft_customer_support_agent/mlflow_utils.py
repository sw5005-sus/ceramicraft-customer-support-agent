"""MLflow tracing utilities for the Customer Support Agent."""

import logging

import mlflow

from ceramicraft_customer_support_agent.config import get_settings

logger = logging.getLogger(__name__)

_MLFLOW_INITIALIZED = False


def init_mlflow_tracing() -> None:
    """Initialize MLflow tracing once, if MLflow is configured."""
    global _MLFLOW_INITIALIZED

    if _MLFLOW_INITIALIZED:
        return

    settings = get_settings()
    if not settings.MLFLOW_TRACKING_URI:
        logger.info("MLFLOW_TRACKING_URI not set; MLflow tracing disabled.")
        _MLFLOW_INITIALIZED = True
        return

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.CS_AGENT_MLFLOW_EXPERIMENT_NAME)

    try:
        langchain_mod = getattr(mlflow, "langchain", None)
    except Exception:
        langchain_mod = None

    if langchain_mod is not None:
        autolog_fn = getattr(langchain_mod, "autolog", None)
        if autolog_fn is not None:
            try:
                autolog_fn(log_traces=True)
            except TypeError:
                autolog_fn()

    logger.info(
        "MLflow tracing initialized. tracking_uri=%s experiment=%s",
        settings.MLFLOW_TRACKING_URI,
        settings.CS_AGENT_MLFLOW_EXPERIMENT_NAME,
    )
    _MLFLOW_INITIALIZED = True


def tag_trace(tags: dict) -> None:
    """Add custom tags to the current MLflow trace.

    Silently no-ops if no active trace exists (e.g. tracing disabled).

    Args:
        tags: Key-value pairs to set on the current trace.
    """
    try:
        mlflow.update_current_trace(tags=tags)
    except Exception:
        pass  # tracing must never break the business flow
