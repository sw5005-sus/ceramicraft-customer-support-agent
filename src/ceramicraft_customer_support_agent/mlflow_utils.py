"""MLflow tracing utilities for the Customer Support Agent."""

import logging

from ceramicraft_customer_support_agent.config import get_settings

try:
    import mlflow
except Exception:  # pragma: no cover - degrade gracefully when MLflow is absent
    mlflow = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

logger = logging.getLogger(__name__)

_MLFLOW_INITIALIZED = False


def init_mlflow_tracing() -> None:
    """Initialize MLflow tracing once, if MLflow is configured."""
    global _MLFLOW_INITIALIZED

    if _MLFLOW_INITIALIZED:
        return

    if mlflow is None:
        logger.info("MLflow not installed; tracing disabled.")
        _MLFLOW_INITIALIZED = True
        return

    settings = get_settings()
    if not settings.MLFLOW_TRACKING_URI:
        logger.info("MLFLOW_TRACKING_URI not set; MLflow tracing disabled.")
        _MLFLOW_INITIALIZED = True
        return

    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        # LangGraph tracing via LangChain autologging.
        if hasattr(mlflow, "langchain") and hasattr(mlflow.langchain, "autolog"):
            try:
                mlflow.langchain.autolog(log_traces=True)
            except TypeError:
                mlflow.langchain.autolog()

        logger.info(
            "MLflow tracing initialized. tracking_uri=%s experiment=%s",
            settings.MLFLOW_TRACKING_URI,
            settings.MLFLOW_EXPERIMENT_NAME,
        )
    except Exception as exc:  # pragma: no cover - tracing must not break business flow
        logger.warning("Failed to initialize MLflow tracing: %s", exc)

    _MLFLOW_INITIALIZED = True


def tag_trace(tags: dict) -> None:
    """Add custom tags to the current MLflow trace.

    Silently no-ops if MLflow is not initialized or no active trace exists.

    Args:
        tags: Key-value pairs to set on the current trace.
    """
    if mlflow is None:
        return
    try:
        mlflow.update_current_trace(tags=tags)
    except Exception:
        pass  # tracing must never break the business flow
