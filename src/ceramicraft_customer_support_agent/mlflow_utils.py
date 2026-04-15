"""MLflow tracing utilities for the Customer Support Agent."""

import logging
import warnings

import mlflow

from ceramicraft_customer_support_agent.config import get_settings

logger = logging.getLogger(__name__)

_MLFLOW_INITIALIZED = False


# Suppress known MLflow async ContextVar warning (upstream bug in MLflow 3.x + LangGraph async)
class _ContextVarFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "was created in a different Context" not in record.getMessage()


logging.getLogger("mlflow.utils.autologging_utils").addFilter(_ContextVarFilter())


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
        if hasattr(mlflow, "langchain") and hasattr(mlflow.langchain, "autolog"):
            try:
                mlflow.langchain.autolog(log_traces=True, run_tracer_inline=True)
            except TypeError:
                mlflow.langchain.autolog()
            logger.info("mlflow.langchain.autolog enabled.")
        else:
            logger.info("mlflow.langchain.autolog not available; skipping.")
    except Exception:
        logger.info("mlflow.langchain.autolog failed; skipping.")

    logger.info(
        "MLflow tracing initialized. tracking_uri=%s experiment=%s",
        settings.MLFLOW_TRACKING_URI,
        settings.CS_AGENT_MLFLOW_EXPERIMENT_NAME,
    )
    _MLFLOW_INITIALIZED = True


def tag_trace(tags: dict) -> None:
    """Add custom tags to the current MLflow trace.

    Silently no-ops if no active trace exists (e.g. tracing disabled or
    autolog not available).

    Args:
        tags: Key-value pairs to set on the current trace.
    """
    try:
        if mlflow.get_current_active_span() is None:
            return
        mlflow.update_current_trace(tags=tags)
    except Exception:
        pass  # tracing must never break the business flow
