"""Tests for MLflow tracing utilities."""

from unittest.mock import MagicMock


def _reset_init_flag():
    """Reset the global _MLFLOW_INITIALIZED flag between tests."""
    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mu._MLFLOW_INITIALIZED = False


def test_init_no_tracking_uri(monkeypatch):
    """Should skip silently when MLFLOW_TRACKING_URI is not set."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    _reset_init_flag()

    from ceramicraft_customer_support_agent.mlflow_utils import init_mlflow_tracing

    # Should not raise
    init_mlflow_tracing()


def test_init_mlflow_import_fails(monkeypatch):
    """Should skip gracefully when mlflow is not installed."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Patch the module-level mlflow to None
    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mu._MLFLOW_INITIALIZED = False
    original_mlflow = mu.mlflow
    mu.mlflow = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    try:
        mu.init_mlflow_tracing()  # should not raise
    finally:
        mu.mlflow = original_mlflow
        mu._MLFLOW_INITIALIZED = False


def test_init_idempotent(monkeypatch):
    """Calling init_mlflow_tracing() twice should be safe."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    _reset_init_flag()

    from ceramicraft_customer_support_agent.mlflow_utils import init_mlflow_tracing

    init_mlflow_tracing()
    init_mlflow_tracing()  # second call should be no-op


def test_init_with_tracking_uri(monkeypatch):
    """Should call mlflow setup when MLFLOW_TRACKING_URI is set."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")

    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mu._MLFLOW_INITIALIZED = False

    mock_mlflow = MagicMock()
    mock_mlflow.langchain = MagicMock()
    mock_mlflow.langchain.autolog = MagicMock()
    mu.mlflow = mock_mlflow  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    try:
        mu.init_mlflow_tracing()
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.langchain.autolog.assert_called_once()
    finally:
        # Restore real mlflow (or None if not installed)
        try:
            import mlflow as real_mlflow

            mu.mlflow = real_mlflow  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        except ImportError:
            mu.mlflow = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        mu._MLFLOW_INITIALIZED = False


def test_tag_trace_no_mlflow():
    """tag_trace should be a no-op when mlflow is None."""
    import ceramicraft_customer_support_agent.mlflow_utils as mu

    from typing import Any, cast

    original = mu.mlflow
    mu.mlflow = cast(Any, None)
    try:
        mu.tag_trace({"intent": "browse"})  # must not raise
    finally:
        mu.mlflow = original


def test_tag_trace_calls_update(monkeypatch):
    """tag_trace should call mlflow.update_current_trace with the given tags."""
    from typing import Any, cast
    from unittest.mock import MagicMock

    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mock_mlflow = MagicMock()
    original = mu.mlflow
    mu.mlflow = cast(Any, mock_mlflow)
    try:
        mu.tag_trace({"intent": "browse", "authenticated": "true"})
        mock_mlflow.update_current_trace.assert_called_once_with(
            tags={"intent": "browse", "authenticated": "true"}
        )
    finally:
        mu.mlflow = original


def test_tag_trace_silences_errors():
    """tag_trace should not propagate exceptions from mlflow."""
    from typing import Any, cast
    from unittest.mock import MagicMock

    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mock_mlflow = MagicMock()
    mock_mlflow.update_current_trace.side_effect = RuntimeError("boom")
    original = mu.mlflow
    mu.mlflow = cast(Any, mock_mlflow)
    try:
        mu.tag_trace({"intent": "order"})  # must not raise
    finally:
        mu.mlflow = original
