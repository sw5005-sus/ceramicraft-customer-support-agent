"""Tests for MLflow tracing utilities."""

from unittest.mock import MagicMock, patch


def _reset_init_flag():
    """Reset the global _MLFLOW_INITIALIZED flag between tests."""
    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mu._MLFLOW_INITIALIZED = False


def _mock_settings(tracking_uri: str = "", experiment: str = "ceramicraft-cs-agent"):
    """Return a MagicMock Settings with MLflow fields populated."""
    cfg = MagicMock()
    cfg.MLFLOW_TRACKING_URI = tracking_uri
    cfg.CS_AGENT_MLFLOW_EXPERIMENT_NAME = experiment
    return cfg


def test_init_no_tracking_uri():
    """Should skip silently when MLFLOW_TRACKING_URI is not set."""
    _reset_init_flag()

    with patch(
        "ceramicraft_customer_support_agent.mlflow_utils.get_settings",
        return_value=_mock_settings(tracking_uri=""),
    ):
        from ceramicraft_customer_support_agent.mlflow_utils import init_mlflow_tracing

        init_mlflow_tracing()  # should not raise


def test_init_idempotent():
    """Calling init_mlflow_tracing() twice should be safe."""
    _reset_init_flag()

    with patch(
        "ceramicraft_customer_support_agent.mlflow_utils.get_settings",
        return_value=_mock_settings(tracking_uri=""),
    ):
        from ceramicraft_customer_support_agent.mlflow_utils import init_mlflow_tracing

        init_mlflow_tracing()
        init_mlflow_tracing()  # second call should be no-op


def test_init_with_tracking_uri():
    """Should call mlflow setup when MLFLOW_TRACKING_URI is set."""
    import sys

    import ceramicraft_customer_support_agent.mlflow_utils as mu

    mu._MLFLOW_INITIALIZED = False

    mock_autolog = MagicMock()
    mock_langchain_mod = MagicMock()
    mock_langchain_mod.autolog = mock_autolog

    with (
        patch(
            "ceramicraft_customer_support_agent.mlflow_utils.get_settings",
            return_value=_mock_settings(
                tracking_uri="http://localhost:5000",
                experiment="test-experiment",
            ),
        ),
        patch("ceramicraft_customer_support_agent.mlflow_utils.mlflow") as mock_mlflow,
        patch.dict(sys.modules, {"mlflow.langchain": mock_langchain_mod}),
    ):
        mu.init_mlflow_tracing()

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_autolog.assert_called_once_with(log_traces=True)

    mu._MLFLOW_INITIALIZED = False


def test_tag_trace_calls_update_when_active_span():
    """tag_trace should call update_current_trace when an active span exists."""
    with patch("ceramicraft_customer_support_agent.mlflow_utils.mlflow") as mock_mlflow:
        mock_mlflow.get_current_active_span.return_value = MagicMock()

        from ceramicraft_customer_support_agent.mlflow_utils import tag_trace

        tag_trace({"intent": "browse", "authenticated": "true"})
        mock_mlflow.update_current_trace.assert_called_once_with(
            tags={"intent": "browse", "authenticated": "true"}
        )


def test_tag_trace_skips_when_no_active_span():
    """tag_trace should no-op when there is no active span."""
    with patch("ceramicraft_customer_support_agent.mlflow_utils.mlflow") as mock_mlflow:
        mock_mlflow.get_current_active_span.return_value = None

        from ceramicraft_customer_support_agent.mlflow_utils import tag_trace

        tag_trace({"intent": "browse"})
        mock_mlflow.update_current_trace.assert_not_called()


def test_tag_trace_silences_errors():
    """tag_trace should not propagate exceptions from mlflow."""
    with patch("ceramicraft_customer_support_agent.mlflow_utils.mlflow") as mock_mlflow:
        mock_mlflow.get_current_active_span.side_effect = RuntimeError("boom")

        from ceramicraft_customer_support_agent.mlflow_utils import tag_trace

        tag_trace({"intent": "order"})  # must not raise
