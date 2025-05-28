from __future__ import annotations

"""Integration helpers for monitoring and experiment tracking."""

from typing import Any, Dict

import json
import urllib.request

try:  # optional dependency
    from prometheus_client import Counter, start_http_server
except Exception:  # pragma: no cover - optional
    class Counter:
        """Simple stand-in for Prometheus Counter when dependency missing."""

        class _Value:
            def __init__(self) -> None:
                self.val = 0

            def get(self) -> float:
                return self.val

            def set(self, v: float) -> None:
                self.val = v

        def __init__(self, *_, **__):
            """Initialize the dummy counter."""
            self._value = self._Value()

        def inc(self, amount: int = 1) -> None:
            """Increase the counter by ``amount``."""
            self._value.set(self._value.get() + amount)

    def start_http_server(*_, **__):  # type: ignore
        """Placeholder for ``start_http_server`` when Prometheus is absent."""
        pass

try:  # optional dependency
    import mlflow
except Exception:  # pragma: no cover - optional
    class DummyMLflow:
        """Fallback MLflow object used when MLflow isn't installed."""

        def log_metric(self, *_, **__):
            """Ignore metric logging when MLflow is unavailable."""
            pass

    mlflow = DummyMLflow()

# Prometheus metric tracking messages sent to LLMs
LLM_MESSAGES_SENT = Counter(
    "llm_messages_sent_total",
    "Total number of messages sent to LLMs",
)


def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus HTTP metrics server on the given port."""
    start_http_server(port)


def log_mlflow_metrics(metrics: Dict[str, Any]) -> None:
    """Log a dictionary of metrics to MLflow."""
    for key, value in metrics.items():
        try:
            mlflow.log_metric(key, float(value))
        except Exception:  # pragma: no cover - optional
            pass


def log_to_langsmith(payload: Dict[str, Any], server_url: str) -> None:
    """Send a JSON payload to a Langsmith server."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server_url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)  # pragma: no cover - optional network
    except Exception:
        # Ignore errors when server is unreachable
        pass


