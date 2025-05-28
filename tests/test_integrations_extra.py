import json
import urllib.request

from faster_llm.integrations import log_mlflow_metrics, log_to_langsmith


def test_log_mlflow_metrics(monkeypatch):
    logged = []

    def fake_log_metric(key, value):
        logged.append((key, value))

    monkeypatch.setattr("faster_llm.integrations.mlflow.log_metric", fake_log_metric, raising=False)

    log_mlflow_metrics({"a": 1, "b": 2})

    assert ("a", 1.0) in logged
    assert ("b", 2.0) in logged


def test_log_to_langsmith(monkeypatch):
    captured = {}

    class DummyResp:
        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def fake_urlopen(req):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return DummyResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    log_to_langsmith({"x": 1}, "http://localhost")

    assert captured["url"] == "http://localhost"
    assert json.loads(captured["data"].decode()) == {"x": 1}

