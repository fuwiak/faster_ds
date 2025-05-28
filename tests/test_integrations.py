import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faster_llm.integrations import start_metrics_server, LLM_MESSAGES_SENT
from faster_llm.LLM import send_to_llm


def test_metrics_counter_increment(monkeypatch):
    before = LLM_MESSAGES_SENT._value.get()
    send_to_llm("hi")
    after = LLM_MESSAGES_SENT._value.get()
    assert after == before + 1


def test_start_metrics_server(monkeypatch):
    called = {}

    def fake(port):
        called["port"] = port

    monkeypatch.setattr("faster_llm.integrations.start_http_server", fake)
    start_metrics_server(9999)
    assert called["port"] == 9999


