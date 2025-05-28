import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faster_llm.LLM import send_to_llm, send_mcp_request


def test_send_to_llm(capsys):
    send_to_llm("hello")
    captured = capsys.readouterr()
    assert "[LLM]: hello" in captured.out


def test_send_to_llm_mcp(monkeypatch):
    recorded = {}

    def fake(method, params, server_url):
        recorded["method"] = method
        recorded["params"] = params
        recorded["server_url"] = server_url
        return {}

    monkeypatch.setattr("faster_llm.LLM.send_mcp_request", fake)
    send_to_llm("world", server_url="http://example.com")

    assert recorded == {
        "method": "deliver_message",
        "params": {"message": "world"},
        "server_url": "http://example.com",
    }
