import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faster_llm
from faster_llm.LLM.mcp import send_mcp_request
from faster_llm.LLM.client import MCPClient


def test_package_has_version():
    assert isinstance(faster_llm.__version__, str)


def test_send_mcp_request_uses_client(monkeypatch):
    captured = {}

    def fake_send_request(self, method, params):
        captured["url"] = self.server_url
        captured["method"] = method
        captured["params"] = params
        return {"ok": True}

    monkeypatch.setattr(MCPClient, "send_request", fake_send_request)

    result = send_mcp_request("ping", {"x": 1}, "http://example.com")

    assert result == {"ok": True}
    assert captured == {
        "url": "http://example.com",
        "method": "ping",
        "params": {"x": 1},
    }
