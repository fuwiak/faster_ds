import json
import urllib.request

from faster_llm.LLM.client import MCPClient


def test_mcpclient_send_request(monkeypatch):
    captured = {}

    class DummyResponse:
        def read(self):
            return b'{"jsonrpc": "2.0", "result": 42}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def fake_urlopen(req):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = MCPClient("http://example.com")
    result = client.send_request("foo", {"bar": 1})

    assert result == {"jsonrpc": "2.0", "result": 42}
    payload = json.loads(captured["data"].decode())
    assert payload["method"] == "foo"
    assert payload["params"] == {"bar": 1}
    assert captured["url"] == "http://example.com"


def test_mcpclient_deliver_message(monkeypatch):
    recorded = {}

    def fake_send_request(self, method, params):
        recorded["method"] = method
        recorded["params"] = params
        return {"ok": True}

    monkeypatch.setattr(MCPClient, "send_request", fake_send_request)

    client = MCPClient("http://example.com")
    result = client.deliver_message("hello")

    assert result == {"ok": True}
    assert recorded == {"method": "deliver_message", "params": {"message": "hello"}}

