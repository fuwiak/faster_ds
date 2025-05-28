import json
from unittest.mock import patch

import faster_ds.LLM as llm


def test_send_mcp_request():
    payload = {"jsonrpc": "2.0", "id": "1", "result": "ok"}

    class DummyResponse:
        def __init__(self, data):
            self._data = data
        def read(self):
            return json.dumps(self._data).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    with patch("faster_ds.LLM.mcp.request.urlopen", return_value=DummyResponse(payload)) as urlopen:
        result = llm.mcp.send_mcp_request("ping", {"foo": "bar"}, "http://test")

    urlopen.assert_called_once()
    sent_request = urlopen.call_args[0][0]
    sent_body = json.loads(sent_request.data.decode("utf-8"))
    assert sent_body["method"] == "ping"
    assert sent_body["params"] == {"foo": "bar"}
    assert result == payload


def test_send_to_llm_via_mcp():
    class DummyResponse:
        def read(self):
            return b"{}"
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    with patch("faster_ds.LLM.mcp.request.urlopen", return_value=DummyResponse()) as urlopen:
        llm.send_to_llm("hello", server_url="http://test")

    urlopen.assert_called_once()
    sent_request = urlopen.call_args[0][0]
    sent_body = json.loads(sent_request.data.decode("utf-8"))
    assert sent_body["method"] == "deliver_message"
    assert sent_body["params"]["message"] == "hello"
