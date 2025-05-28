from __future__ import annotations

from typing import Any, Dict
import json
import urllib.request


def send_mcp_request(method: str, params: Dict[str, Any], server_url: str) -> Dict[str, Any]:
    """Send a JSON-RPC 2.0 request using the Model Context Protocol (MCP)."""
    payload = {"jsonrpc": "2.0", "id": 0, "method": method, "params": params}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as response:  # pragma: no cover - network
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # pragma: no cover - optional network
        return {}
