"""Lightweight Model Context Protocol client."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MCPClient:
    """Client for communicating with an MCP server via JSON-RPC 2.0."""

    server_url: str
    _request_id: int = field(init=False, default=0)

    def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._request_id += 1
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.server_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as response:  # pragma: no cover - network
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # pragma: no cover - optional network
            return {}

    def deliver_message(self, message: Any) -> Dict[str, Any]:
        return self.send_request("deliver_message", {"message": message})
