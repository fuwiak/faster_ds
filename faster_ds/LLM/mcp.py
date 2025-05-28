"""Basic client utilities for the Model Context Protocol (MCP)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

import json
from urllib import request


def send_mcp_request(method: str, params: Dict[str, Any], server_url: str) -> Dict[str, Any]:
    """Send a JSON-RPC 2.0 request via MCP to ``server_url``.

    Parameters
    ----------
    method:
        The MCP method/command to invoke.
    params:
        Parameters to send with the request.
    server_url:
        Endpoint of the MCP server.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON response from the server.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(server_url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as resp:
        resp_data = resp.read().decode("utf-8")
    return json.loads(resp_data)
