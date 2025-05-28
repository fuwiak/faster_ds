"""Helper functions for MCP interactions."""

from __future__ import annotations

from typing import Any, Dict

from .client import MCPClient


def send_mcp_request(method: str, params: Dict[str, Any], server_url: str) -> Dict[str, Any]:
    """Send a JSON-RPC 2.0 request using the Model Context Protocol (MCP)."""
    client = MCPClient(server_url)
    return client.send_request(method, params)
