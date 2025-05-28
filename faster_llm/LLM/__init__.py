"""Utility helpers for interacting with Large Language Models."""

from __future__ import annotations

from typing import Any

from .client import MCPClient
from .mcp import send_mcp_request
from ..integrations import LLM_MESSAGES_SENT


def send_to_llm(message: Any, *, server_url: str | None = None) -> None:
    """Send information to an LLM.
    If ``server_url`` is provided the message is sent using the Model Context
    Protocol (MCP) via JSON-RPC 2.0. Otherwise the message is printed to the
    console. This function acts as a lightweight bridge between the library and
    external MCP-compatible tools.
    """
    if server_url is not None:
        send_mcp_request("deliver_message", {"message": message}, server_url)
    else:
        print(f"[LLM]: {message}")
    # record metrics for monitoring
    LLM_MESSAGES_SENT.inc()


__all__ = ["send_to_llm", "send_mcp_request", "MCPClient"]
