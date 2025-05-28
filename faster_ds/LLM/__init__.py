"""Utility helpers for interacting with Large Language Models."""

from typing import Any, Dict

from .mcp import send_mcp_request


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


__all__ = ["send_to_llm", "send_mcp_request"]
