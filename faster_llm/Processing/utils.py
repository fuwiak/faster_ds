from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from faster_llm.LLM import send_to_llm


def mcp_notify(func: Callable) -> Callable:
    """Decorator to optionally notify an MCP server after a function call."""

    @wraps(func)
    def wrapper(*args: Any, notify: bool = False, server_url: str | None = None, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if notify:
            send_to_llm(f"{func.__name__} executed", server_url=server_url)
        return result

    return wrapper
