"""Utility helpers for interacting with Large Language Models."""

from typing import Any
import os

try:  # Optional OpenAI dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


def send_to_llm(message: Any) -> None:
    """Send information to an LLM via the OpenAI API if available.

    The function always prints the message to stdout to preserve the previous
    behaviour which is relied upon in tests. If the ``openai`` package is
    installed and the ``OPENAI_API_KEY`` environment variable is set, the
    message is also sent to OpenAI's chat completion endpoint. Any errors during
    the API call are silently ignored so that the absence of network access does
    not break local execution.
    """

    # Always print the message for logging purposes
    print(f"[LLM]: {message}")

    if openai is None:  # pragma: no cover - optional dependency
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:  # pragma: no cover - optional dependency
        return

    openai.api_key = api_key
    try:  # pragma: no cover - avoid failing tests without network
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": str(message)}],
        )
    except Exception:
        # Network or authentication issues should not crash the caller
        pass
