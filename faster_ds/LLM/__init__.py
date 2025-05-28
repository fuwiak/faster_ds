"""Utility helpers for interacting with Large Language Models."""

from typing import Any


def send_to_llm(message: Any) -> None:
    """Send information to an LLM.

    Currently this is a simple stub that prints the provided message. In a real
    environment this would forward the data to a connected LLM service.
    """
    print(f"[LLM]: {message}")
