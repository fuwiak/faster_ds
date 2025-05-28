"""Wrappers for popular AI agent frameworks."""

from __future__ import annotations

from typing import Any

try:  # optional LangChain import
    from langchain.schema import AIMessage
except Exception:  # pragma: no cover - optional
    class AIMessage:
        """Fallback AIMessage when LangChain isn't installed."""

        def __init__(self, content: str) -> None:
            self.content = content


def to_langchain_message(message: str) -> AIMessage:
    """Wrap ``message`` in a LangChain ``AIMessage`` object."""
    return AIMessage(message)


try:  # optional LlamaIndex import
    from llama_index.core import Document
except Exception:  # pragma: no cover - optional
    class Document:  # type: ignore
        """Simplified stand-in for LlamaIndex ``Document``."""

        def __init__(self, text: str) -> None:
            self.text = text


def to_llamaindex_document(text: str) -> Document:
    """Create a LlamaIndex ``Document`` (or placeholder) from ``text``."""
    return Document(text)


try:  # optional CrewAI import
    from crewai import Crew
except Exception:  # pragma: no cover - optional
    class Crew:  # type: ignore
        """Basic replacement for a CrewAI ``Crew`` when the dependency is missing."""

        def __init__(self, *_, **__) -> None:
            self.messages: list[str] = []

        def run(self, message: str) -> None:
            self.messages.append(message)


def send_to_crewai(message: str) -> Crew:
    """Send a message to a CrewAI ``Crew`` instance."""
    crew = Crew()
    crew.run(message)
    return crew


try:  # optional AutoGen import
    from autogen import AssistantAgent
except Exception:  # pragma: no cover - optional
    class AssistantAgent:  # type: ignore
        """Minimal stub of AutoGen's ``AssistantAgent``."""

        def __init__(self, name: str, llm_config: dict | None = None) -> None:
            self.name = name
            self.messages: list[str] = []

        def send(self, message: str) -> None:
            self.messages.append(message)


def send_with_autogen(message: str, *, llm_config: dict | None = None) -> AssistantAgent:
    """Create an AutoGen assistant and send it ``message``."""
    agent = AssistantAgent("assistant", llm_config=llm_config or {})
    agent.send(message)
    return agent


__all__ = [
    "to_langchain_message",
    "to_llamaindex_document",
    "send_to_crewai",
    "send_with_autogen",
]
