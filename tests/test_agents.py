import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faster_llm import (
    send_to_crewai,
    send_with_autogen,
    to_langchain_message,
    to_llamaindex_document,
)


def test_to_langchain_message():
    msg = to_langchain_message("hi")
    assert hasattr(msg, "content")
    assert msg.content == "hi"


def test_to_llamaindex_document():
    doc = to_llamaindex_document("hello")
    assert hasattr(doc, "text")
    assert doc.text == "hello"


def test_send_to_crewai():
    crew = send_to_crewai("ping")
    assert hasattr(crew, "messages")
    assert "ping" in crew.messages


def test_send_with_autogen():
    agent = send_with_autogen("pong")
    assert hasattr(agent, "messages")
    assert "pong" in agent.messages
