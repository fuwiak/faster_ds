import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faster_ds.LLM import send_to_llm


def test_send_to_llm(capsys):
    send_to_llm("hello")
    captured = capsys.readouterr()
    assert "[LLM]: hello" in captured.out
