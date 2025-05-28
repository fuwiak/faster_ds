# -*- coding: utf-8 -*-

"""Package Description."""

__version__ = "0.0.1"
__short_description__ = "Numpy/Pandas based module make faster data analysis"
__license__ = "MIT"
__author__ = "fuwiak"
__author_email__ = "poczta130@gmail.com"
__maintainer__ = "unknown maintainer"
__maintainer_email__ = "maintainer@example.com"
__github_username__ = "fuwiak"

from .agents import (
    send_to_crewai,
    send_with_autogen,
    to_langchain_message,
    to_llamaindex_document,
)

__all__ = [
    "send_to_crewai",
    "send_with_autogen",
    "to_langchain_message",
    "to_llamaindex_document",
]
