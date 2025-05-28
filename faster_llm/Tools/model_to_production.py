"""Utilities for persisting models to disk."""

from __future__ import annotations

import pickle


class ToProduction:
    """Utilities for saving and loading models."""

    @staticmethod
    def dump_to_pickle(model, filename: str) -> None:
        """Serialize ``model`` to ``filename`` using pickle."""
        pickle.dump(model, open(filename, "wb"))

    @staticmethod
    def load_from_pickle(filename: str):
        """Load a model previously saved with :meth:`dump_to_pickle`."""
        loaded_model = pickle.load(open(filename, "rb"))
        return loaded_model
