"""Hyperparameter tuning utilities with optional MCP notifications."""

import pandas as pd
import numpy as np

from .utils import mcp_notify


class ModelTuning:
    """Simple wrappers around common hyperparameter search utilities."""

    @staticmethod
    @mcp_notify
    def grid_search(
        model,
        param_grid: dict,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ):
        """Run ``GridSearchCV`` and return the fitted search object."""
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(model, param_grid)
        search.fit(X, y)
        return search


