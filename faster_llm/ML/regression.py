"""Utilities for regression models."""

from __future__ import annotations

import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from faster_llm.doc import doc
from .base import BaseModel
from .classification import Model as ClassificationModel


class Model(BaseModel):
    """Simple regression model wrapper with optional LLM reporting."""

    @doc(ClassificationModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_metrics(self) -> dict:
        return {
            "r2": r2_score(self.y_test, self.y_pred),
            "mae": mean_absolute_error(self.y_test, self.y_pred),
            "mse": mean_squared_error(self.y_test, self.y_pred),
        }

    @doc(ClassificationModel.send_metrics_to_llm)
    def send_metrics_to_llm(self) -> None:
        from faster_llm.LLM import send_to_llm

        send_to_llm(
            f"Regression metrics: {self._compute_metrics()}",
            server_url=self.server_url,
        )
