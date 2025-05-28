from __future__ import annotations

"""Minimal wrapper for Keras style models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from faster_llm.LLM import send_to_llm


@dataclass
class KerasModel:
    """Train and evaluate a Keras-like model and optionally report metrics."""

    model: Any
    X: pd.DataFrame
    y: pd.Series
    epochs: int = 1
    test_size: float = 0.2
    send_to_llm_flag: bool = False
    server_url: str | None = None
    y_pred: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, verbose=0)
        self.y_pred = np.asarray(self.model.predict(self.X_test))
        metrics = self._compute_metrics()
        if self.send_to_llm_flag:
            self.send_metrics_to_llm(metrics)

    def _compute_metrics(self) -> dict:
        preds = self.y_pred
        if preds is None:
            return {}
        preds = preds.squeeze()
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        else:
            preds = (preds > 0.5).astype(int)
        return {"accuracy": accuracy_score(self.y_test, preds)}

    def send_metrics_to_llm(self, metrics: dict | None = None) -> None:
        if metrics is None:
            metrics = self._compute_metrics()
        send_to_llm(f"Keras metrics: {metrics}", server_url=self.server_url)
