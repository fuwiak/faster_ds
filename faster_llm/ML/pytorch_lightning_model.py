from __future__ import annotations

"""Minimal wrapper for PyTorch Lightning style models."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from faster_llm.LLM import send_to_llm
from .pytorch_model import PyTorchModel


@dataclass
class PyTorchLightningModel(PyTorchModel):
    """Wrapper around a PyTorch Lightning model."""

    def __post_init__(self) -> None:
        # Reuse PyTorchModel behaviour assuming model has `fit` and `predict`.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )
        if hasattr(self.model, "fit"):
            self.model.fit(self.X_train, self.y_train, epochs=self.epochs)
        self.y_pred = self.model.predict(self.X_test)
        metrics = self._compute_metrics()
        if self.send_to_llm_flag:
            self.send_metrics_to_llm(metrics)

    def send_metrics_to_llm(self, metrics: dict | None = None) -> None:
        if metrics is None:
            metrics = self._compute_metrics()
        send_to_llm(
            f"PyTorch Lightning metrics: {metrics}", server_url=self.server_url
        )
