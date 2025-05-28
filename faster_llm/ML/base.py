"""Shared utilities for ML models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from faster_llm.LLM import send_to_llm


@dataclass
class BaseModel:
    """Common training and evaluation workflow for scikit-learn models."""

    model: Any
    X: pd.DataFrame
    y: pd.Series
    test_size: float = 0.2
    send_to_llm_flag: bool = False
    server_url: str | None = None
    mlflow_tracking: bool = False
    langsmith_url: str | None = None
    y_pred: pd.Series | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        metrics = self._compute_metrics()
        if self.send_to_llm_flag:
            self.send_metrics_to_llm(metrics)
        if self.mlflow_tracking:
            from faster_llm.integrations import log_mlflow_metrics
            log_mlflow_metrics(metrics)
        if self.langsmith_url:
            from faster_llm.integrations import log_to_langsmith
            log_to_langsmith({"metrics": metrics}, self.langsmith_url)

    def _compute_metrics(self) -> dict:
        raise NotImplementedError

    def send_metrics_to_llm(self, metrics: dict | None = None) -> None:
        if metrics is None:
            metrics = self._compute_metrics()
        send_to_llm(f"Model metrics: {metrics}", server_url=self.server_url)

