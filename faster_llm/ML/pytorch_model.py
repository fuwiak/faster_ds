from __future__ import annotations

"""Minimal wrapper for PyTorch style models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from faster_llm.LLM import send_to_llm


@dataclass
class PyTorchModel:
    """Train and evaluate a PyTorch-like model and optionally report metrics."""

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
        if hasattr(self.model, "fit"):
            self.model.fit(self.X_train, self.y_train, epochs=self.epochs)
        else:
            try:
                import torch
                self.model.train()
                optimizer = getattr(self.model, "optimizer", torch.optim.SGD(self.model.parameters(), lr=0.01))
                criterion = getattr(self.model, "criterion", torch.nn.CrossEntropyLoss())
                X_tensor = torch.tensor(self.X_train.values, dtype=torch.float32)
                y_tensor = torch.tensor(self.y_train.values)
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    output = self.model(X_tensor)
                    loss = criterion(output, y_tensor)
                    loss.backward()
                    optimizer.step()
            except Exception:
                raise RuntimeError("PyTorch model training not available")
        if hasattr(self.model, "predict"):
            preds = self.model.predict(self.X_test)
        else:
            try:
                import torch
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(torch.tensor(self.X_test.values, dtype=torch.float32))
                preds = preds.detach().numpy()
            except Exception:
                raise RuntimeError("PyTorch model prediction not available")
        self.y_pred = np.asarray(preds)
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
        send_to_llm(f"PyTorch metrics: {metrics}", server_url=self.server_url)
