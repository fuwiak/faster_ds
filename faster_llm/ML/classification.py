"""Utilities for training and evaluating classification models."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)

from .base import BaseModel


class Model(BaseModel):
    """Thin wrapper around a scikit-learn classifier."""

    def _compute_metrics(self) -> dict:
        return {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred, average="binary"),
            "precision": precision_score(self.y_test, self.y_pred, average="binary"),
            "f1": f1_score(self.y_test, self.y_pred, average="binary"),
        }

    def send_metrics_to_llm(self) -> None:
        """Send computed classification metrics to an attached LLM."""
        from faster_llm.LLM import send_to_llm

        send_to_llm(
            f"Classification metrics: {self._compute_metrics()}",
            server_url=self.server_url,
        )

    @staticmethod
    def plot_roc_curve(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series) -> None:
        y_pred = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def plot_log_loss(model: sklearn.base.BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        plt.plot(log_loss(y_test, y_pred_proba))
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series) -> None:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=["True Neg", "True Pos"], columns=["Pred Neg", "Pred Pos"])
        cm_df.index.name = "Actual"
        cm_df.columns.name = "Predicted"
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt="g")
        plt.show()
