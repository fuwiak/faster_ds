"""Utility functions for feature selection."""

from __future__ import annotations

import pandas as pd


class FeatureSelection:
    """Utility class for feature selection algorithms."""

    def __init__(self) -> None:
        """Initialize the feature selection class."""

    @staticmethod
    def cor_selector(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        method: str,
        threshold: float,
    ) -> list:
        """Select features based on correlation statistics."""
        return NotImplementedError

    @staticmethod
    def chi2_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
        """Select features using the chi-squared statistic."""
        return NotImplementedError

    @staticmethod
    def rfe_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
        """Select features using recursive feature elimination."""
        return NotImplementedError

    @staticmethod
    def lasso_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
        """Select features using Lasso regression importance."""
        return NotImplementedError

    @staticmethod
    def xgb_reg_feat_importances(
        X: pd.DataFrame, y: pd.Series, n_features: int
    ) -> list:
        """Return XGBoost regression feature importances."""
        return NotImplementedError
