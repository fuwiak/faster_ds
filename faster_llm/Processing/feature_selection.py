"""Utility functions for feature selection."""

from __future__ import annotations

import pandas as pd

from .utils import mcp_notify


class FeatureSelection:
    """Utility class for feature selection algorithms."""

    def __init__(self) -> None:
        """Initialize the feature selection class."""

    @staticmethod
    @mcp_notify
    def cor_selector(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        method: str = "pearson",
        threshold: float = 0.0,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> list:
        """Select top ``n_features`` correlated with ``y``."""
        if method not in {"pearson", "kendall", "spearman"}:
            raise ValueError("Invalid correlation method")
        cor = X.corrwith(y, method=method).abs()
        if threshold > 0:
            cor = cor[cor > threshold]
        return list(cor.sort_values(ascending=False).head(n_features).index)

    @staticmethod
    @mcp_notify
    def chi2_selector(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> list:
        """Select features using the chi-squared statistic."""
        from sklearn.feature_selection import SelectKBest, chi2

        selector = SelectKBest(chi2, k=n_features)
        selector.fit(X, y)
        cols = selector.get_support(indices=True)
        return X.columns[cols].tolist()

    @staticmethod
    @mcp_notify
    def rfe_selector(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> list:
        """Select features using recursive feature elimination."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFE

        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=n_features)
        selector = selector.fit(X, y)
        return X.columns[selector.support_].tolist()

    @staticmethod
    @mcp_notify
    def lasso_selector(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> list:
        """Select features using Lasso regression importance."""
        from sklearn.linear_model import Lasso

        model = Lasso(alpha=0.01)
        model.fit(X, y)
        importance = pd.Series(abs(model.coef_), index=X.columns)
        return importance.sort_values(ascending=False).head(n_features).index.tolist()

    @staticmethod
    @mcp_notify
    def xgb_reg_feat_importances(
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> list:
        """Return feature importances from a gradient boosting regressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor()
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return imp.sort_values(ascending=False).head(n_features).index.tolist()
