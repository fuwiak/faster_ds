"""Algorithms for dealing with imbalanced datasets."""


from .utils import mcp_notify


class HandlingImbalanced:
    """Utility methods for balancing data."""

    @staticmethod
    @mcp_notify
    def smote(
        X,
        y,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ):
        """Apply the SMOTE algorithm to balance classes."""
        from imblearn.over_sampling import SMOTE

        smote = SMOTE()
        X_sm, y_sm = smote.fit_resample(X, y)
        return X_sm, y_sm
