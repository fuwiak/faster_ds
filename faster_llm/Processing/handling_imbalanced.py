"""Algorithms for dealing with imbalanced datasets."""


class HandlingImbalanced:
    """Utility methods for balancing data."""

    @staticmethod
    def smote(X, y):
        """Apply the SMOTE algorithm to balance classes."""
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(ratio="minority")
        X_sm, y_sm = smote.fit_sample(X, y)
        return X_sm, y_sm
