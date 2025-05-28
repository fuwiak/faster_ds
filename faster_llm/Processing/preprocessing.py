"""Utility helpers for dataframe preprocessing."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from .utils import mcp_notify


class Preprocessing:
    """Collection of common preprocessing helpers."""

    @staticmethod
    @mcp_notify
    def csv_as_df(
        dataset_name: str,
        path: str = "data/",
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Load a CSV file and return a :class:`~pandas.DataFrame`."""
        return pd.read_csv(f"{path}{dataset_name}")

    @staticmethod
    @mcp_notify
    def column_names(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> List[str]:
        """Return the dataframe column names."""
        return list(df.columns)

    @staticmethod
    @mcp_notify
    def set_X_y(
        df: pd.DataFrame,
        target: str,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into ``X`` and ``y``."""
        X = df.drop(columns=target)
        y = df[target]
        return X, y

    @staticmethod
    @mcp_notify
    def get_numerical_columns(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> List[str]:
        """Return a list of numerical column names."""
        return list(df.select_dtypes(include="number").columns)

    @staticmethod
    @mcp_notify
    def get_categorical_columns(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> List[str]:
        """Return a list of categorical column names."""
        return list(df.select_dtypes(include="object").columns)

    @staticmethod
    @mcp_notify
    def is_missing(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.Series:
        """Return number of missing values per column."""
        return df.isnull().sum()

    @staticmethod
    @mcp_notify
    def count_missing(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> int:
        """Return total number of missing values in dataframe."""
        return int(df.isnull().sum().sum())

    @staticmethod
    @mcp_notify
    def normalization(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Return z-score normalized dataframe."""
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    @staticmethod
    @mcp_notify
    def standarization(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Scale values to [0, 1] range using max value."""
        return df / df.max()

    @staticmethod
    @mcp_notify
    def na_handling(
        df: pd.DataFrame,
        strategy: str = "mean",
        specific_value: str | int | float = 0,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Fill missing values using a chosen strategy."""
        strategies: Dict[str, pd.DataFrame] = {
            "mean": df.fillna(df.mean()),
            "mode": df.apply(lambda col: col.fillna(col.mode().iloc[0])),
            "0": df.fillna(0),
            "zero": df.fillna(0),
            "specific_value": df.fillna(specific_value),
            "next_row": df.fillna(method="ffill"),
            "previous_row": df.fillna(method="bfill"),
        }
        if strategy not in strategies:
            raise ValueError("Strategy not found")
        return strategies[strategy]

    @staticmethod
    @mcp_notify
    def na_column_handling(
        df: pd.DataFrame,
        col: str,
        strategy: str,
        specific_value: str | int | float = 0,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Fill missing values for a single column."""
        if strategy == "polynomial":
            df[col] = df[col].interpolate(method="polynomial", order=2)
        elif strategy in {"previous_row", "bfill"}:
            df[col] = df[col].fillna(method="bfill")
        elif strategy in {"next_row", "ffill"}:
            df[col] = df[col].fillna(method="ffill")
        elif strategy in {"0", "zero"}:
            df[col] = df[col].fillna(0)
        elif strategy == "specific_value":
            df[col] = df[col].fillna(specific_value)
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            raise ValueError("Wrong specified strategy")
        return df

    @staticmethod
    @mcp_notify
    def na_column_handling_dict(
        df: pd.DataFrame,
        col: str,
        specific_value: str | int | float = 0,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> Dict[str, pd.Series]:
        """Return a mapping of strategies to filled columns."""
        return {
            "polynomial": df[col].interpolate(method="polynomial", order=2),
            "previous_row": df[col].fillna(method="bfill"),
            "next_row": df[col].fillna(method="ffill"),
            "0": df[col].fillna(0),
            "specific_value": df[col].fillna(specific_value),
            "mean": df[col].fillna(df[col].mean()),
            "mode": df[col].fillna(df[col].mode().iloc[0]),
        }

    @staticmethod
    @mcp_notify
    def na_non_na_set(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Return rows containing at least one missing value."""
        return df[df.isnull().any(axis=1)]

    @staticmethod
    @mcp_notify
    def show_columns_with_nan(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> List[str]:
        """Return list of columns that contain NaN values."""
        return df.columns[df.isna().any()].tolist()

    @staticmethod
    @mcp_notify
    def encode_object(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Label encode object columns in train and test sets."""
        from sklearn import preprocessing

        for col in X_train.columns:
            if X_train[col].dtype == "object" or X_test[col].dtype == "object":
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(X_train[col].values) + list(X_test[col].values))
                X_train[col] = lbl.transform(list(X_train[col].values))
                X_test[col] = lbl.transform(list(X_test[col].values))
        return X_train, X_test

    @staticmethod
    @mcp_notify
    def encode_to_num_df(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Label encode all columns of a dataframe."""
        from sklearn.preprocessing import LabelEncoder

        return df.apply(LabelEncoder().fit_transform)

    @staticmethod
    @mcp_notify
    def decode_label_df(
        df: pd.DataFrame,
        le,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Inverse transform a dataframe using a fitted encoder."""
        return df.apply(le.inverse_transform)

    @staticmethod
    @mcp_notify
    def encode_single_column(
        df: pd.DataFrame,
        col_name: str,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ):
        """Label encode a single column and return the encoder."""
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        df[col_name] = le.fit_transform(df[col_name])
        return df, le

    @staticmethod
    @mcp_notify
    def decode_single_column(
        df: pd.DataFrame,
        col_name: str,
        le,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Inverse transform a single encoded column."""
        df[col_name] = le.inverse_transform(df[col_name])
        return df

    @staticmethod
    @mcp_notify
    def one_hot_encode(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Return one-hot encoded dataframe."""
        return pd.get_dummies(df)

    @staticmethod
    @mcp_notify
    def decode_one_hot(
        df: pd.DataFrame,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:  # pragma: no cover
        """Decode a one-hot encoded DataFrame back to its original form."""
        raise NotImplementedError

    @staticmethod
    @mcp_notify
    def encode_to_num_series(
        col: pd.Series,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> Tuple[pd.Series, Dict[int, str]]:
        """Encode a single series and return mapping."""
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        col_enc = le.fit_transform(col)
        mapping = dict(zip(le.transform(le.classes_), le.classes_))
        return pd.Series(col_enc, index=col.index), mapping

    @staticmethod
    @mcp_notify
    def remove_collinear_var(
        df: pd.DataFrame,
        threshold: float = 0.9,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Remove highly correlated variables."""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)

    @staticmethod
    @mcp_notify
    def remove_to_lot_missing(
        df: pd.DataFrame,
        threshold: float = 0.7,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Remove columns with a high percentage of missing values."""
        missing = df.isnull().sum() / len(df)
        cols = missing[missing > threshold].index
        return df.drop(columns=cols)

    @staticmethod
    @mcp_notify
    def test_train(
        X: pd.DataFrame,
        y: pd.Series,
        ratio: float = 0.3,
        random_state: int = 100,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size=ratio, random_state=random_state)

