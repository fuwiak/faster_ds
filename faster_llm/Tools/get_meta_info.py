"""Utilities for inspecting pandas objects."""

import numpy as np
import pandas as pd


class GetInfo:
    """Helper methods for obtaining metadata about data frames."""

    @staticmethod
    def get_info(df: pd.DataFrame) -> None:
        """Print DataFrame info (placeholder)."""
        raise NotImplementedError

    @staticmethod
    def num_megabytes(df: pd.DataFrame) -> float:
        """Return approximate memory footprint of ``df`` in megabytes."""
        return sum(df.memory_usage() / 1024**2)

