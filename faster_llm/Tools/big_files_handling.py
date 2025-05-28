"""Tools for reducing memory usage of large data frames."""

import numpy as np
import pandas as pd


class BigFiles:
    """Utility methods for optimizing large datasets."""
    
    @staticmethod
    def reduce_to_16bit(X: pd.DataFrame) -> pd.DataFrame:
        """Downcast numerical columns to 16-bit precision."""
        for col in X.columns:
            if X[col].dtype==np.int64:
                X[col] = X[col].astype(np.int16)
            if X[col].dtype==np.float64:
                X[col] = X[col].astype(np.float16)
        return X
    
    @staticmethod
    def reduce_to_category(df: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns to ``category`` dtype."""
        df = df.select_dtypes(include=['object']).copy()
        df = df.astype('category')
        return df
   
