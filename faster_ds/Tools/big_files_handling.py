import numpy as np
import pandas as pd


class BigFiles:
    
    @staticmethod
    def reduce_to_16bit(X):
        for col in X.columns:
            if X[col].dtype==np.int64:
                X[col] = X[col].astype(np.int16)
            if X[col].dtype==np.float64:
                X[col] = X[col].astype(np.float16)
        return X
    
    @staticmethod
    def reduce_to_category(df):
        df = df.select_dtypes(include=['object']).copy()
        df = df.astype('category')
        return df
   
