import numpy as np
import pandas as pd


class BigFiles:
    
    @staticmethod
    def get_info(df):
        return df.info

    @staticmethod
    def num_megabytes(df):
        return sum(df.memory_usage()/1024**2)
    
    @staticmethod
    def reduce_to_int16(X):
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
    
