
def num_megabytes(df):
    return sum(df.memory_usage()/1024**2)

  
def reduce_to_int16(X):
  for col in X.columns:
      if X[col].dtype==np.int64:
          X[col] = X[col].astype(np.int16)
      if X[col].dtype==np.float64:
          X[col] = X[col].astype(np.float16)
