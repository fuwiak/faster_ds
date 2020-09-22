class HandlingImbalanced:
  @staticmethod
  def smote(X,y):
    """
    SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, 
    based on those that already exist. It works randomly picingk a point from the minority class 
    and computing the k-nearest neighbors for this point. 
    The synthetic points are added between the chosen point and its neighbors.
    """
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
    return X_sm, y_sm
