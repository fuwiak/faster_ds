


def smote(X,y);
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(ratio='minority')
  X_sm, y_sm = smote.fit_sample(X, y)
  return X_sm, y_sm
