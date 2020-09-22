import pandas as pd
import numpy as np


class ModelTuning:
  @staticmethod
	def random_search(clf, params):
		from sklearn.model_selection import RandomizedSearchCV
		clf = RandomizedSearchCV(clf, params, random_state=0)
		return clf
	
	@staticmethod
	def grid_search(clf, num_split=5):
		from sklearn.model_selection import StratifiedKFold
		clf_gs = GridSearchCV(clf, parameters, n_jobs=-1, cv = StratifiedKFold(shuffle=True,n_splits=num_split), 
                      scoring='accuracy',verbose=2, refit=True)

		clf_gs.fit(X_train,Y_train)
		
		return clf_gs

		
	@property
	def show_grid_time():
		raise NotImplementedError

	@property
	def show_random_search_time():
		raise NotImplementedError


