import numpy as np
import pandas as pd
import pickle

class ToProduction:
  
	@staticmethod
	def dump_to_pickle(clf, filename):
		import pickle
		pickle.dump(model, open(filename, 'wb'))

	@staticmethod
	def load_from_pickle(clf, filename):
		import pickle
		loaded_model = pickle.load(open(filename, 'rb'))
		return loaded_model
