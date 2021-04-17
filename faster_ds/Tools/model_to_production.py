import numpy as np
import pandas as pd
import pickle

class ToProduction:
  
	@staticmethod
	def dump_to_pickle(clf, filename: str)-> None:
		pickle.dump(model, open(filename, 'wb'))

	@staticmethod
	def load_from_pickle(clf, filename: str):
		loaded_model = pickle.load(open(filename, 'rb'))
		return loaded_model
