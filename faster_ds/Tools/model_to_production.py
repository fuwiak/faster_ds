import numpy as np
import pandas as pd
import pickle

class ToProduction:
  
	@staticmethod
	def dump_to_pickle(model, filename: str)-> None:
		pickle.dump(model, open(filename, 'wb'))

	@staticmethod
	def load_from_pickle(filename: str)->None:
		loaded_model = pickle.load(open(filename, 'rb'))
		return loaded_model
