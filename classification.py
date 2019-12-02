import pandas as pd



class model:
	def __init__(self, data):
		self.data = data

	def csv_as_df(self):
		return pd.read_csv(self.data)

	def column_names(self):
		return list(pd.read_csv(self.data).columns)


	def set_X(self, X_name):
		# assert type(X_name) == 'str'
		df = pd.read_csv(self.data)

		return df[X_name]

	def set_Y():
		pass

	def test_train(self, ratio):
		pass

	def set_algorithm_name():
		pass

	def plot_roc_curve():
		pass

	def confusion_matrix():
		pass

	def compare_algorithms():
		pass

	def random_search():
		pass

	def grid_search():
		pass

	def show_grid_time():
		pass


	def show_random_time():
		pass
	



	

