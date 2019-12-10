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
		#X_name have to be a list
		df = pd.read_csv(self.data)

		return df[X_name]

	def set_Y(self, Y_name):
		df = pd.read_csv(self.data)

		return df[Y_name]

	def test_train(self, ratio,random_state=100):
		from sklearn.model_selection import train_test_split

		train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=ratio, random_state)
		return train_X, test_X, train_y, test_y 

	def set_algorithm_name():
		#list_of_algo
		pass

	def plot_roc_curve():
		pass

	def plot_log_loss():
		pass

	def plot_acc_epoch():
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

	def dump_to_pickle():
		pass

	def read_pickle():
		pass




	

