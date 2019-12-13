import pandas as pd



class model:
	def __init__(self, data):
		self.data = data

	def csv_as_df(self, sep="\t"):
		return pd.read_csv(self.data, sep)

	def column_names(self):
		return list(pd.read_csv(self.data).columns)


	def set_X(self, X_name,sep="\t"):
		# assert type(X_name) == 'str'
		#X_name have to be a list
		df = pd.read_csv(self.data, sep)

		return df[X_name]

	def set_Y(self, Y_name, sep="\t"):
		df = pd.read_csv(self.data,sep)

		return df[Y_name]

	def test_train(self,X, y, ratio,random_state=100):
		from sklearn.model_selection import train_test_split


		train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=ratio)
		return train_X, test_X, train_y, test_y 


	def na_handling(self, name_of_strategy):
		#list of stategies -> mean, mode, 0, next_row, previous_row
		pass

	def na_non_na_set(self, data):
		#split set to set with all na's and without
		pass


	def print_algorithm_list():
		#print list of available algorithms
		pass

	def plot_roc_curve():
		pass

	def plot_log_loss():
		pass

	def plot_acc_epoch():
		pass


	def confusion_matrix():
		pass

	def compare_algorithms(self, sorted_by_measure='accuracy'):
		#show grid with compared results - accuracy, recall, ppv, f1-measure, mcc
		pass

	def random_search():
		pass

	def grid_search():
		pass

	def show_grid_time():
		pass


	def show_random_serch_time():
		pass

	def dump_to_pickle():
		pass

	def read_pickle():
		pass






	

