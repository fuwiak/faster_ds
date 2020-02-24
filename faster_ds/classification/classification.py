import pandas as pd


'''

methods for binary classification

'''


class model:
	
	@staticmethod
	def csv_as_df(data, sep="\t"):
		"""

		type(data)==Pandas.dataframe

		"""

		return pd.read_csv(data, sep)


	@staticmethod
	def column_names(data):
		return list(pd.read_csv(data).columns)

	@staticmethod
	def set_X(X_name,sep="\t"):
		# assert type(X_name) == 'str'
		#X_name have to be a list
		df = pd.read_csv(data, sep)

		return df[X_name]

	@staticmethod
	def set_Y(Y_name, sep="\t"):
		df = pd.read_csv(data,sep)

		return df[Y_name]



	@staticmethod
	def test_train(X, y, ratio,random_state=100):
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

	def plot_roc_curve(self, train_X, train_y, test_X):
		predictions = clf.fit(train_X, train_y).predict(test_X)
		fp, tp, th = roc_curve(test_y, predictions)
		roc_auc_mla = auc(fp, tp)
		plt.plot(fp, tp, lw=2, alpha=0.3)
		plt.title('ROC Curve comparison')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')    
		plt.show()

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






	

