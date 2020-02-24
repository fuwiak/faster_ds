import pandas as pd


'''

methods for binary classification

'''


class model:

	@staticmethod
	def clf_details(clf):
		return clf

	
	@staticmethod
	def plot_roc_curve(train_X, train_y, test_X):
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

	@staticmethod
	def plot_log_loss():
		pass

	@staticmethod
	def plot_acc_epoch():
		pass

	@staticmethod
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






	

