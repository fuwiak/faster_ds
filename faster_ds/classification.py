import pandas as pd


'''

methods for binary classification

'''


class model:

	@staticmethod
	def clf_details(clf):
		return clf

	
	@staticmethod
	def plot_roc_curve(clf, train_X, train_y, test_X):
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

		from sklearn.metrics import log_loss
		import  matplotlib.pylab  as plt
		from numpy import array
		
		# plot input to loss
		plt.plot(yhat, losses_0, label='true=0')
		plt.plot(yhat, losses_1, label='true=1')
		plt.legend()
		plt.show()

	@staticmethod
	def plot_acc_epoch():
		pass

	@staticmethod
	def confusion_matrix(clf, train_X, train_y, test_y):
		clf = clf.fit(train_X, train_y)
		predictions = clf.predict(test_X)

		from sklearn.metrics import classification_report
		print(classification_report(test_y, predictions))


	def compare_algorithms(sorted_by_measure='accuracy'):
		#show grid with compared results - accuracy, recall, ppv, f1-measure, mcc
		pass

	def random_search():
		pass
	
	@staticmethod
	def grid_search(clf, num_split=5):
		from sklearn.model_selection import StratifiedKFold
		clf_gs = GridSearchCV(clf, parameters, n_jobs=-1, cv = StratifiedKFold(shuffle=True,n_splits=num_split), 
                      scoring='accuracy',verbose=2, refit=True)

		clf_gs.fit(X_train,Y_train)
		
		return clf_gs

		

	def show_grid_time():
		pass


	def show_random_search_time():
		pass

	@staticmethod
	def dump_to_pickle(clf, filename):
		import pickle
		pickle.dump(model, open(filename, 'wb'))

	@staticmethod
	def load_from_pickle(clf, filename):
		import pickle
		loaded_model = pickle.load(open(filename, 'rb'))
		return loaded_model






	

