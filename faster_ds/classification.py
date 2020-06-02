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
		""" Home-made mini-batch learning
		    -> not to be used in out-of-core setting!
		"""
		N_TRAIN_SAMPLES = X_train.shape[0]
		N_EPOCHS = 25
		N_BATCH = 128
		N_CLASSES = np.unique(y_train)

		scores_train = []
		scores_test = []

			epoch = 0
			while epoch < N_EPOCHS:
			    print('epoch: ', epoch)
			    # SHUFFLING
			    random_perm = np.random.permutation(X_train.shape[0])
			    mini_batch_index = 0
			    while True:
				# MINI-BATCH
				indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
				mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
				mini_batch_index += N_BATCH

				if mini_batch_index >= N_TRAIN_SAMPLES:
				    break

			    # SCORE TRAIN
			    scores_train.append(mlp.score(X_train, y_train))

			    # SCORE TEST
			    scores_test.append(mlp.score(X_test, y_test))

			    epoch += 1

			""" Plot """
			fig, ax = plt.subplots(2, sharex=True, sharey=True)
			ax[0].plot(scores_train)
			ax[0].set_title('Train')
			ax[1].plot(scores_test)
			ax[1].set_title('Test')
			fig.suptitle("Accuracy over epochs", fontsize=14)
			plt.show()

	@staticmethod
	def confusion_matrix(clf, train_X, train_y, test_y):
		clf = clf.fit(train_X, train_y)
		predictions = clf.predict(test_X)

		from sklearn.metrics import classification_report
		print(classification_report(test_y, predictions))


	def compare_algorithms2df(sorted_by_measure='accuracy'):
		#show grid with compared results - accuracy, recall, ppv, f1-measure, mcc
		
		MLA_columns = []
		MLA_compare = pd.DataFrame(columns = MLA_columns)


		row_index = 0
		for alg in MLA:


		    predicted = alg.fit(X_train, Y_train).predict(X1_test)
		    fp, tp, th = roc_curve(Y1_test, predicted)
		    MLA_name = alg.__class__.__name__
		    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
		    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_train, Y_train), 4)
		    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X1_test, Y1_test), 4)
		    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(Y1_test, predicted)
		    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(Y1_test, predicted)
		    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
		row_index+=1
    
		MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
		return MLA_compare
	
	def roc_curve_MLA():
		
		index = 1
		for alg in MLA:


		    predicted = alg.fit(X_train, Y_train).predict(X1_test)
		    fp, tp, th = roc_curve(Y1_test, predicted)
		    roc_auc_mla = auc(fp, tp)
		    MLA_name = alg.__class__.__name__
		    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

		    index+=1

		plt.title('ROC Curve comparison')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')    
		plt.show()

	
	
	
	
	

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

	@staticmethod
	def dump_to_pickle(clf, filename):
		import pickle
		pickle.dump(model, open(filename, 'wb'))

	@staticmethod
	def load_from_pickle(clf, filename):
		import pickle
		loaded_model = pickle.load(open(filename, 'rb'))
		return loaded_model
	
	@staticmethod
	def metrics(self):
		"""
		- log loss
		- accuracy
		- confusion matrix
		- ROC Curve
		
		"""





	

