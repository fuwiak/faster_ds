import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    recall_score,
    precision_score,
)
from sklearn.metrics import log_loss, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from faster_ds.LLM import send_to_llm





class Model:
        def __init__(self, model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, send_to_llm_flag: bool = False):
		"""
		:param model: sklearn model
		:param X: features
		:param y: target
		:param test_size: test size

		"""
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
                self.model = model
                self.model.fit(self.X_train, self.y_train)
                self.y_pred = self.model.predict(self.X_test)
                self.metrics = self._compute_metrics()
                if send_to_llm_flag:
                        self.send_metrics_to_llm()


	@staticmethod
	def clf_details(model: sklearn.base.BaseEstimator)->str:
		"""
		:param clf: classifier
		:return: classifier details

		"""
		return str(dir(model))

	
	@staticmethod
	def plot_roc_curve(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series)-> None:
		"""
		:param model: sklearn model
		:param X: features
		:param y: target
		:return: plot roc curve

		"""
		y_pred = model.predict_proba(X)[:, 1]
		fpr, tpr, _ = roc_curve(y, y_pred)
		roc_auc = auc(fpr, tpr)
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()



	@staticmethod
	def plot_log_loss(model, X_test, y_test):
		"""
		:param model: classifier
		:param X_test: test data
		:param y_test: test labels
		:return: plot of log loss

		"""
		y_pred_proba = model.predict_proba(X_test)[::, 1]
		log_loss(y_test, y_pred_proba)
		plt.plot(log_loss(y_test, y_pred_proba))
		plt.show()

	@staticmethod
	def plot_acc_epoch(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series)-> None:
		"""
		:param model: sklearn model
		:param X: features
		:param y: target
		:return: plot accuracy per epoch

		"""
		accuracy = []
		for i in range(1, 100):
			model.fit(X, y)
			y_pred = model.predict(X)
			accuracy.append(accuracy_score(y, y_pred))
		plt.plot(accuracy)
		plt.show()

	@staticmethod
	def plot_confusion_matrix(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series)-> None:
		"""
		:param model: sklearn model
		:param X: features
		:param y: target
		:return: plot confusion matrix

		"""
		y_pred = model.predict(X)
		cm = confusion_matrix(y, y_pred)
		cm = pd.DataFrame(cm, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos'])
		cm.index.name = 'Actual'
		cm.columns.name = 'Predicted'
		plt.figure(figsize=(10, 7))
		sns.heatmap(cm, cmap='Blues', annot=True, annot_kws={"size": 16}, fmt='g')
		plt.show()






	# @staticmethod
	# def plot_acc_epoch():
	# 	""" Home-made mini-batch learning
	# 	    -> not to be used in out-of-core setting!
	# 	"""
	# 	N_TRAIN_SAMPLES = X_train.shape[0]
	# 	N_EPOCHS = 25
	# 	N_BATCH = 128
	# 	N_CLASSES = np.unique(y_train)

	# 	scores_train = []
	# 	scores_test = []

	# 		epoch = 0
	# 		while epoch < N_EPOCHS:
	# 		    print('epoch: ', epoch)
	# 		    # SHUFFLING
	# 		    random_perm = np.random.permutation(X_train.shape[0])
	# 		    mini_batch_index = 0
	# 		    while True:
	# 			# MINI-BATCH
	# 			indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
	# 			mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
	# 			mini_batch_index += N_BATCH

	# 			if mini_batch_index >= N_TRAIN_SAMPLES:
	# 			    break

	# 		    # SCORE TRAIN
	# 		    scores_train.append(mlp.score(X_train, y_train))

	# 		    # SCORE TEST
	# 		    scores_test.append(mlp.score(X_test, y_test))

	# 		    epoch += 1

	# 		""" Plot """
	# 		fig, ax = plt.subplots(2, sharex=True, sharey=True)
	# 		ax[0].plot(scores_train)
	# 		ax[0].set_title('Train')
	# 		ax[1].plot(scores_test)
	# 		ax[1].set_title('Test')
	# 		fig.suptitle("Accuracy over epochs", fontsize=14)
	# 		plt.show()

	@staticmethod
        def confusion_matrix(model, X_test, y_test):
                """
                :param model: classifier
                :param X_test: test data
                :param y_test: test labels
                :return: confusion matrix

                """
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                return cm

        def _compute_metrics(self) -> dict:
                """Return basic classification metrics as a dictionary."""
                return {
                        "accuracy": accuracy_score(self.y_test, self.y_pred),
                        "recall": recall_score(self.y_test, self.y_pred, average="binary"),
                        "precision": precision_score(self.y_test, self.y_pred, average="binary"),
                        "f1": f1_score(self.y_test, self.y_pred, average="binary"),
                }

        def send_metrics_to_llm(self) -> None:
                """Send computed metrics to an attached LLM service."""
                send_to_llm(f"Classification metrics: {self.metrics}")

	# @staticmethod
	# def compare_algorithms2df(sorted_by_measure='accuracy'):
	# 	"""
	# 	show grid with compared results - accuracy, recall, ppv, f1-measure, mcc
	# 	:param sorted_by_measure: measure to sort by
	# 	:return: dataframe of all algorithms sorted by measure
	#
	# 	"""
	#
	# 	MLA_columns = []
	# 	MLA_compare = pd.DataFrame(columns = MLA_columns)
	#
	#
	# 	row_index = 0
	# 	for alg in MLA:
	#
	#
	# 	    predicted = alg.fit(X_train, y_train).predict(X_test)
	# 	    fp, tp, th = roc_curve(y_test, predicted)
	# 	    MLA_name = alg.__class__.__name__
	# 	    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
	# 	    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_train, Y_train), 4)
	# 	    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X1_test, Y1_test), 4)
	# 	    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(Y1_test, predicted)
	# 	    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(Y1_test, predicted)
	# 	    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
	# 	row_index+=1
    #
	# 	MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
	# 	return MLA_compare
	#
	# @staticmethod
	# def roc_curve_MLA(MLA):
	# 	"""
	# 	:param MLA: list of classifiers
	# 	:return: plot of roc curve for each classifier
	#
	# 	"""
	# 	index = 1
	# 	for alg in MLA:
	#
	#
	# 	    predicted = alg.fit(X_train, Y_train).predict(X1_test)
	# 	    fp, tp, th = roc_curve(Y1_test, predicted)
	# 	    roc_auc_mla = auc(fp, tp)
	# 	    MLA_name = alg.__class__.__name__
	# 	    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))
	#
	# 	    index+=1
	#
	# 	plt.title('ROC Curve comparison')
	# 	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	# 	plt.plot([0,1],[0,1],'r--')
	# 	plt.xlim([0,1])
	# 	plt.ylim([0,1])
	# 	plt.ylabel('True Positive Rate')
	# 	plt.xlabel('False Positive Rate')
	# 	plt.show()
	#
	#
	#
	# def metrics(self):
	# 	"""
	# 	:return: accuracy, recall, ppv, f1-measure, mcc
	#
	# 	"""
	# 	accuracy = accuracy_score(self.y_test, self.y_pred)
	# 	recall = recall_score(self.y_test, self.y_pred)
	# 	precision = precision_score(self.y_test, self.y_pred)
	# 	f1 = f1_score(self.y_test, self.y_pred)
	# 	mcc = matthews_corrcoef(self.y_test, self.y_pred)
	# 	return accuracy, recall, precision, f1, mcc
	#




	

