from sklearn import preprocessing  # to normalise existing X
from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class ClusterModel:
	"""
	:param path: path to the file
	:param file_name: file name
	:param full_file_name: full file name
	:param df: dataframe
	:param X: data
	:param y: labels
	:param X_train: train data
	:param X_test: test data
	:param y_train: train labels
	:param y_test: test labels

	"""
	def __init__(self, path, file_name, full_file_name, df, X, y, X_train, X_test, y_train, y_test):
		self.path = path
		self.file_name = file_name
		self.full_file_name = full_file_name
		self.df = df
		self.X = X
		self.y = y
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
	
	@staticmethod
	def data_normalization(X: pd.DataFrame) -> pd.DataFrame:
		"""

		normalize dataframe

		Parameters
        ----------
        X : dataframe like, array
            The values for the normalization.

		
		Returns
        -------
		X_norm

        pandas dataframe with normalized columns

		"""
		columns_name = list(X.columns)

		X_Norm = preprocessing.normalize(np.array(X))
		

		X_Norm = pd.DataFrame(X_Norm,columns=columns_name)
		
		return X_Norm

	@staticmethod
	def km_labels(X_norm, n_clusters=5):
		"""
		return k-means labels
		Parameters
		----------
		X_norm: dataframe like, array

		Returns
		-------
		numpy like array


		"""
		km2 = cluster.KMeans(n_clusters=n_clusters,init='random').fit(X_norm)
		labels = km2.labels_
		labels = np.unique(labels)
		return labels

	@staticmethod
	def km_centers(X_norm, n_clusters=5):
		"""
		return k-means centers
		Parameters
		----------
		X_norm: dataframe like, array
		n_clusters: int


		"""

		km2 = cluster.KMeans(n_clusters=n_clusters,init='random').fit(X_norm)
		cluster_centers = km2.cluster_centers_
		return cluster_centers


	@staticmethod
	def predict_labels(X_norm,n_clusters=5):
		"""
		returns k-means labels for new data
		Parameters
		----------
		X_norm: dataframe like, array
		n_clusters: int

		"""
		km2 = cluster.KMeans(n_clusters=n_clusters,init='random')
		y_km2 = km2.fit_predict(X)
		return y_km2



	@staticmethod
	def plot_centroids(X_norm, labels):
		"""
		plot centroids
		"""
		plt.scatter(X_norm[:,0],X_norm[:,1], c=labels, cmap='rainbow',marker='*', label='centroids')
		plt.show()

	@staticmethod
	def plot_clusters(X_norm, cluster_number):
		"""
		plot clusters
		Parameters
		----------
		X_norm: dataframe like, array
		cluster_number: int

		Returns
		-------
		plot of clusters

		"""
		km2 = cluster.KMeans(n_clusters=cluster_number,init='random')
		y_km2 = km2.fit_predict(X_norm)
		plt.scatter(X_norm[:,0],X_norm[:,1], c=y_km2, cmap='rainbow')
		plt.show()














