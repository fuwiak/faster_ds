from sklearn import preprocessing  # to normalise existing X
from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# class CusterInterface:
#     def method1(self, path: str, file_name: str) -> str:
#          """method description"""
#         pass

#     def method2(self, full_file_name: str) -> dict:
#         """method description"""
#         pass



class ClusterModel:
	@staticmethod
	def data_normalization(X):
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
		returns k-means centroids
		

		"""

		km2 = cluster.KMeans(n_clusters=n_clusters,init='random').fit(X_norm)
		cluster_centers = km2.cluster_centers_
		return cluster_centers


	@staticmethod
	def predict_labels(X ,n_clusters=5):
		km2 = cluster.KMeans(n_clusters=n_clusters,init='random')
		y_km2 = km2.fit_predict(X)
		return y_km2



	@staticmethod
	def plot_centroids(X_norm, labels):
		plt.scatter(X_norm[:,0],X_norm[:,1], c=labels, cmap='rainbow',marker='*', label='centroids')
		plt.show()

	@staticmethod
	def plot_clusters(X_norm, cluster_number):
		raise NotImplementedError













