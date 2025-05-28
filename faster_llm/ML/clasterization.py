from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class ClusterModel:
	def __init__(self, X: pd.DataFrame, n_clusters: int = 2, random_state: int = 0):
		"""
		:param X: features
		:param n_clusters: number of clusters
		:param random_state: random state

		"""
		self.X = X
		self.n_clusters = n_clusters
		self.random_state = random_state
		self.model = cluster.KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
		self.model.fit(self.X)
		self.labels = self.model.labels_














