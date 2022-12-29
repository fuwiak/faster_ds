import numpy as np
import pandas as pd
import matplotlib.pylab as plt



class FeatureSelection:

	def __init__(self):
		pass


	@staticmethod
	def cor_selector(X: pd.DataFrame, y: pd.Series, n_features: int, method: str, threshold: float) -> list:
		"""
		:param X: pandas DataFrame
		:param y: pandas Series
		:param n_features: number of features to select
		:param method: correlation method
		:param threshold: threshold to select features
		:return: list of selected features

		"""
		return NotImplementedError
	
	@staticmethod
	def chi2_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
		"""
		:param X: pandas DataFrame
		:param y: pandas Series
		:param n_features: number of features to select
		:return: list of selected features
		"""
		return NotImplementedError
	
	@staticmethod
	def rfe_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
		"""
		:param X: pandas DataFrame
		:param y: pandas Series
		:param n_features: number of features to select
		:return: list of selected features
		"""
		return NotImplementedError
	
	
	@staticmethod
	def Lasso_selector(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
		"""
		:param X: pandas DataFrame
		:param y: pandas Series
		:param n_features: number of features to select
		:return: list of selected features
		"""
		return NotImplementedError
	


	@staticmethod
	def xgb_reg_feat_importances(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
		"""
		:param X: pandas DataFrame
		:param y: pandas Series
		:param n_features: number of features to select
		:return: list of selected features
		"""
		return NotImplementedError

	
	










