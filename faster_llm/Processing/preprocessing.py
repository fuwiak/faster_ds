import pandas as pd
import numpy as np



class Preprocessing:
	"""
	# Sample usage:
	# df_enc, mapping_encoding = encode_to_num_df(X)
	# df_enc = encode_to_num_df(X)

	"""

	@staticmethod
	def csv_as_df(dataset_name: str, path: str = "data/") -> pd.DataFrame:
		"""
		:param dataset_name: name of the dataset
		:param path: path to the dataset
		:return: pandas DataFrame
		"""
		df = pd.read_csv(path + dataset_name)
		return df

	@staticmethod
	def column_names(df: pd.DataFrame) -> list:
		"""
		:param df: pandas DataFrame
		:return: list of column names
		"""
		return df.columns

	@staticmethod
	def set_X_y(df: pd.DataFrame, target: str) -> tuple:
		"""
		:param df: pandas DataFrame
		:param target: target column
		:return: tuple of X and y
		"""
		X = df.drop(target, axis=1)
		y = df[target]
		return X, y

	@staticmethod
	def get_numerical_columns(df: pd.DataFrame) -> list:
		"""
		:param df: pandas DataFrame
		:return: list of numerical columns
		"""
		return df.select_dtypes(include=np.number).columns


	@staticmethod
	def get_categorical_columns(df: pd.DataFrame) -> list:
		"""
		:param df: pandas DataFrame
		:return: list of categorical columns
		"""
		return df.select_dtypes(include=np.object).columns


	@staticmethod
	def is_missing(df: pd.DataFrame) -> pd.DataFrame:
		"""
		:param df: pandas DataFrame
		:return: pandas DataFrame with missing values
		"""
		return df.isnull().sum()


	@staticmethod
	def count_missing(df: pd.DataFrame) -> int:
		"""
		:param df: pandas DataFrame
		:return: number of missing values
		"""
		return df.isnull().sum().sum()

	@staticmethod
	def normalization(df: pd.DataFrame) -> pd.DataFrame:
		"""
		:param df: pandas DataFrame
		:return: normalized pandas DataFrame
		"""
		mean = df.mean()
		std = df.std()
		return (df - mean) / std

	
	@staticmethod
	def standarization(df: pd.DataFrame) -> pd.DataFrame:
		"""
		:param df: pandas DataFrame
		:return: standarized pandas DataFrame
		"""
		return df/df.max()

	@staticmethod
	def na_handling(df: pd.DataFrame, strategy: str = "mean", specific_value: str = "0") -> pd.DataFrame:
		"""
		:param df: pandas DataFrame
		:param strategy: strategy to handle missing values
		:param specific_value: specific value to fill missing values
		:return: pandas DataFrame with no missing values

		"""

		#list of stategies -> mean, mode, 0, spefic_value, next_row, previous_row
		if strategy == "mean":
			return df.fillna(df.mean(), inplace=True)
		elif strategy == "mode":
			return df.apply(lambda col: col.fillna(col.mode().iloc[0]), axis=0)
		elif strategy == "0":
			return df.fillna(0, inplace=True)
		elif strategy == "specific_value":
			return df.fillna(specific_value, inplace=True)
		elif strategy == "next_row":
			return df.fillna(method="ffill", inplace=True)
		elif strategy == "previous_row":
			return df.fillna(method="backfill", inplace=True)
		else:
			raise ValueError("Strategy not found")

	
	# @staticmethod
	# def na_column_handling(df, col,name_of_strategy,specific_value="0"):
	# 	if name_of_strategy=="polynomial":
	# 		df[col] = df[col].interpolate(method='polynomial', order=2)
	# 		return df
	#
	# 	elif name_of_strategy=="previous_row":
	# 		df[col]=df[col].fillna(method="backfill", inplace=True)
	# 		return df
	# 	elif name_of_strategy=="next_row":
	# 		df[col] = df[col].fillna(method="ffill", inplace=True)
	# 		return df
	# 	elif name_of_strategy=="0":
	# 		df[col] = df[col].fillna(0, inplace=True)
	# 		return df
	#
	# 	elif name_of_strategy=="specific_value":
	# 		df[col] = df[col].fillna(specific_value, inplace=True)
	# 		return df
	#
	#
	# 	elif name_of_strategy=="mean":
	# 		df[col] =df[col].fillna(df.mean(), inplace=True)
	# 		return df
	# 	elif name_of_strategy=="mode":
	# 		df[col] =df[col].fillna(df.mode(dropna=True), inplace=True)
	# 		return df
	# 	else:
	# 		print("Wrong specified strategy")
	#
	# def na_column_handling_dict(df, col,name_of_strategy,specific_value="0"):
	#   col_dict = {}
	#   col_dict["polynomial"] = df[col].interpolate(method='polynomial', order=2)
	#   col_dict["previous_row"] = df[col]=df[col].fillna(method="backfill", inplace=True)
	#   col_dict["next_row"] = df[col].fillna(method="ffill", inplace=True)
	#   col_dict["O"] = df[col].fillna(0, inplace=True)
	#   col_dict["specific_value"] = df[col].fillna(specific_value, inplace=True)
	#   col_dict["mean"] = df[col].fillna(df.mean(), inplace=True)
	#   col_dict["mode"] = df[col].apply(lambda col: col.fillna(col.mode().iloc[0]), axis=0)
	#   return col_dict
	#
	#
	# @staticmethod
	# def na_non_na_set(df):
	#
	# 	#split set to set with all na's and without
	# 	return df[df.isnull().any(axis=1)]
	#
	# @staticmethod
	# def show_columns_with_nan(df):
	# 	list_ = df.columns[df.isna().any()].tolist()
	# 	return list_
	#
	#
	#
	# # @staticmethod
	# # def encode_object(X_train, X_test):
	# # 	from sklearn import preprocessing
	#
	# # 	# Label Encoding
	# # 	for f in X_train.columns:
	# # 	    if X_train[f].dtype=='object' or X_test[f].dtype=='object':
	# # 		lbl = preprocessing.LabelEncoder()
	# # 		lbl.fit(list(X_train[f].values) + list(X_test[f].values))
	# # 		X_train[f] = lbl.transform(list(X_train[f].values))
	# # 		X_test[f] = lbl.transform(list(X_test[f].values))
	# # 	return X_train, X_test
	#
	#
	#
	#
	#
	#
	#
	# @staticmethod
	# def encode_to_num_df(df):
	# 	from sklearn.preprocessing import LabelEncoder
	# 	df = df.apply(LabelEncoder().fit_transform)
	# 	return df
	#
	# @staticmethod
	# def decode_label_df(df, le):
	# 	df = df.apply(le.inverse_transform)
	# 	return df
	#
	# @staticmethod
	# def encode_single_column(df, col_name):
	# 	from sklearn.preprocessing import LabelEncoder
	# 	le = LabelEncoder()
	# 	df[col_name] = le.fit_transform(df[col_name])
	# 	return df
	#
	# @staticmethod
	# def decode_single_column(df, col_name, le):
	# 	from sklearn.preprocessing import LabelEncoder
	# 	le = LabelEncoder()
	# 	df[col_name] = le.inverse_transform(df[col_name])
	# 	return df
	#
	# @staticmethod
	# def one_hot_encode(df):
	# 	# One hot encoding
	# 	df  = pd.get_dummies(df)
	# 	return df
	#
	# @staticmethod
	# def decode_one_hot(df):
	# 	pass
	#
	# def encode_to_num_df(col: pd.Series):
	# 	"""
	# 	Sample usage:
	# 	mapping_encoding = []
	# 	for name in X.columns:
	# 		df_enc[name], d = encode_to_num_df(X[name])
	# 		mapping_encoding.append(d)
	#
	#
	#
	# 	"""
	# 	from sklearn.preprocessing import LabelEncoder
	# 	le = LabelEncoder()
	# 	col_enc = le.fit_transform(col)
	# 	d = dict(zip(le.transform(le.classes_),le.classes_))
	# 	return col_enc, d
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	# @staticmethod
	# def remove_collinear_var(df,threshold=0.9):
	# 	"""Remove Collinear Variables"""
	# 	corr_matrix = df.corr().abs()
	# 	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	# 	to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
	# 	result = df.drop(columns = to_drop, inplace=True)
	# 	return result
	#
	# @staticmethod
	# def remove_to_lot_missing(df, threshold=0.7):
	# 	missing = (df.isnull().sum() / len(df))
	# 	df_missing = missing.index[train_missing > threshold]
	# 	result = df.drop(columns = missing, inplace=True)
	# 	return result
	#
	# @staticmethod
	# def test_train(X, y, ratio=0.3,random_state=100):
	# 	from sklearn.model_selection import train_test_split
	#
	#
	# 	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=ratio)
	# 	return train_X, test_X, train_y, test_y
	#
	#
	#
	#



