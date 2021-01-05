import pandas as pd
import numpy as np



class PR:

	@staticmethod
	def csv_as_df(dataset_name, sep="\t"):
		assert type(dataset_name)=='str', 'file name must be a string'
		"""

		type(dataset_name)==str

		"""

		return pd.read_csv(dataset_name, sep)


	@staticmethod
	def column_names(dataframe):
		"""

		type(dataframe)==pandas.core.frame.DataFrame

		return

		list

		"""
		if isinstance(dataframe, pd.DataFrame):
			return list(dataframe.columns)

		


		




	@staticmethod
	def set_X_y(df, y_name):
		"""
		Returns 

		X(dataframe)
		y(Series)


		Parameters
		-----------
		df
			Pandas dataframe
		y_name
			Name of y(label name) --> str


		"""

		X = df[df.columns.difference([y_name])]
		y = df[y_name]

		return X,y

	@staticmethod
	def get_numerical_columns(df):
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		numerical_columns = df.select_dtypes(include=numerics).columns
		return df[numerical_columns]

	@staticmethod
	def get_categorical_columns(df):
		category = ['object']
		categorical_columns = df.select_dtypes(include=category).columns
		return df[categorical_columns]


	@staticmethod
	def is_missing(df):
	    "Determine if ANY Value in a Series is Missing"
	    x = df.isnull().values.any()
	    print(x)

	@staticmethod
	def count_missing(df, total=True):
		"Count Missing Values in DataFrame"

		if total:
			print(df.isnull().sum().sum())
		else:
			#by column
			print(print(df.isnull().sum()))

	@staticmethod
	def normalization(df):
	    normalized = df.apply(lambda x: x/max(x))
	    return normalized

	@staticmethod
	def na_handling(df, name_of_strategy,specific_value="0"):
		# sklearn.impute.SimpleImputer


		#list of stategies -> mean, mode, 0, spefic_value, next_row, previous_row

		if name_of_strategy=="previous_row":
			df.fillna(method="backfill", inplace=True)
			return df
		elif name_of_strategy=="next_row":
			df.fillna(method="ffill", inplace=True)
			return df
		elif name_of_strategy=="0":
			df.fillna(0, inplace=True)
			return df
		elif name_of_strategy=="specific_value":
			df.fillna(spefic_value, inplace=True)
			return df
		

		elif name_of_strategy=="mean":
			df.fillna(df.mean(), inplace=True)
			return df
		elif name_of_strategy=="mode":
			df.fillna(df.mode(dropna=True), inplace=True)
			return df
		else:
			print("Wrong specified strategy")
	
	@staticmethod
	def na_column_handling(df, col,name_of_strategy,specific_value="0"):
		if name_of_strategy=="polynomial":
			df[col] = df[col].interpolate(method='polynomial', order=2)
			return df
		
		elif name_of_strategy=="previous_row":
			df.fillna(method="backfill", inplace=True)
			return df
		elif name_of_strategy=="next_row":
			df.fillna(method="ffill", inplace=True)
			return df
		elif name_of_strategy=="0":
			df.fillna(0, inplace=True)
			return df
		
		elif name_of_strategy=="specific_value":
			df.fillna(spefic_value, inplace=True)
			return df
		

		elif name_of_strategy=="mean":
			df.fillna(df.mean(), inplace=True)
			return df
		elif name_of_strategy=="mode":
			df.fillna(df.mode(dropna=True), inplace=True)
			return df
		else:
			print("Wrong specified strategy")


	@staticmethod
	def na_non_na_set(df):

		#split set to set with all na's and without
		return df[df.isnull().any(axis=1)]
	
	@staticmethod
	def show_columns_with_nan(df):
		list_ = df.columns[df.isna().any()].tolist()
		return list_
	
	
	
	# @staticmethod
	# def encode_object(X_train, X_test):
	# 	from sklearn import preprocessing

	# 	# Label Encoding
	# 	for f in X_train.columns:
	# 	    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
	# 		lbl = preprocessing.LabelEncoder()
	# 		lbl.fit(list(X_train[f].values) + list(X_test[f].values))
	# 		X_train[f] = lbl.transform(list(X_train[f].values))
	# 		X_test[f] = lbl.transform(list(X_test[f].values))
	# 	return X_train, X_test
	
	
	
	
	


	@staticmethod
	def encode_to_num_df(df):
		from sklearn.preprocessing import LabelEncoder
		df = df.apply(LabelEncoder().fit_transform)
		return df
	
	@staticmethod
	def decode_label_df(df, le):
		df = df.apply(le.inverse_transform)
		return df
	
	@staticmethod
	def encode_single_column(df, col_name):
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		df[col_name] = le.fit_transform(df[col_name])
		return df
	
	@staticmethod
	def decode_single_column(df, col_name, le):
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		df[col_name] = le.inverse_transform(df[col_name])
		return df	
	
	@staticmethod
	def one_hot_encode(df):
		# One hot encoding
		df  = pd.get_dummies(df)
		return df
	
	@staticmethod
	def decode_one_hot(df):
		pass

	@staticmethod
	def remove_collinear_var(df,threshold=0.9):
		"""Remove Collinear Variables"""
		corr_matrix = df.corr().abs()
		upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
		to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
		result = df.drop(columns = to_drop, inplace=True)
		return result

	@staticmethod
	def remove_to_lot_missing(df, threshold=0.7):
		missing = (df.isnull().sum() / len(df))
		df_missing = missing.index[train_missing > threshold]
		result = df.drop(columns = missing, inplace=True)
		return result

	@staticmethod
	def test_train(X, y, ratio=0.3,random_state=100):
		from sklearn.model_selection import train_test_split


		train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=ratio)
		return train_X, test_X, train_y, test_y 







