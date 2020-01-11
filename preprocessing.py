import pandas as pd


def set_X_y(df, y_name):

	X = df[df.columns.difference([y_name])]
	y = df[y_name]

	return X,y


def get_numerical_columns(df):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	numerical_columns = df.select_dtypes(include=numerics).columns
	return df[numerical_columns]

def get_numerical_columns(df):
	category = ['object']
	numerical_columns = df.select_dtypes(include=category).columns
	return df[numerical_columns]



def is_missing(df):
    "Determine if ANY Value in a Series is Missing"
    x = df.isnull().values.any()
    print(x)


def count_missing(df, total=True):
	"Count Missing Values in DataFrame"

	if total:
		print(df.isnull().sum().sum())
	else:
		#by column
		print(print(df.isnull().sum()))


def normalization(df):
    normalized = df.apply(lambda x: x/max(x))
    return normalized


def encode_to_num_df(df):
	from sklearn.preprocessing import LabelEncoder
	df = df.apply(LabelEncoder().fit_transform)
	return df


def one_hot_encode(df):
	# One hot encoding
	df  = pd.get_dummies(df)
	return df

def remove_collinear_var(df,threshold=0.9):
	"""Remove Collinear Variables"""
	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
	result = df.drop(columns = to_drop)
	return result

def remove_to_lot_missing(df, threshold=0.7):
	missing = (df.isnull().sum() / len(df))
	df_missing = missing.index[train_missing > threshold]
	result = df.drop(columns = missing)
	return result







