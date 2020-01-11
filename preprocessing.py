import pandas as pd



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


 
