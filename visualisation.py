import classification as cl



def select_categorical(df):
	'''
	
	select category/object columns from df


	returns dataframe

	'''


	col=df.select_dtypes(include=['object'])
	return col


md = cl.model("sample_data/titanic.csv")
df = md.csv_as_df()
columns = md.column_names()
columns = columns[0].split("\t")
y_name = "Survived"
X_names = [x for x in columns if x !=y_name]
X = md.set_X(X_names) #type dataframe
y = md.set_Y(y_name)


