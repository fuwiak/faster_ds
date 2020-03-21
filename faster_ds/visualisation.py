import classification as cl
import seaborn as sns


def select_categorical(df):
	"""
    Returns dataframe with object type
    
    Parameters
    -----------
    df
        Pandas data frame with item data
           
    
    Returns
    -----------
    col
    	Pandas dataframe with object data
    """


	col=df.select_dtypes(include=['object'])
	return col


def select_numerical(df):
	"""
    Returns dataframe with object type
    
    Parameters
    -----------
    df
        Pandas data frame with item data
           
    
    Returns
    -----------
    col
    	Pandas dataframe with numeric data
    """


	col=df.select_dtypes(include=['float', 'int'])
	return col



def select_logical(df):
	"""
    Returns dataframe with object type
    
    Parameters
    -----------
    df
        Pandas data frame with item data
           
    
    Returns
    -----------
    col
    	Pandas dataframe with numeric data
    """


	col=df.select_dtypes(include=['bool'])
	return col

def show_hist(df, bins=5):
	"""
    show histogram on numerical data
    
    Parameters
    -----------
    df
        Pandas data frame with item data
    bins
        Number of bins, default=5
           
    
    -----------







	col=df.select_dtypes(include=['float', 'int'])
	col.hist(bins=bins)
	"""

def show_heatmap(df):
	plt.figure(figsize=(12,10))
	cor = df.corr()
	sns.heatmap(cor, annot=True)
	plt.show()









if __name__ == "__main__":

	md = cl.model("sample_data/titanic.csv")
	df = md.csv_as_df()
	columns = md.column_names()
	columns = columns[0].split("\t")
	y_name = "Survived"
	X_names = [x for x in columns if x !=y_name]
	X = md.set_X(X_names) #type dataframe
	y = md.set_Y(y_name)


