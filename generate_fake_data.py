from faker import Faker
import pandas as pd


'''

class which generates fake and sample data

'''

class fake_data:

	@staticmethod
	def one_sentence():
		 """
	    Returns text(str)
	    Parameters
	    -----------
	    
	    
	        No parameters
	    
	  

    	"""
		temp = Faker()
		return temp.text()

	@staticmethod
	def many_sentences(how_many):
		 """
	    
	    Returns Pandas Dataframe with text
	    
	    Parameters
	    -----------
	    
	    how_many
	        number of rows of text in dataframe
	    """


		temp = Faker()
		data = []
		for i in range(how_many):
			data.append(temp.text())

		return pd.DataFrame(data, columns=["text"])




	def classfication_data(self, *params):
		pass


	def regression_data(self, *params):
		pass


	def clasterization_data(self, *params):
		pass