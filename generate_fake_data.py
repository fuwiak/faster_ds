from faker import Faker
import pandas as pd
import datetime
import numpy as np
import matplotlib.pylab as plt

'''

	class which generates fake and sample data

'''

class fake_data:

	@staticmethod
	def one_sentence():
		"""
	    
	    Returns text(string)
	    
	    Parameters
	    -----------
	    
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

	@staticmethod
	def classfication_data(how_many):
		"""
		Returns Pandas Dataframe with data to classification

		Parameters
		-----------
		how_many
			number of rows of text in dataframe


		"""
		data=[]
		for i in range(how_many):
			temp= Faker('en_US')
			row = [

			temp.prefix(),
			temp.name(),
			temp.date(pattern="%d-%m-%Y", end_datetime=datetime.date(2020, 1,1)),
			temp.phone_number(),
			temp.email(),
			temp.address(),
			temp.zipcode(),
			temp.city(),
			temp.state(),
			temp.country(),
			temp.year(),
			temp.time(),
			temp.url(),
			np.random.randint(0,2,1)[0]
			]
			data.append(row)

		headers = ["Prefix", "Name", "Birth Date", "Phone Number", "Additional Email Id",
		"Address", "Zip Code", "City","State", "Country", "Year", "Time", "Link", "HaveAjob"]

		df = pd.DataFrame(data, columns=headers)
		return df
	@staticmethod
	def regression_data(how_many):	
		"""
		Returns Pandas Dataframe with data to regression

		Parameters
		-----------
		how_many
			number of rows of text in dataframe


		"""
		data=[]
		for i in range(how_many):
			temp= Faker('en_US')
			row = [

			temp.prefix(),
			temp.name(),
			temp.date(pattern="%d-%m-%Y", end_datetime=datetime.date(2020, 1,1)),
			temp.phone_number(),
			temp.email(),
			temp.address(),
			temp.zipcode(),
			temp.city(),
			temp.state(),
			temp.country(),
			temp.year(),
			temp.time(),
			temp.url(),
			np.random.randint(1000,20000,1)[0]
			]
			data.append(row)

		headers = ["Prefix", "Name", "Birth Date", "Phone Number", "Additional Email Id",
		"Address", "Zip Code", "City","State", "Country", "Year", "Time", "Link", "Salary"]

		df = pd.DataFrame(data, columns=headers)
		return df

	@staticmethod
	def clasterization_data(n_samples=1000, n_features=3):
		from sklearn.datasets.samples_generator import make_blobs

		centers = [(-5, -5), (5, 5), (10, 10)]
		cluster_std = [0.8, 1, 2]

		X, y = make_blobs(n_samples=n_samples, cluster_std=cluster_std, centers=centers, n_features=n_features, random_state=1)

		plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
		plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")
		plt.scatter(X[y == 2, 0], X[y == 1, 1], color="pink", s=10, label="Cluster3")
		plt.show()







