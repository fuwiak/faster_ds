import pandas as pd
import matplotlib.pylab as plt


class PCA:
	@staticmethod
	def pca2df(df, target_name,n_components=2):
		
		from sklearn.preprocessing import StandardScaler
		
		x = df[df.columns.difference([target_name])]
		
		y = df[target_name]
		# Standardizing the features
		x = StandardScaler().fit_transform(x)
		
		from sklearn.decomposition import PCA
		pca = PCA(n_components=n_components)
		principalComponents = pca.fit_transform(x)
		principalDf = pd.DataFrame(data = principalComponents
			     , columns = ['principal component 1', 'principal component 2'])


		finalDf = pd.concat([principalDf, df[[target_name]]], axis = 1) # pca and target
		return finalDf
	
	@staticmethod
	def pca(df,target_name, n_components=2):
		from sklearn.preprocessing import StandardScaler
		x = df[df.columns.difference([target_name])]
		
		y = df[target_name]
		# Standardizing the features
		x = StandardScaler().fit_transform(x)
		
		from sklearn.decomposition import PCA
		pca = PCA(n_components=n_components)
		
		return pca
		
		
	
	
	@staticmethod
	def pca_info(pca):
		return pca.explained_variance_ratio_
	
	@staticmethod
	def plot_explained_variance(pca):
		
		plt.figure(1, figsize=(14, 13))
		plt.clf()
		plt.axes([.2, .2, .7, .7])
		plt.plot(pca.explained_variance_ratio_, linewidth=2)
		plt.axis('tight')
		plt.xlabel('n_components')
		plt.ylabel('explained_variance_ratio_')
		plt.show()
		
	
	def visualize(pca):
		
		
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)
class ICA:
	pass

		
