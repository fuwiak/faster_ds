
class PCA:
	@staticmethod
	def pca2comp(target_name,n_comp=2):
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		principalComponents = pca.fit_transform(x)
		principalDf = pd.DataFrame(data = principalComponents
			     , columns = ['principal component 1', 'principal component 2'])


		finalDf = pd.concat([principalDf, df[[target_name]]], axis = 1) # pca and target

		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)
		targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
		colors = ['r', 'g', 'b']
		for target, color in zip(targets,colors):
		    indicesToKeep = finalDf['target'] == target
		    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
			       , finalDf.loc[indicesToKeep, 'principal component 2']
			       , c = color
			       , s = 50)
		ax.legend(targets)
		ax.grid()

		return pca.explained_variance_ratio_
