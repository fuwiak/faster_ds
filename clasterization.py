from sklearn import preprocessing  # to normalise existing X
from sklearn import cluster
X_Norm = preprocessing.normalize(DF)

km2 = cluster.KMeans(n_clusters=5,init='random').fit(X_Norm)

labels = km2.labels_
labels = np.unique(labels)

km2.cluster_centers_
plt.scatter(X_Norm[:,0],X_Norm[:,2], c=km2.labels_, cmap='rainbow')
