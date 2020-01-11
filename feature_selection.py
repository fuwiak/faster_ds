


#filter methods




#wrapper methods


#embedded methods


def xgb_reg_feat_importances(X,y, ratio=0.3,random_state=42):
	import pandas as pd
	from sklearn.model_selection import train_test_split

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=ratio, random_state)
	
	import xgboost as xgb
	clf = xgb.XGBRegressor(objective="reg:linear", random_state)
	clf = clf.fit(train_X, train_y)
	feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
	feat_importances.nlargest(5).plot(kind='barh')
	feat = [feat_importances.nlargest(5).axes[0][i] for i in range(len(feat_importances.nlargest(5).axes[0]))]

	return feat


def fast_regression_pipelines(X,y, ratio=0.3,random_state=42):
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	from sklearn.pipeline import Pipeline
	from sklearn.externals import joblib
	from sklearn.linear_model import LogisticRegression

	from sklearn import tree

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)




	# Construct some pipelines
	pipe_lr = Pipeline([('scl', StandardScaler()),
	            ('pca', PCA(n_components=2)),
	            ('clf', LogisticRegression(random_state=42))])

	pipe_xgb = Pipeline([('scl', StandardScaler()),
	            ('pca', PCA(n_components=2)),
	            ('clf', xgb.XGBRegressor(objective="reg:linear", random_state=42))])

	pipe_dt = Pipeline([('scl', StandardScaler()),
	            ('pca', PCA(n_components=2)),
	            ('clf', tree.DecisionTreeClassifier(random_state=42))])

	pipe_lin= Pipeline([('scl', StandardScaler()),
	            ('pca', PCA(n_components=2)),
	            ('clf', linear_model.LinearRegression())])



	pipelines = [pipe_lr, pipe_xgb, pipe_dt, pipe_lin]

	pipe_dict = {0: 'Logistic Regression', 1: 'xboost', 2: 'Decision Tree', 3:'Linear Regression'}

	# Fit the pipelines
	for pipe in pipelines:
		pipe.fit(train_X, train_y)

	# Compare accuracies
	for idx, val in enumerate(pipelines):
		print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(test_X, test_y)))

	# Identify the most accurate model on test data
	best_acc = 0.0
	best_clf = 0
	best_pipe = ''
	for idx, val in enumerate(pipelines):
		if val.score(test_X, test_y) > best_acc:
			best_acc = val.score(test_X, test_y)
			best_pipe = val
			best_clf = idx
	print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

	return pipe_dict[best_clf]


def regression_dataframe_metrics(X,y, ratio=0.3,random_state=42):
	pipelines = [pipe_lr, pipe_xgb, pipe_dt, pipe_lin]
	MLA = pipelines

	pipe_dict = {0: 'Logistic Regression', 1: 'xboost', 2: 'Decision Tree', 3:'Linear Regression'}



	MLA_columns = []
	MLA_compare = pd.DataFrame(columns = MLA_columns)
	row_index = 0
	for alg in MLA:
	    predicted = alg.fit(train_X, train_y).predict(test_X)

	    MLA_name = list(pipe_dict.values())[row_index]
	    MLA_compare.loc[row_index,' Name'] = MLA_name
	    MLA_compare.loc[row_index, 'MAE'] = metrics.mean_absolute_error(predicted, test_y)
	    MLA_compare.loc[row_index, 'MSE'] = metrics.mean_squared_error(predicted, test_y)
	    MLA_compare.loc[row_index, 'RMSE'] = np.sqrt(metrics.mean_squared_error(predicted, test_y))
	    MLA_compare.loc[row_index, 'R2'] = alg.score(X,y)
	    
	    
	    
	    
	    row_index+=1

	MLA_compare.sort_values(by = ['R2'], ascending = False, inplace = True)
	
	return MLA_compare










