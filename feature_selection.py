



# You could start with preprocessing.py
# X, y = set_X_y(df, y_name)


class FS:
	'''
	filter methods
	'''
	
	#Using Pearson Correlation
	@staticmethod
	def cor_selector(X, y, N=100):
	    cor_list = []
	    feature_name = X.columns.tolist()
	    # calculate the correlation with y for each feature
	    for i in X.columns.tolist():
		cor = np.corrcoef(X[i], y)[0, 1]
		cor_list.append(cor)
	    # replace NaN with 0
	    cor_list = [0 if np.isnan(i) else i for i in cor_list]
	    # feature name
	    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-N:]].columns.tolist()
	    # feature selection? 0 for not select, 1 for select
	    cor_support = [True if i in cor_feature else False for i in feature_name]
	    return cor_support, cor_feature

	# cor_support, cor_feature = cor_selector(X, y)
	# print(str(len(cor_feature)), 'selected features')
	# print("list of selected columns ", cor_feature)

	#CHI2
	
	@staticmethod
	def chi2_selector(X,y, N=100):
		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2
		from sklearn.preprocessing import MinMaxScaler
		X_norm = MinMaxScaler().fit_transform(X)
		chi_selector = SelectKBest(chi2, k=100)
		chi_selector.fit(X_norm, y)
		chi_support = chi_selector.get_support()
		chi_feature = X.loc[:,chi_support].columns.tolist()
		print(str(len(chi_feature)), 'selected features')
		print("list of selected columns ", chi_feature)
		return chi_feature


	#wrapper methods


	#embedded methods

	@staticmethod
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

	@staticmethod
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

	@staticmethod
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











