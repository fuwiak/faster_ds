



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
	
	
	@staticmethod
	def RFE_selector(X_norm, y, estimator=LogisticRegression()):
		from sklearn.feature_selection import RFE
		from sklearn.linear_model import LogisticRegression
		rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
		rfe_selector.fit(X_norm, y)
		rfe_support = rfe_selector.get_support()
		rfe_feature = X.loc[:,rfe_support].columns.tolist()
		print(str(len(rfe_feature)), 'selected features')
		return rfe_feature
	
	
	@staticmethod
	def Lasso_selector(X_norm, y):
		from sklearn.feature_selection import SelectFromModel
		from sklearn.linear_model import LogisticRegression
		embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
		embeded_lr_selector.fit(X_norm, y)
		embeded_lr_support = embeded_lr_selector.get_support()
		embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
		print(str(len(embeded_lr_feature)), 'selected features')
		return embeded_lr_feature
	
	


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

	
	










