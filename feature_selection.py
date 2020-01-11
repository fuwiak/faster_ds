


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


