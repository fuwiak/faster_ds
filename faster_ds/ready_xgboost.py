import matplotlib.pylab as plt

parameters = {'nthread':[6], 
              'objective':['binary:logistic'],
              'learning_rate': [0.01, 0.1],
              'max_depth': [5,8,13],
              'n_estimators': [200,500,1000,3000],
              'seed': [1337]}


def xgb_cv(**parameters, num_split=5, N=-1):
	import xgboost as xgb
	xgb_model = xgb.XGBClassifier()

	clf = GridSearchCV(xgb_model, parameters, n_jobs=N, 
	                   cv = StratifiedKFold(shuffle=True,n_splits=num_split), 
	                   scoring='accuracy',
	                   verbose=2, refit=True)

	clf.fit(scaled_X_train.values, y_train)

	return clf

def dump_to_pickle(clf, filename):
	import pickle

	pickle.dump(model, open(filename, 'wb'))

def load_from_pickle(clf, filename):
	import pickle
	loaded_model = pickle.load(open(filename, 'rb'))
	return loaded_model

def pickle_acc(clf, X_test, Y_test):

	result = clf.score(X_test, Y_test)
	return(result)

def xgb_reg(X, y, test_size=0.3, random_state=100):
	import xgboost as xgb
	clf = xgb.XGBRegressor(objective="reg:linear", random_state=random_state)
	clf = clf.fit(train_X, train_y)
	predictions = clf.predict(test_X)
	feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
	feat_importances.nlargest(5).plot(kind='barh')
	plt.xlabel("feat importances(data range(0,1))")
	plt.show()
	
	return clf.feature_importances_
	






def create_data_matrix(X,y):
	features = X
	target = y
	dmatrix = xg.DMatrix(features.values,
	                     target.values,
	                     feature_names=features.columns.values)
	# clf = xg.train(params, dmatrix)
	return dmatrix




def show_tree_importances(**parameters, data_dmatrix):



	xg_reg = xgb.train(params=parameters, dtrain=data_dmatrix, num_boost_round=10)

	import matplotlib.pyplot as plt

	xgb.plot_tree(xg_reg,num_trees=0)
	plt.rcParams['figure.figsize'] = [50, 10]
	plt.show()


	xgb.plot_importance(xg_reg)
	plt.rcParams['figure.figsize'] = [5, 5]
	plt.show()
	
