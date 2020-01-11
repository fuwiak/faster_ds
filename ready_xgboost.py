
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